"""
attendance_web.py
Web-based Real-time Face Recognition Attendance System using InsightFace + IP Camera (RTSP)
Now with Embedding Caching System for fast DB loading
Converted to a basic web service using Flask.
Assumptions:
- Students' images are organized in subfolders under "data/", e.g., data/BICE21_Section_A_OODP/ containing student folders like "sid_name/" with images.
- Hardcoded teachers, classes, classrooms for simplicity.
- Basic login (plain passwords - use hashing in production).
- One session at a time.
- Video streamed to browser with annotations.
- Manual end session, preview attendance, then confirm to save.
"""
import os
import glob
import cv2
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from insightface.app import FaceAnalysis
import threading
import signal
import sys
from typing import List
from flask import Flask, render_template_string, request, redirect, url_for, Response, session

# ==== Settings ====
STUDENTS_DIR = "data"
ABSENT_TXT_PREFIX = "absent_list_"
OUTPUT_CSV_PREFIX = "attendance_live_"
SIM_THRESHOLD = 0.25
MIN_PRESENT_TIME = 30  # seconds required to be marked present
PROVIDERS = ["CPUExecutionProvider"]
CAP_RECONNECT_DELAY = 3.0
CAP_READ_TIMEOUT = 5.0

# Hardcoded data
TEACHERS = {"teacher1": "password1"}  # username: password (plain for basic - hash in prod)
CLASSES = [
    "BICE21 Section A | OODP",
    "BICE21 Section B | OODP",
    "BICE22 Section A | C Programming",
    "BICE22 Section B | C Programming"
]
CLASSROOMS = {
    "301": "rtsp://admin:password@ipAddress:port/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif",
    "302": "rtsp://admin:password@ipAddress:port/cam/realmonitor?channel=1&subtype=0",
    "303": "rtsp://admin:password@10.10.50.171:554/cam/realmonitor?channel=1&subtype=0",
    "304": "rtsp://admin:password@10.10.50.172:554/cam/realmonitor?channel=1&subtype=0"
}  # Replace with actual RTSP URLs

# Global FaceAnalysis
face_app = None

# Globals for current session
current_cap = None
current_attendance = None
current_db_embeds = None
current_db_meta = None
current_shutdown_flag = False
current_frame = None
frame_lock = threading.Lock()
current_class = None
processing_thread = None

# Flask app
webapp = Flask(__name__)
webapp.secret_key = "super_secret_key"  # Change in production

# ==== FaceAnalysis Setup ====
def build_face_app():
    print("🔧 Building FaceAnalysis model...")
    app = FaceAnalysis(name="buffalo_l", providers=PROVIDERS)
    app.prepare(ctx_id=0, det_size=(640, 640))
    print(f"✅ Loaded FaceAnalysis models: {[m for m in app.models.keys()]}")
    return app

# ==== Student class ====
@dataclass
class Student:
    sid: str
    name: str
    folder: str

# ==== Load roster ====
def load_roster_from_folders(students_dir: str) -> List[Student]:
    students = []
    for folder in sorted(os.listdir(students_dir)):
        fpath = os.path.join(students_dir, folder)
        if not os.path.isdir(fpath):
            continue
        if "_" in folder:
            sid, name = folder.split("_", 1)
        else:
            sid, name = folder, folder
        students.append(Student(sid=sid.strip(), name=name.strip(), folder=fpath))
    print(f"✅ Loaded roster: {len(students)} students from {students_dir}")
    return students

# ==== Normalize ====
def l2_normalize(x, axis=1, eps=1e-10):
    return x / np.clip(np.sqrt(np.sum(x**2, axis=axis, keepdims=True)), eps, None)

# ==== Compute embedding ====
def compute_face_embedding(app: FaceAnalysis, img_bgr):
    faces = app.get(img_bgr)
    if len(faces) == 0:
        return None
    f = max(faces, key=lambda z: float(z.det_score))
    emb = get_emb(f)
    if emb is None:
        return None
    emb = np.array(emb, dtype=np.float32).reshape(1, -1)
    emb = l2_normalize(emb, axis=1)
    return emb

def get_emb(f):
    if hasattr(f, "normed_embedding") and f.normed_embedding is not None:
        return f.normed_embedding
    elif hasattr(f, "embedding") and f.embedding is not None:
        return f.embedding
    return None

# ==== Build DB with caching ====
def build_embedding_db(app: FaceAnalysis, roster: List[Student], cache_file):
    if os.path.exists(cache_file):
        print("⚡ Loading cached embeddings...")
        data = np.load(cache_file, allow_pickle=True)
        return data["embeds"], data["meta"].tolist()
    db_vecs, meta = [], []
    for st in tqdm(roster, desc="Building DB (first time)"):
        img_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            img_paths += glob.glob(os.path.join(st.folder, ext))
        if not img_paths:
            continue
        embs = []
        for p in img_paths:
            img = cv2.imread(p)
            if img is None:
                continue
            e = compute_face_embedding(app, img)
            if e is not None:
                embs.append(e)
        if not embs:
            continue
        E = np.vstack(embs)
        mean_emb = l2_normalize(np.mean(E, axis=0, keepdims=True), axis=1)
        db_vecs.append(mean_emb)
        meta.append(st)
    db_vecs = np.vstack(db_vecs).astype(np.float32)
    # save cache
    np.savez(cache_file, embeds=db_vecs, meta=np.array(meta, dtype=object))
    print(f"✅ Embeddings cached to {cache_file}")
    return db_vecs, meta

# ==== Get students dir for class ====
def get_students_dir(class_name):
    dir_name = class_name.replace(" | ", "_").replace(" ", "_")
    return os.path.join(STUDENTS_DIR, dir_name)

# ==== Sanitize filename for cache ====
def sanitize_filename(name):
    return name.replace(" | ", "_").replace(" ", "_").replace("|", "_")

# ==== RTSP Capture Thread ====
class RTSPCapture:
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.last_frame = None
        self.last_frame_time = 0.0
        self.running = False
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _open(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        return self.cap.isOpened()

    def _run(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                if not self._open():
                    time.sleep(CAP_RECONNECT_DELAY)
                    continue
            ret, frame = self.cap.read()
            if not ret or frame is None:
                try:
                    self.cap.release()
                except:
                    pass
                self.cap = None
                time.sleep(CAP_RECONNECT_DELAY)
                continue
            with self.lock:
                self.last_frame = frame
                self.last_frame_time = time.time()
            time.sleep(0.01)

    def read(self, timeout=CAP_READ_TIMEOUT):
        start = time.time()
        while time.time() - start < timeout:
            with self.lock:
                if self.last_frame is not None:
                    return True, self.last_frame.copy()
            time.sleep(0.01)
        return False, None

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass

# ==== Processing Loop ====
def process_loop():
    global current_shutdown_flag, current_frame, current_attendance, face_app
    while not current_shutdown_flag:
        got, frame = current_cap.read()
        if not got or frame is None:
            with frame_lock:
                current_frame = None
            time.sleep(0.1)
            continue
        now = time.time()
        faces = face_app.get(frame)
        detected_ids = set()
        for f in faces:
            emb = get_emb(f)
            if emb is None:
                continue
            emb = np.array(emb, np.float32).reshape(1, -1)
            emb = l2_normalize(emb, axis=1)
            sims = np.dot(current_db_embeds, emb.T).flatten()
            if len(sims) == 0:
                continue
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim >= SIM_THRESHOLD:
                st = current_db_meta[best_idx]
                detected_ids.add(st.sid)
                info = current_attendance[st.sid]
                if info["last_seen"] is None:
                    info["last_seen"] = now
                else:
                    elapsed = now - info["last_seen"]
                    if elapsed < 5.0:
                        info["present_time"] += elapsed
                    info["last_seen"] = now
                x1, y1, x2, y2 = map(int, f.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{st.name} ({st.sid}) {best_sim:.2f}"
                cv2.putText(frame, label, (x1, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                x1, y1, x2, y2 = map(int, f.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Unknown {best_sim:.2f}", (x1, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        with frame_lock:
            current_frame = frame.copy()
        time.sleep(0.01)

# ==== Routes ====
@webapp.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in TEACHERS and TEACHERS[username] == password:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            return 'Invalid credentials'
    return '''
    <form method="post">
        Username: <input type="text" name="username"><br>
        Password: <input type="password" name="password"><br>
        <input type="submit">
    </form>
    '''

@webapp.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    classes_html = '<ul>'
    for c in CLASSES:
        classes_html += f'<li><a href="{url_for("select_room", class_name=c)}">{c}</a></li>'
    classes_html += '</ul>'
    return f'<h1>Welcome {session["username"]}</h1>{classes_html}'

@webapp.route('/select_room', methods=['GET', 'POST'])
def select_room():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    class_name = request.args.get('class_name')
    if not class_name or class_name not in CLASSES:
        return 'Invalid class', 400
    students_dir = get_students_dir(class_name)
    if not os.path.exists(students_dir):
        return f'Error: Student data folder not found for {class_name} at {students_dir}', 400
    if request.method == 'POST':
        room = request.form.get('room')
        rtsp = CLASSROOMS.get(room)
        if rtsp:
            start_session(class_name, rtsp)
            return redirect(url_for('attendance_page'))
        else:
            return 'Invalid room', 400
    rooms_html = '<form method="post"><select name="room">'
    for r in CLASSROOMS:
        rooms_html += f'<option value="{r}">{r}</option>'
    rooms_html += '</select><input type="submit" value="Start"></form>'
    return f'<h1>Select Classroom for {class_name}</h1>{rooms_html}'

def start_session(class_name, rtsp_url):
    global current_cap, current_attendance, current_db_embeds, current_db_meta, current_shutdown_flag, current_class, processing_thread, face_app
    if face_app is None:
        face_app = build_face_app()
    current_shutdown_flag = False
    current_class = class_name
    students_dir = get_students_dir(class_name)
    roster = load_roster_from_folders(students_dir)
    if not roster:
        raise ValueError(f"No students found in {students_dir}")
    cache_file = f"embeddings_{sanitize_filename(class_name)}.npz"
    current_db_embeds, current_db_meta = build_embedding_db(face_app, roster, cache_file)
    current_attendance = {
        st.sid: {"name": st.name, "present_time": 0.0, "last_seen": None}
        for st in current_db_meta
    }
    current_cap = RTSPCapture(rtsp_url)
    current_cap.start()
    processing_thread = threading.Thread(target=process_loop, daemon=True)
    processing_thread.start()

@webapp.route('/attendance_page')
def attendance_page():
    if not session.get('logged_in') or current_class is None:
        return redirect(url_for('login'))
    return render_template_string('''
    <h1>Attendance for {{ current_class }}</h1>
    <img src="{{ url_for('video_feed') }}" width="960" height="540">
    <form method="post" action="{{ url_for('end_session') }}">
        <input type="submit" value="End Session">
    </form>
    ''', current_class=current_class)

@webapp.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    global current_frame, current_shutdown_flag
    while not current_shutdown_flag:
        with frame_lock:
            if current_frame is None:
                blank = np.zeros((540, 960, 3), np.uint8)
                cv2.putText(blank, "Waiting for RTSP...", (20, 270),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', blank)
                frame_bytes = buffer.tobytes()
            else:
                ret, buffer = cv2.imencode('.jpg', current_frame)
                frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

@webapp.route('/end_session', methods=['POST'])
def end_session():
    global current_shutdown_flag, processing_thread, current_cap, current_attendance
    current_shutdown_flag = True
    if processing_thread is not None:
        processing_thread.join(timeout=5.0)
    if current_cap is not None:
        current_cap.stop()
    # Compute summary
    total = len(current_attendance)
    present_count = sum(1 for info in current_attendance.values() if info["present_time"] >= MIN_PRESENT_TIME)
    absent_count = total - present_count
    summary = f"Total: {total} Present: {present_count} Absent: {absent_count}<br>"
    present_list = "<h2>Present</h2><ul>"
    for sid, info in current_attendance.items():
        if info["present_time"] >= MIN_PRESENT_TIME:
            present_list += f"<li>{sid} {info['name']}</li>"
    present_list += "</ul>"
    absent_list = "<h2>Absent</h2><ul>"
    for sid, info in current_attendance.items():
        if info["present_time"] < MIN_PRESENT_TIME:
            absent_list += f"<li>{sid} {info['name']}</li>"
    absent_list += "</ul>"
    form = '<form method="post" action="/confirm"><input type="submit" value="Confirm and Submit"></form>'
    return summary + present_list + absent_list + form

@webapp.route('/confirm', methods=['POST'])
def confirm():
    global current_attendance, current_class, current_cap, current_db_embeds, current_db_meta, current_shutdown_flag, processing_thread
    if current_attendance is None or current_class is None:
        return 'No active session'
    # Save
    class_safe = sanitize_filename(current_class)
    output_csv = f"{OUTPUT_CSV_PREFIX}{class_safe}.csv"
    absent_txt = f"{ABSENT_TXT_PREFIX}{class_safe}.txt"
    df_attend = pd.DataFrame([
        {
            "id": sid,
            "name": info["name"],
            "present_time_sec": round(info["present_time"], 1),
            "present": info["present_time"] >= MIN_PRESENT_TIME
        }
        for sid, info in current_attendance.items()
    ])
    df_attend.to_csv(output_csv, index=False)
    print(f"✅ Attendance CSV saved: {output_csv}")
    with open(absent_txt, "w") as f:
        for sid, info in current_attendance.items():
            if info["present_time"] < MIN_PRESENT_TIME:
                f.write(f"{sid}_{info['name']}\n")
    print(f"✅ Absent list saved: {absent_txt}")
    # Reset
    current_cap = None
    current_attendance = None
    current_db_embeds = None
    current_db_meta = None
    current_class = None
    processing_thread = None
    current_shutdown_flag = False
    return 'Attendance confirmed and saved.<br><a href="/dashboard">Back to Dashboard</a>'

if __name__ == "__main__":

    webapp.run(debug=True, host='0.0.0.0', port=5000)
