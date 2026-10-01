"""
Render demo screenshots and a preview video from the running NIDS app.

    cd T014_Project
    python webapp/record_demo.py

The real server, model and defence run; the real phone web page (attacker.html)
is loaded in an embedded browser and clicked through. Frames are rendered
offscreen (no desktop screen recording) and encoded with a bundled ffmpeg.

Outputs: docs/screenshots/*.png and docs/app_demo_preview.mp4
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu --no-sandbox")
os.environ.setdefault("NIDS_PORT", "8011")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtCore import QCoreApplication, QRectF, Qt, QUrl
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication

import app
import server

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
SHOTS = DOCS / "screenshots"
FRAMES = DOCS / "_frames"
VIDEO = DOCS / "app_demo_preview.mp4"
FPS = 8
PHONE_W, PHONE_H = 390, 844
CANVAS_W, CANVAS_H = 2060, 1060

QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
qa = QApplication(sys.argv)
qa.setStyle("Fusion")
qa.setStyleSheet(app.STYLE)

server.start_server()
threading.Thread(target=server.load_engine, daemon=True).start()
win = app.Main()
win.resize(1600, 1060)
win.show()

phone = QWebEngineView()
phone.setFixedSize(PHONE_W, PHONE_H)
phone.show()

frame_no = 0
caption = ""


def compose():
    """Dashboard on the left, the phone page on the right, caption + footer."""
    canvas = QPixmap(CANVAS_W, CANVAS_H)
    canvas.fill(QColor("#eef1f6"))
    p = QPainter(canvas)
    p.setRenderHint(QPainter.Antialiasing)
    p.drawPixmap(0, 0, win.grab())
    x = 1600 + 30
    p.setPen(QColor("#6a7588"))
    p.setFont(QFont("Noto Sans", 11, QFont.Bold))
    p.drawText(x, 60, "PHONE (browser page)")
    p.setPen(QColor("#c9d2e0"))
    p.drawRoundedRect(x - 3, 77, PHONE_W + 6, PHONE_H + 6, 10, 10)
    p.drawPixmap(x, 80, phone.grab())
    p.setPen(QColor("#1b2333"))
    p.setFont(QFont("Noto Sans", 12, QFont.Bold))
    p.drawText(QRectF(x, 934, PHONE_W, 84), Qt.TextWordWrap, caption)
    p.setPen(QColor("#6a7588"))
    p.setFont(QFont("Noto Sans", 9))
    p.drawText(QRectF(1610, 1026, 440, 30), Qt.TextWordWrap,
               "Rendered from the running app (offscreen). Synthetic stand-in model/data.")
    p.end()
    return canvas


def pump(seconds, cap=None, until=None):
    """Keep the app running for `seconds` while writing video frames."""
    global frame_no, caption
    if cap is not None:
        caption = cap
    end = time.time() + seconds
    step = 1.0 / FPS
    while time.time() < end:
        t0 = time.time()
        qa.processEvents()
        compose().save(str(FRAMES / f"{frame_no:05d}.jpg"), "JPG", 88)
        frame_no += 1
        if until and until():
            break
        wait = step - (time.time() - t0)
        if wait > 0:
            end_wait = time.time() + wait
            while time.time() < end_wait:
                qa.processEvents()
                time.sleep(0.01)


def shot(name, phone_only=False):
    qa.processEvents()
    pm = phone.grab() if phone_only else compose()
    pm.save(str(SHOTS / name), "PNG")
    print("saved", name)


def js(code):
    phone.page().runJavaScript(code)


def wait_for(cond, timeout):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        qa.processEvents()
        time.sleep(0.05)
    return cond()


for d in (SHOTS, FRAMES):
    d.mkdir(parents=True, exist_ok=True)
for f in FRAMES.glob("*.jpg"):
    f.unlink()

if not wait_for(lambda: server.model is not None, 180):
    sys.exit(f"Model did not load: {server.model_error}")

server.config.update(sim=True, sim_rate=4, sim_interval=22, strikes_to_block=3)

# 1. idle dashboard with simulated background network
pump(9, "1. NIDS running. Background network is simulated (rows tagged SIM): mostly normal traffic.")
shot("01_dashboard_normal_traffic.png")

# 2. phone opens the page
phone.load(QUrl(server.phone_url().replace(server.lan_ip(), "127.0.0.1")))
wait_for(lambda: False, 0.1)
pump(4, "2. The phone opens the page (scan the QR code). It shows: connected, model ready.")
shot("02_phone_connected.png", phone_only=True)

# 3. phone sends normal traffic
js("run(0)")
pump(6, "3. Phone sends NORMAL traffic. The model says BENIGN, nothing is blocked.")

# 4. phone launches a DDoS burst -> detected -> blocked
blocked_before = len(server.blocked)
js("run(2)")
wait_for(lambda: len(server.blocked) > blocked_before, 15)
pump(5, "4. Phone launches a DDoS burst. After 3 alerts the PC blocks the phone (HTTP 403).")
shot("03_attack_detected_and_blocked.png")
shot("04_phone_blocked.png", phone_only=True)
pump(3, "4. The phone is now blocked; further requests are dropped.")

# 5. a random attack from a simulated IP
sim_attack = lambda: any(e["src"] == "SIM" and e["level"] in ("ALERT", "BLOCK") for e in server.access_log)
pump(40, "5. Waiting for a random attack from a simulated IP...", until=sim_attack)
pump(4, "5. A random attacker (SIM) is detected by the model and blocked automatically.")
shot("05_random_attack_blocked.png")
pump(3, "Top sources, logs, verdicts and research results stay on one dashboard.")

print("frames:", frame_no)
import imageio_ffmpeg

subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", str(FPS),
                "-i", str(FRAMES / "%05d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24",
                str(VIDEO)], check=True)
for f in FRAMES.glob("*.jpg"):
    f.unlink()
FRAMES.rmdir()
logs = Path(__file__).resolve().parent / "logs"
if logs.exists():
    import shutil
    shutil.rmtree(logs)
print("video:", VIDEO, round(VIDEO.stat().st_size / 1e6, 2), "MB")
os._exit(0)
