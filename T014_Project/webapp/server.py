"""
Detection engine + phone-facing HTTP server for the NIDS desktop app.

The PySide6 app (webapp/app.py) imports this module, calls load_engine() and
start_server(), and reads the shared state directly. The phone (same Wi-Fi)
opens http://<PC-LAN-IP>:8000/ and sends traffic to /api/send.

Standard library only for the HTTP part. TensorFlow is used for the model.

HONESTY NOTE: the phone's HTTP requests are real, but the model classifies
CICIDS2017 flow-feature sequences sampled from the test set for the traffic
type chosen on the phone. A browser cannot generate those flow features.
"""

import collections
import csv
import itertools
import json
import http.client
import os
import random
import re
import socket
import threading
import select
import time
from urllib.parse import urlsplit
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_PATH = PROJECT_ROOT / "models" / "cnn_gru_top40_final.keras"
X_PATH = PROJECT_ROOT / "data" / "processed" / "sequences_class" / "test_top_40_X.npy"
Y_PATH = PROJECT_ROOT / "data" / "processed" / "sequences_class" / "test_top_40_y.npy"

PORT = int(os.environ.get("NIDS_PORT", "8000"))
PROXY_PORT = int(os.environ.get("NIDS_PROXY_PORT", "8080"))
CLASS_NAMES = ["BENIGN", "DoS Hulk", "DDoS", "PortScan",
               "DoS GoldenEye", "FTP-Patator", "SSH-Patator"]
# A test set this small can only be the synthetic stand-in data.
SYNTHETIC_MAX_ROWS = 1000

# ------------------------------------------------------------
# Model + data
# ------------------------------------------------------------
model = None
X = y = None
class_indices = {}
model_error = None
is_synthetic = False


def load_engine():
    """Load the .keras model and test sequences. Slow (TensorFlow import)."""
    global model, X, y, class_indices, model_error, is_synthetic
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        X = np.load(X_PATH).astype("float32")
        y = np.load(Y_PATH)
        class_indices = {c: np.where(y == c)[0] for c in range(7)}
        # A test set this small can only be the synthetic stand-in data.
        is_synthetic = len(X) < SYNTHETIC_MAX_ROWS
        model_error = None
    except Exception as exc:
        model_error = f"{type(exc).__name__}: {exc}"
    return model is not None


# ------------------------------------------------------------
# Live state
# ------------------------------------------------------------
lock = threading.Lock()
events = []           # newest last
blocked = {}          # ip -> {"since": ts, "reason": str}
strikes = {}          # ip -> consecutive malicious detections
config = {"defence": True, "threshold": 0.70, "strikes_to_block": 3,
          "proxy": True, "phone": True, "sim": False, "sim_rate": 3, "sim_interval": 30}
sources = {}          # ip -> {"origin", "flows", "attacks", "dropped"}
counters = {"total": 0, "attacks": 0, "blocked_requests": 0, "by_class": [0] * 7}

# Every real HTTP request that reaches this server (also written to logs/traffic.log)
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_FIELDS = ["id", "time", "src", "ip", "port", "method", "path", "status",
              "rx", "tx", "ms", "level", "note", "ua"]
access_log = collections.deque(maxlen=50000)
_log_ids = itertools.count(1)
_log_file = None


def record_request(entry):
    """Append to the in-memory log and to logs/traffic.log."""
    global _log_file
    entry["id"] = next(_log_ids)
    access_log.append(entry)
    try:
        if _log_file is None:
            LOG_DIR.mkdir(exist_ok=True)
            _log_file = open(LOG_DIR / "traffic.log", "a", buffering=1)
        _log_file.write(" | ".join(str(entry[k]) for k in LOG_FIELDS) + "\n")
    except OSError:
        pass


def now():
    return time.strftime("%H:%M:%S")


def classify(class_id):
    idx = int(random.choice(class_indices[class_id]))
    probs = model(X[idx:idx + 1], training=False).numpy()[0]
    pred = int(np.argmax(probs))
    return idx, pred, float(probs[pred]), [float(p) for p in probs]


def handle_send(ip, class_id, origin="LIVE"):
    with lock:
        src = sources.setdefault(ip, {"origin": origin, "flows": 0, "attacks": 0, "dropped": 0})
        if ip in blocked:
            src["dropped"] += 1
            counters["blocked_requests"] += 1
            events.append({"t": now(), "ip": ip, "kind": "dropped",
                           "sent": CLASS_NAMES[class_id], "pred": None,
                           "conf": None, "action": "REQUEST DROPPED (403)"})
            return 403, {"status": "blocked",
                         "message": "Blocked by NIDS: your IP was flagged.",
                         "_level": "BLOCK",
                         "_note": f"{CLASS_NAMES[class_id]} flow dropped - source is blocked"}

        idx, pred, conf, probs = classify(class_id)
        malicious = pred != 0 and conf >= config["threshold"]
        counters["total"] += 1
        counters["by_class"][pred] += 1
        src["flows"] += 1
        action = "allowed"
        if malicious:
            counters["attacks"] += 1
            src["attacks"] += 1
            strikes[ip] = strikes.get(ip, 0) + 1
            action = (f"ALERT {strikes[ip]}/{config['strikes_to_block']}" if config["defence"]
                      else f"ALERT {strikes[ip]} (defence off)")
            if config["defence"] and strikes[ip] >= config["strikes_to_block"]:
                blocked[ip] = {"since": now(),
                               "reason": f"{CLASS_NAMES[pred]} detected"}
                action = "SOURCE BLOCKED"
        else:
            strikes[ip] = 0

        events.append({"t": now(), "ip": ip, "kind": "attack" if malicious else "normal",
                       "sent": CLASS_NAMES[class_id], "pred": CLASS_NAMES[pred],
                       "conf": round(conf * 100, 2), "action": action,
                       "sample": idx, "probs": probs})
        del events[:-300]
        level = "BLOCK" if action == "SOURCE BLOCKED" else ("ALERT" if malicious else "INFO")
        return 200, {"status": "blocked" if ip in blocked else "ok",
                     "predicted": CLASS_NAMES[pred],
                     "confidence": round(conf * 100, 2), "action": action,
                     "_level": level, "_probs": probs,
                     "_note": (f"{CLASS_NAMES[class_id]} -> {CLASS_NAMES[pred]} "
                               f"{conf * 100:.2f}% [{action}] #{idx}")}


# ------------------------------------------------------------
# Real results loaded from results/
# ------------------------------------------------------------
def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_results():
    out = {}
    out["features"] = read_csv(RESULTS_DIR / "final" / "cnn_gru_feature_comparison.csv")
    out["loao"] = read_csv(RESULTS_DIR / "final" / "loao_results.csv")
    out["anova"] = read_csv(PROJECT_ROOT / "data" / "processed" / "feature_selection"
                            / "anova_feature_ranking.csv")[:20]
    out["class_weights"] = read_csv(PROJECT_ROOT / "data" / "processed"
                                    / "class_weights" / "class_weights.csv")
    text = (RESULTS_DIR / "protocol_check" / "deep_leakage_check_top30.txt").read_text()
    def grab(pattern):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None
    out["leakage"] = {
        "train_rows": int(grab(r"Training shape: \((\d+)")),
        "test_rows": int(grab(r"Testing shape : \((\d+)")),
        "common_rows": int(grab(r"Common feature rows: (\d+)")),
        "train_overlap_pct": grab(r"Training overlap percentage: ([\d.]+)"),
        "test_overlap_pct": grab(r"Testing overlap percentage: ([\d.]+)"),
        "mismatch": int(grab(r"label mismatch: (\d+)")),
    }
    rep = (RESULTS_DIR / "cnn_gru_top40_report.txt").read_text()
    out["support"] = {n: int(s) for n, s in re.findall(
        r"^\s*(BENIGN|DoS Hulk|DDoS|PortScan|DoS GoldenEye|FTP-Patator|SSH-Patator)"
        r"\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+(\d+)", rep, re.M)}
    return out


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css",
                 ".js": "application/javascript", ".svg": "image/svg+xml"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _begin(self):
        self._t0 = time.perf_counter()
        self._rx = 0

    def _record(self, code, tx, level=None, note="", probs=None):
        route = self.path.split("?")[0]
        if level is None:
            level = "DEBUG" if route == "/api/ping" else ("WARN" if code >= 400 else "INFO")
        record_request({
            "time": time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}",
            "ip": self.client_address[0], "port": self.client_address[1],
            "method": self.command, "path": route, "status": code,
            "rx": getattr(self, "_rx", 0), "tx": tx,
            "ms": round((time.perf_counter() - getattr(self, "_t0", time.perf_counter())) * 1000, 1),
            "level": level, "note": note, "probs": probs, "src": "LIVE",
            "ua": self.headers.get("User-Agent", "-")})

    def _json(self, code, obj):
        level = note = probs = None
        if isinstance(obj, dict):
            level, note = obj.pop("_level", None), obj.pop("_note", "")
            probs = obj.pop("_probs", None)
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
        self._record(code, len(body), level, note or "", probs)

    def _file(self, name):
        path = (STATIC_DIR / name).resolve()
        if STATIC_DIR not in path.parents or not path.is_file():
            self._json(404, {"error": "not found"})
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPES.get(path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self._record(200, len(body), None, f"static {name}")

    def do_GET(self):
        self._begin()
        route = self.path.split("?")[0]
        if route in ("/", "/attacker"):
            return self._file("attacker.html")
        if route.startswith("/static/"):
            return self._file(route[len("/static/"):])
        if route == "/api/ping":
            return self._json(200, {"model_ready": model is not None,
                                    "blocked": self.client_address[0] in blocked})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        self._begin()
        length = int(self.headers.get("Content-Length", 0) or 0)
        self._rx = length
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._json(400, {"error": "bad json"})
        route = self.path.split("?")[0]
        ip = self.client_address[0]

        if route == "/api/send":
            if not config["phone"]:
                return self._json(503, {"error": "phone page is turned off"})
            if model is None:
                return self._json(503, {"error": "model not loaded", "detail": model_error})
            cid = data.get("class_id")
            if not isinstance(cid, int) or not 0 <= cid <= 6:
                return self._json(400, {"error": "class_id must be 0-6"})
            code, body = handle_send(ip, cid)
            return self._json(code, body)
        if route == "/api/config":
            with lock:
                if "defence" in data:
                    config["defence"] = bool(data["defence"])
                if "strikes_to_block" in data:
                    config["strikes_to_block"] = max(1, min(20, int(data["strikes_to_block"])))
            return self._json(200, config)
        if route == "/api/unblock":
            with lock:
                if data.get("ip") == "*":
                    blocked.clear()
                    strikes.clear()
                else:
                    blocked.pop(data.get("ip"), None)
                    strikes.pop(data.get("ip"), None)
            return self._json(200, {"ok": True})
        if route == "/api/reset":
            with lock:
                events.clear(); blocked.clear(); strikes.clear(); sources.clear(); access_log.clear()
                counters.update(total=0, attacks=0, blocked_requests=0, by_class=[0] * 7)
            return self._json(200, {"ok": True})
        self._json(404, {"error": "not found"})


# ------------------------------------------------------------
# Forward proxy: real client traffic passes through the NIDS
# ------------------------------------------------------------
# Set a device's Wi-Fi proxy to <PC-IP>:PROXY_PORT. Every request is forwarded
# for real (HTTP and HTTPS CONNECT tunnels). Blocked clients get 403.
# The model needs CICIDS2017 flow features, which a proxy cannot measure, so
# the client's behaviour (request rate / number of distinct destinations) picks
# the traffic type, and a matching test-set flow is classified by the model.
PROXY_WINDOW = 2.0
PROXY_HULK_RATE = 40      # requests in window -> DoS Hulk-like
PROXY_DDOS_RATE = 80      # requests in window -> DDoS-like
PROXY_SCAN_TARGETS = 8    # distinct host:port in window -> PortScan-like
_proxy_hist = {}          # ip -> deque[(time, host:port)]


def proxy_behaviour(ip, target):
    t = time.time()
    with lock:
        h = _proxy_hist.setdefault(ip, collections.deque())
        h.append((t, target))
        while h and t - h[0][0] > PROXY_WINDOW:
            h.popleft()
        n, targets = len(h), len({x[1] for x in h})
    if n >= PROXY_DDOS_RATE:
        return 2
    if n >= PROXY_HULK_RATE:
        return 1
    if targets >= PROXY_SCAN_TARGETS:
        return 3
    return 0


HOP = {"proxy-connection", "connection", "keep-alive", "te", "trailers",
       "transfer-encoding", "upgrade", "proxy-authorization", "proxy-authenticate"}


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _gate(self, target):
        """Classify this request; returns (allowed, note, level, probs)."""
        ip = self.client_address[0]
        if model is None:
            return True, "model not loaded - not inspected", "WARN", None
        code, body = handle_send(ip, proxy_behaviour(ip, target))
        return (code != 403, body.get("_note", ""), body.get("_level", "INFO"),
                body.pop("_probs", None))

    def _log(self, code, tx, note, level, probs, ms):
        record_request({
            "time": time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}",
            "src": "LIVE", "ip": self.client_address[0], "port": self.client_address[1],
            "method": self.command, "path": self.path[:120], "status": code,
            "rx": 0, "tx": tx, "ms": ms, "level": level, "note": "proxy: " + note,
            "probs": probs, "ua": self.headers.get("User-Agent", "-")})

    def _deny(self, t0, note, level, probs):
        body = b"Blocked by NIDS: your IP was flagged.\n"
        self.send_response(403)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True
        self._log(403, len(body), note, level, probs, round((time.perf_counter() - t0) * 1000, 1))

    def _off(self):
        body = b"NIDS proxy is turned off.\n"
        self.send_response(503)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def do_CONNECT(self):
        if not config["proxy"]:
            return self._off()
        t0 = time.perf_counter()
        ok, note, level, probs = self._gate(self.path)
        if not ok:
            return self._deny(t0, note, level, probs)
        host, _, port = self.path.partition(":")
        try:
            up = socket.create_connection((host, int(port or 443)), timeout=10)
        except OSError as exc:
            self.send_error(502, str(exc))
            return self._log(502, 0, note, "WARN", probs, 0)
        self.send_response(200, "Connection Established")
        self.end_headers()
        self._log(200, 0, note, level, probs, round((time.perf_counter() - t0) * 1000, 1))
        self.close_connection = True
        socks = [self.connection, up]
        try:
            while True:
                r, _, _ = select.select(socks, [], [], 30)
                if not r:
                    break
                done = False
                for a in r:
                    data = a.recv(65536)
                    if not data:
                        done = True
                        break
                    (up if a is self.connection else self.connection).sendall(data)
                if done:
                    break
        except OSError:
            pass
        finally:
            up.close()

    def _forward(self):
        if not config["proxy"]:
            return self._off()
        t0 = time.perf_counter()
        u = urlsplit(self.path)
        if not u.hostname:      # not proxy-style request
            self.send_error(400, "This is a proxy port; use it as a Wi-Fi HTTP proxy")
            return
        port = u.port or 80
        ok, note, level, probs = self._gate(f"{u.hostname}:{port}")
        length = int(self.headers.get("Content-Length", 0) or 0)
        payload = self.rfile.read(length) if length else None
        if not ok:
            return self._deny(t0, note, level, probs)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        try:
            c = http.client.HTTPConnection(u.hostname, port, timeout=15)
            c.request(self.command, (u.path or "/") + (f"?{u.query}" if u.query else ""),
                      payload, headers)
            r = c.getresponse()
            body = r.read()
            self.send_response(r.status, r.reason)
            for k, v in r.getheaders():
                if k.lower() not in HOP and k.lower() != "content-length":
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            c.close()
            self._log(r.status, len(body), note, level, probs,
                      round((time.perf_counter() - t0) * 1000, 1))
        except (OSError, http.client.HTTPException) as exc:
            self.send_error(502, str(exc))
            self._log(502, 0, note, "WARN", probs, round((time.perf_counter() - t0) * 1000, 1))

    do_GET = do_POST = do_PUT = do_DELETE = do_HEAD = do_OPTIONS = do_PATCH = _forward


# ------------------------------------------------------------
# Background network (clearly labelled SIM in the log)
# ------------------------------------------------------------
# RFC 5737 documentation ranges: guaranteed never to be real hosts.
SIM_NETS = ["203.0.113.", "198.51.100.", "192.0.2."]


def rand_sim_ip():
    return random.choice(SIM_NETS) + str(random.randint(2, 254))


def _sim_send(ip, class_id):
    t0 = time.perf_counter()
    code, body = handle_send(ip, class_id, origin="SIM")
    record_request({
        "time": time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}",
        "src": "SIM", "ip": ip, "port": random.randint(1024, 65535),
        "method": "FLOW", "path": "(flow)", "status": code, "rx": 0, "tx": 0,
        "ms": round((time.perf_counter() - t0) * 1000, 1),
        "level": body.pop("_level", "INFO"), "note": body.pop("_note", ""),
        "probs": body.pop("_probs", None), "ua": "network traffic generator"})


def sim_loop():
    """Mostly normal traffic from regular hosts (some chattier than others);
    every so often a fresh random IP runs a short attack burst."""
    hosts = weights = None
    episode = None
    next_benign = next_episode = 0.0
    was_on = False
    while True:
        try:
            if not config["sim"] or model is None:
                was_on = False
                time.sleep(0.3)
                continue
            now = time.time()
            if not was_on:
                was_on = True
                hosts = list({rand_sim_ip() for _ in range(30)})
                weights = [1 / (i + 1) ** 0.8 for i in range(len(hosts))]
                next_benign = now
                next_episode = now + random.uniform(0.5, 1.0) * config["sim_interval"]
                episode = None

            if episode is None and now >= next_episode:
                episode = {"ip": rand_sim_ip(), "cls": random.randint(1, 6),
                           "left": random.randint(5, 14), "next": now}
                next_episode = now + random.uniform(0.6, 1.4) * config["sim_interval"]

            if episode and now >= episode["next"]:
                if episode["ip"] in blocked and random.random() < 0.5:
                    episode = None          # attacker gives up once blocked
                else:
                    _sim_send(episode["ip"], episode["cls"])
                    episode["left"] -= 1
                    episode["next"] = now + random.uniform(0.25, 0.9)
                    if episode["left"] <= 0:
                        episode = None

            if now >= next_benign:
                _sim_send(random.choices(hosts, weights)[0], 0)
                next_benign = now + random.expovariate(max(0.2, float(config["sim_rate"])))
            time.sleep(0.02)
        except Exception:
            time.sleep(0.5)


def start_server():
    """Start the phone-facing HTTP server on a daemon thread."""
    httpd = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        proxy = ThreadingHTTPServer(("0.0.0.0", PROXY_PORT), ProxyHandler)
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
    except OSError:
        pass                    # proxy port busy: the rest of the app still works
    threading.Thread(target=sim_loop, daemon=True).start()
    return httpd


def proxy_addr():
    return f"{lan_ip()}:{PROXY_PORT}"


def phone_url():
    return f"http://{lan_ip()}:{PORT}/"
