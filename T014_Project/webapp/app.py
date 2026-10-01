"""
NIDS Command Center - CNN-GRU intrusion detection on CICIDS2017.

Run from the project root:
    .venv/bin/python webapp/app.py

One window, one dashboard: a phone on the same Wi-Fi (QR code) sends traffic,
the CNN-GRU model classifies it live, attackers are blocked, every request is
logged, and the research results (from results/) sit alongside.
"""

import collections
import csv
import sys
import threading
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import segno
from PySide6.QtCharts import (QBarCategoryAxis, QBarSeries, QBarSet, QChart,
                              QChartView, QLineSeries, QValueAxis)
from PySide6.QtCore import (QAbstractTableModel, QMargins, QModelIndex, QPointF,
                            Qt, QTimer)
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QFileDialog, QFrame,
                               QGridLayout, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QListWidget, QMainWindow,
                               QProgressBar, QPushButton, QScrollArea,
                               QSpinBox, QTableView, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

import server

CLASS_NAMES = server.CLASS_NAMES
SHORT_NAMES = ["Norm", "Hulk", "DDoS", "Scan", "Gold", "FTP", "SSH"]
ACCENT, DANGER, OK, WARN, MUTED = "#2457d6", "#c62f3d", "#1c8a4f", "#b76e00", "#6a7588"

STYLE = """
* { font-family: "Segoe UI", "Inter", "Noto Sans", sans-serif; font-size: 12px; color: #1b2333; }
QMainWindow, QWidget#page, QScrollArea { background: #f5f7fa; border: none; }
QFrame#card { background: #ffffff; border: 1px solid #dde3ec; border-radius: 8px; }
QLabel#title { font-size: 19px; font-weight: 600; }
QLabel#sub { color: #6a7588; }
QLabel#h { font-size: 10px; font-weight: 600; color: #6a7588; letter-spacing: 1px; }
QLabel#kpi { font-size: 22px; font-weight: 600; color: #2457d6; }
QLabel#kpired { font-size: 22px; font-weight: 600; color: #c62f3d; }
QLabel#pillwarn { background: #fff6e0; border: 1px solid #f0d08a; border-radius: 10px; padding: 3px 12px; color: #7a5200; }
QLabel#pillok { background: #eaf7ef; border: 1px solid #a6d9b9; border-radius: 10px; padding: 3px 12px; color: #14663a; }
QLabel#pillbad { background: #fdf1f2; border: 1px solid #e7b6bb; border-radius: 10px; padding: 3px 12px; color: #8e1a26; }
QPushButton { background: #ffffff; border: 1px solid #c9d2e0; border-radius: 6px; padding: 5px 12px; }
QPushButton:hover { background: #f0f4fb; }
QPushButton:checked { background: #eef3ff; border-color: #2457d6; color: #2457d6; }
QLineEdit { background: #ffffff; border: 1px solid #c9d2e0; border-radius: 6px; padding: 4px 8px; }
QTableWidget, QTableView, QListWidget { background: #ffffff; border: 1px solid #dde3ec; border-radius: 6px; gridline-color: #eef1f6; selection-background-color: #dfe8fb; selection-color: #1b2333; }
QHeaderView::section { background: #f5f7fa; border: none; border-bottom: 1px solid #dde3ec; padding: 4px; font-weight: 600; color: #4a5568; }
QProgressBar { border: none; background: #eef1f6; border-radius: 3px; max-height: 8px; text-align: center; }
QProgressBar::chunk { background: #2457d6; border-radius: 3px; }
"""


# ------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------
def card(layout=None):
    f = QFrame()
    f.setObjectName("card")
    if layout is not None:
        layout.setContentsMargins(12, 10, 12, 10)
        f.setLayout(layout)
    return f


def label(text, name=None, wrap=False):
    l = QLabel(text)
    if name:
        l.setObjectName(name)
    l.setWordWrap(wrap)
    return l


def titled_card(title, widget=None):
    lay = QVBoxLayout()
    lay.setSpacing(6)
    lay.addWidget(label(title.upper(), "h"))
    if widget is not None:
        lay.addWidget(widget, 1)
    return card(lay), lay


def kpi_card(caption, name="kpi"):
    lay = QVBoxLayout()
    lay.setSpacing(0)
    lay.addWidget(label(caption.upper(), "h"))
    val = label("0", name)
    lay.addWidget(val)
    return card(lay), val


def make_table(headers, rows, right=()):
    t = QTableWidget(len(rows), len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QTableWidget.NoEditTriggers)
    t.setSelectionMode(QTableWidget.NoSelection)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    t.verticalHeader().setDefaultSectionSize(22)
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            it = QTableWidgetItem(str(v))
            if c in right:
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            t.setItem(r, c, it)
    t.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    t.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    t.setFixedHeight(34 + 22 * len(rows))
    return t


def style_chart(chart):
    chart.setBackgroundRoundness(0)
    chart.setMargins(QMargins(2, 2, 2, 2))
    chart.setBackgroundBrush(QBrush(QColor("#ffffff")))
    chart.setBackgroundPen(QPen(Qt.NoPen))


def chart_view(chart, min_h):
    v = QChartView(chart)
    v.setRenderHint(QPainter.Antialiasing)
    v.setMinimumHeight(min_h)
    v.setStyleSheet("background: transparent; border: none;")
    return v


def bar_chart(categories, series, y_max, colors, fmt="%.0f", min_h=150, labels=True):
    chart = QChart()
    style_chart(chart)
    chart.legend().setVisible(len(series) > 1)
    chart.legend().setAlignment(Qt.AlignBottom)
    bars = QBarSeries()
    bars.setLabelsVisible(labels)
    sets = []
    for i, (name, vals) in enumerate(series.items()):
        s = QBarSet(name)
        s.append(vals)
        s.setColor(QColor(colors[i % len(colors)]))
        bars.append(s)
        sets.append(s)
    chart.addSeries(bars)
    ax = QBarCategoryAxis()
    ax.append(categories)
    chart.addAxis(ax, Qt.AlignBottom)
    bars.attachAxis(ax)
    ay = QValueAxis()
    ay.setRange(0, y_max)
    ay.setTickCount(3)
    ay.setLabelFormat(fmt)
    chart.addAxis(ay, Qt.AlignLeft)
    bars.attachAxis(ay)
    return chart_view(chart, min_h), ay, sets


def qr_pixmap(url, size):
    buf = BytesIO()
    segno.make(url, error="m").save(buf, kind="png", scale=8, border=1,
                                    dark="#1b2333", light="#ffffff")
    img = QImage.fromData(buf.getvalue())
    return QPixmap.fromImage(img).scaled(size, size, Qt.KeepAspectRatio, Qt.FastTransformation)


# ------------------------------------------------------------------
# log table model (virtual: handles tens of thousands of rows)
# ------------------------------------------------------------------
LOG_COLUMNS = [("#", "id", 50), ("Time", "time", 92), ("Origin", "src", 46), ("Source", None, 130), ("Method", "method", 52),
               ("Path", "path", 80), ("Code", "status", 42), ("Rx B", "rx", 46), ("Tx B", "tx", 46),
               ("ms", "ms", 44), ("Level", "level", 54), ("Detail", "note", 0)]
LEVEL_FG = {"INFO": "#1b2333", "DEBUG": "#8a94a6", "ALERT": DANGER, "BLOCK": "#8e1a26", "WARN": WARN}
LEVEL_BG = {"ALERT": "#fdf1f2", "BLOCK": "#f9dfe2"}
MAX_VIEW_ROWS = 20000


class LogModel(QAbstractTableModel):
    def __init__(self):
        super().__init__()
        self.rows = []

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(LOG_COLUMNS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return LOG_COLUMNS[section][0]

    def data(self, index, role=Qt.DisplayRole):
        e = self.rows[index.row()]
        key = LOG_COLUMNS[index.column()][1]
        if role == Qt.DisplayRole:
            return f"{e['ip']}:{e['port']}" if key is None else str(e[key])
        if role == Qt.ForegroundRole:
            if e["src"] == "SIM" and e["level"] == "INFO":
                return QBrush(QColor("#5b6578"))
            return QBrush(QColor(LEVEL_FG.get(e["level"], "#1b2333")))
        if role == Qt.BackgroundRole and e["level"] in LEVEL_BG:
            return QBrush(QColor(LEVEL_BG[e["level"]]))
        if role == Qt.TextAlignmentRole and key in ("rx", "tx", "ms", "status", "id"):
            return int(Qt.AlignRight | Qt.AlignVCenter)
        if role == Qt.ToolTipRole:
            return f"{e['note']}\nUser-Agent: {e['ua']}"

    def append(self, entries):
        if not entries:
            return
        first = len(self.rows)
        self.beginInsertRows(QModelIndex(), first, first + len(entries) - 1)
        self.rows.extend(entries)
        self.endInsertRows()
        extra = len(self.rows) - MAX_VIEW_ROWS
        if extra > 0:
            self.beginRemoveRows(QModelIndex(), 0, extra - 1)
            del self.rows[:extra]
            self.endRemoveRows()

    def reset(self, entries):
        self.beginResetModel()
        self.rows = entries[-MAX_VIEW_ROWS:]
        self.endResetModel()


# ------------------------------------------------------------------
# the dashboard
# ------------------------------------------------------------------
class Dashboard(QWidget):
    TIMELINE_SECONDS = 60
    LEVELS = ["INFO", "ALERT", "BLOCK", "WARN", "DEBUG"]

    def __init__(self, R):
        super().__init__()
        self.setObjectName("page")
        self.R = R
        self.last_id = 0
        self.rx = self.tx = 0
        self.rate = 0.0
        self.acc_normal = self.acc_attack = 0
        self.buckets = collections.deque([(0, 0)] * self.TIMELINE_SECONDS, maxlen=self.TIMELINE_SECONDS)
        self.latest_probs = None
        self._qr_url = None

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)
        root.addWidget(self._header())
        root.addLayout(self._kpis())
        root.addLayout(self._main(), 1)
        root.addLayout(self._research())

    # ---------------- header ----------------
    def _header(self):
        lay = QHBoxLayout()
        lay.setSpacing(14)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(label("NIDS Command Center", "title"))
        titles.addWidget(label("CNN-GRU intrusion detection  |  CICIDS2017  |  7 classes  |  live phone-to-PC demo", "sub"))
        lay.addLayout(titles)
        lay.addStretch()
        self.pill = label("Loading model…", "pillwarn")
        lay.addWidget(self.pill)
        self.qr = QLabel()
        self.qr.setFixedSize(84, 84)
        lay.addWidget(self.qr)
        conn = QVBoxLayout()
        conn.setSpacing(0)
        conn.addWidget(label("CONNECT PHONE (same Wi-Fi)", "h"))
        self.url = label("")
        self.url.setFont(QFont("monospace", 11))
        self.url.setTextInteractionFlags(Qt.TextSelectableByMouse)
        conn.addWidget(self.url)
        conn.addWidget(label("Scan the QR code or open the address", "sub"))
        lay.addLayout(conn)
        return card(lay)

    # ---------------- KPI row ----------------
    def _kpis(self):
        row = QHBoxLayout()
        row.setSpacing(10)
        specs = [("Flows analysed", "kpi"), ("Attacks detected", "kpired"), ("Sources blocked", "kpired"),
                 ("Requests dropped", "kpi"), ("Requests / s", "kpi"), ("Traffic (KB)", "kpi")]
        self.kpi = []
        for cap, nm in specs:
            c, v = kpi_card(cap, nm)
            row.addWidget(c)
            self.kpi.append(v)
        return row

    # ---------------- main area ----------------
    def _main(self):
        row = QHBoxLayout()
        row.setSpacing(10)
        left = QVBoxLayout()
        left.setSpacing(10)
        top = QHBoxLayout()
        top.setSpacing(10)
        top.addWidget(self._timeline(), 3)
        top.addWidget(self._sources(), 2)
        left.addLayout(top, 2)
        left.addWidget(self._logs(), 5)
        right = QVBoxLayout()
        right.setSpacing(10)
        right.addWidget(self._defence())
        right.addWidget(self._probabilities())
        right.addWidget(self._detections(), 1)
        row.addLayout(left, 5)
        row.addLayout(right, 2)
        return row

    def _timeline(self):
        self.tl_normal, self.tl_attack = QLineSeries(), QLineSeries()
        self.tl_normal.setName("Normal")
        self.tl_attack.setName("Attack / blocked")
        self.tl_normal.setPen(QPen(QColor(OK), 2))
        self.tl_attack.setPen(QPen(QColor(DANGER), 2))
        chart = QChart()
        style_chart(chart)
        chart.addSeries(self.tl_normal)
        chart.addSeries(self.tl_attack)
        chart.legend().setAlignment(Qt.AlignRight)
        self.tl_x = QValueAxis()
        self.tl_x.setRange(-self.TIMELINE_SECONDS, 0)
        self.tl_x.setTickCount(7)
        self.tl_x.setLabelFormat("%d s")
        self.tl_y = QValueAxis()
        self.tl_y.setRange(0, 5)
        self.tl_y.setLabelFormat("%d")
        self.tl_y.setTickCount(3)
        chart.addAxis(self.tl_x, Qt.AlignBottom)
        chart.addAxis(self.tl_y, Qt.AlignLeft)
        for s in (self.tl_normal, self.tl_attack):
            s.attachAxis(self.tl_x)
            s.attachAxis(self.tl_y)
        c, _ = titled_card("Traffic per second (last 60 s)", chart_view(chart, 120))
        return c

    def _sources(self):
        self.src_table = QTableWidget(0, 5)
        self.src_table.setHorizontalHeaderLabels(["Source IP", "Origin", "Flows", "Attacks", "Status"])
        self.src_table.verticalHeader().setVisible(False)
        self.src_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.src_table.setSelectionMode(QTableWidget.NoSelection)
        self.src_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.src_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.src_table.verticalHeader().setDefaultSectionSize(20)
        self.src_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        c, _ = titled_card("Top sources (by attacks)", self.src_table)
        return c

    def _logs(self):
        lay = QVBoxLayout()
        lay.setSpacing(6)
        head = QHBoxLayout()
        head.addWidget(label("LIVE TRAFFIC LOG  -  every request reaching this PC", "h"))
        head.addStretch()
        lay.addLayout(head)

        bar = QHBoxLayout()
        self.level_chk = {}
        for lv in self.LEVELS:
            c = QCheckBox(lv)
            c.setChecked(lv != "DEBUG")
            c.toggled.connect(self._rebuild)
            self.level_chk[lv] = c
            bar.addWidget(c)
        self.src_chk = {}
        for key, text in (("LIVE", "Live phone"), ("SIM", "Simulated")):
            c = QCheckBox(text)
            c.setChecked(True)
            c.toggled.connect(self._rebuild)
            self.src_chk[key] = c
            bar.addWidget(c)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter by IP, path or detail…")
        self.search.textChanged.connect(self._rebuild)
        bar.addWidget(self.search, 1)
        self.pause = QPushButton("Pause")
        self.pause.setCheckable(True)
        self.follow = QCheckBox("Auto-scroll")
        self.follow.setChecked(True)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.clear_view)
        export = QPushButton("Export CSV")
        export.clicked.connect(self._export)
        for w in (self.pause, self.follow, clear, export):
            bar.addWidget(w)
        lay.addLayout(bar)

        self.model = LogModel()
        self.view = QTableView()
        self.view.setModel(self.model)
        self.view.verticalHeader().setVisible(False)
        self.view.verticalHeader().setDefaultSectionSize(20)
        self.view.setSelectionBehavior(QTableView.SelectRows)
        self.view.setSelectionMode(QTableView.SingleSelection)
        self.view.setEditTriggers(QTableView.NoEditTriggers)
        self.view.setShowGrid(False)
        mono = QFont("monospace", 9)
        mono.setStyleHint(QFont.Monospace)
        self.view.setFont(mono)
        hh = self.view.horizontalHeader()
        for i, (_, _, w) in enumerate(LOG_COLUMNS):
            if w:
                hh.setSectionResizeMode(i, QHeaderView.Interactive)
                self.view.setColumnWidth(i, w)
            else:
                hh.setSectionResizeMode(i, QHeaderView.Stretch)
        self.view.selectionModel().currentRowChanged.connect(self._select_row)
        lay.addWidget(self.view, 1)
        self.log_status = label("", "sub")
        lay.addWidget(self.log_status)
        return card(lay)

    def _defence(self):
        lay = QVBoxLayout()
        lay.setSpacing(6)
        lay.addWidget(label("DEFENCE", "h"))
        self.chk = QCheckBox("Auto-block attackers")
        self.chk.setChecked(server.config["defence"])
        self.chk.toggled.connect(lambda v: self._cfg(defence=v))
        lay.addWidget(self.chk)
        row = QHBoxLayout()
        row.addWidget(label("Block after", "sub"))
        self.spin = QSpinBox()
        self.spin.setRange(1, 20)
        self.spin.setValue(server.config["strikes_to_block"])
        self.spin.valueChanged.connect(lambda v: self._cfg(strikes_to_block=v))
        row.addWidget(self.spin)
        row.addWidget(label("alerts", "sub"))
        row.addStretch()
        lay.addLayout(row)
        self.blist = QListWidget()
        self.blist.setFixedHeight(40)
        lay.addWidget(self.blist)
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #dde3ec;")
        lay.addWidget(sep)
        lay.addWidget(label("SIMULATED NETWORK (not real traffic)", "h"))
        self.sim_chk = QCheckBox("Random-IP background traffic")
        self.sim_chk.setChecked(server.config["sim"])
        self.sim_chk.toggled.connect(lambda v: self._cfg(sim=v))
        lay.addWidget(self.sim_chk)
        srow = QHBoxLayout()
        self.sim_rate = QSpinBox()
        self.sim_rate.setRange(1, 200)
        self.sim_rate.setValue(server.config["sim_rate"])
        self.sim_rate.valueChanged.connect(lambda v: self._cfg(sim_rate=v))
        self.sim_int = QSpinBox()
        self.sim_int.setRange(5, 300)
        self.sim_int.setPrefix("attack every ~")
        self.sim_int.setSuffix(" s")
        self.sim_int.setValue(server.config["sim_interval"])
        self.sim_int.valueChanged.connect(lambda v: self._cfg(sim_interval=v))
        srow.addWidget(self.sim_rate)
        srow.addWidget(label("flows/s", "sub"))
        srow.addWidget(self.sim_int)
        lay.addLayout(srow)
        lay.addWidget(label("Normal hosts + random attack bursts, reserved test IPs. Tagged SIM.", "sub"))
        btns = QHBoxLayout()
        ub = QPushButton("Unblock all")
        ub.clicked.connect(self._unblock)
        rb = QPushButton("Reset demo")
        rb.clicked.connect(self._reset)
        btns.addWidget(ub)
        btns.addWidget(rb)
        lay.addLayout(btns)
        return card(lay)

    def _probabilities(self):
        grid = QGridLayout()
        grid.setSpacing(2)
        self.prob_title = label("MODEL OUTPUT  -  latest classified flow", "h")
        grid.addWidget(self.prob_title, 0, 0, 1, 3)
        self.bars, self.pcts = [], []
        for i, name in enumerate(CLASS_NAMES):
            grid.addWidget(label(name), i + 1, 0)
            b = QProgressBar()
            b.setRange(0, 1000)
            b.setTextVisible(False)
            grid.addWidget(b, i + 1, 1)
            p = label("0.00%")
            p.setMinimumWidth(52)
            p.setAlignment(Qt.AlignRight)
            grid.addWidget(p, i + 1, 2)
            self.bars.append(b)
            self.pcts.append(p)
        grid.setColumnStretch(1, 1)
        return card(grid)

    def _detections(self):
        view, self.det_axis, sets = bar_chart(SHORT_NAMES, {"Detections": [0] * 7}, 5,
                                              [ACCENT], min_h=100)
        self.det_set = sets[0]
        c, _ = titled_card("Model verdicts by class (since reset)", view)
        return c

    # ---------------- research strip ----------------
    def _research(self):
        R = self.R
        row = QHBoxLayout()
        row.setSpacing(10)

        # 1. conventional test
        rows = [(r["Feature_Set"], f"{float(r['Accuracy']):.4f}", f"{float(r['Balanced_Accuracy']):.4f}",
                 f"{float(r['Macro_F1']):.4f}") for r in R["features"]]
        t = make_table(["Set", "Accuracy", "Bal. acc.", "Macro F1"], rows, (1, 2, 3))
        c, lay = titled_card("Standard 80/20 test (561,143 rows)", t)
        lay.addWidget(label("Saturates at 1.0000 - but see the leakage audit ->", "sub", wrap=True))
        row.addWidget(c, 3)

        # 2. LOAO
        L = R["loao"]
        mean_recall = sum(float(r["recall"]) for r in L) / len(L)
        view, ay, _ = bar_chart([SHORT_NAMES[int(r["unseen_class"])] for r in L],
                                {"Recall %": [round(float(r["recall"]) * 100, 1) for r in L]},
                                100, [DANGER], min_h=120)
        c, lay = titled_card("Zero-day recall, unseen attacks (LOAO)", view)
        lay.addWidget(label(f"Mean recall {mean_recall * 100:.2f}%  |  mean F1 "
                            f"{sum(float(r['f1']) for r in L) / len(L) * 100:.2f}%", "sub"))
        row.addWidget(c, 4)

        # 3. leakage
        lk = R["leakage"]
        lay = QGridLayout()
        lay.setSpacing(2)
        lay.addWidget(label("DATA LEAKAGE AUDIT (Top-30 features)", "h"), 0, 0, 1, 2)
        for i, (cap, val, nm) in enumerate([
                ("Train rows sharing a vector with test", f"{lk['train_overlap_pct']:.2f}%", "kpired"),
                ("Test rows sharing a vector with train", f"{lk['test_overlap_pct']:.2f}%", "kpired"),
                ("Identical vectors, conflicting labels", str(lk["mismatch"]), "kpi")]):
            lay.addWidget(label(cap, "sub"), 1 + i, 0)
            v = label(val, nm)
            v.setAlignment(Qt.AlignRight)
            lay.addWidget(v, 1 + i, 1)
        row.addWidget(card(lay), 3)

        # 4. model
        self.model_text = label("", wrap=True)
        c, lay = titled_card("Model and pipeline")
        lay.addWidget(label(
            "Clean (2.8M rows, 61 features) -> 80/20 split -> ANOVA Top-N -> scale (train-fit) -> "
            "windows of 10 flows\n"
            "Conv1D 64 -> BN -> MaxPool -> Conv1D 128 -> BN -> GRU 64 -> Dropout 0.3 -> Dense 64 -> "
            "Dropout 0.3 -> Softmax 7\n"
            "Adam 1e-3, inverse-frequency class weights, early stopping", wrap=True))
        lay.addWidget(self.model_text)
        row.addWidget(c, 5)
        return row

    # ---------------- actions ----------------
    def _cfg(self, **kw):
        with server.lock:
            server.config.update(kw)

    def _unblock(self):
        with server.lock:
            server.blocked.clear()
            server.strikes.clear()

    def _reset(self):
        with server.lock:
            server.events.clear()
            server.blocked.clear()
            server.strikes.clear()
            server.access_log.clear()
            server.sources.clear()
            server.counters.update(total=0, attacks=0, blocked_requests=0, by_class=[0] * 7)
        self.clear_view()
        self.buckets.extend([(0, 0)] * self.TIMELINE_SECONDS)
        self.latest_probs = None
        self._show_probs(None)

    def _passes(self, e):
        chk = self.level_chk.get(e["level"])
        if chk is not None and not chk.isChecked():
            return False
        if not self.src_chk[e.get("src", "LIVE")].isChecked():
            return False
        q = self.search.text().strip().lower()
        return not q or q in f"{e['ip']} {e['path']} {e['note']} {e['level']}".lower()

    def _snapshot(self):
        try:
            return list(server.access_log)
        except RuntimeError:
            return None

    def _rebuild(self, *_):
        snap = self._snapshot()
        if snap is not None:
            self.model.reset([e for e in snap if self._passes(e)])
            self.view.scrollToBottom()

    def clear_view(self):
        self.model.reset([])
        snap = self._snapshot()
        if snap:
            self.last_id = snap[-1]["id"]
        self.rx = self.tx = 0

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export traffic log", "traffic_log.csv", "CSV (*.csv)")
        snap = self._snapshot()
        if not path or snap is None:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=server.LOG_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows([e for e in snap if self._passes(e)])

    def _select_row(self, current, _previous):
        row = current.row()
        if 0 <= row < len(self.model.rows) and self.model.rows[row].get("probs"):
            e = self.model.rows[row]
            self.prob_title.setText(f"MODEL OUTPUT  -  log line #{e['id']}")
            self._show_probs(e["probs"])
        else:
            self.prob_title.setText("MODEL OUTPUT  -  latest classified flow")
            self._show_probs(self.latest_probs)

    def _show_probs(self, probs):
        top = max(range(7), key=lambda i: probs[i]) if probs else -1
        for i in range(7):
            v = probs[i] if probs else 0.0
            self.bars[i].setValue(int(v * 1000))
            self.pcts[i].setText(f"{v * 100:.2f}%")
            self.bars[i].setStyleSheet(
                f"QProgressBar::chunk {{ background: {DANGER if i == top and top != 0 else ACCENT}; }}")

    # ---------------- periodic updates ----------------
    def refresh(self):
        self._refresh_header()
        self._refresh_logs()
        self._refresh_side()

    def _refresh_header(self):
        url = server.phone_url()
        if url != self._qr_url:
            self._qr_url = url
            self.url.setText(url)
            self.qr.setPixmap(qr_pixmap(url, 84))
        if server.model is None and server.model_error:
            text, name = f"Model error: {server.model_error[:70]}", "pillbad"
        elif server.model is None:
            text, name = "Loading model (TensorFlow)…", "pillwarn"
        elif server.is_synthetic:
            text, name = f"SYNTHETIC data - {len(server.X)} test flows (pipeline demo, not paper results)", "pillwarn"
        else:
            text, name = f"Model ready - Top-40 CNN-GRU - {len(server.X):,} test sequences", "pillok"
        if self.pill.text() != text:
            self.pill.setText(text)
            self.pill.setObjectName(name)
            self.pill.style().unpolish(self.pill)
            self.pill.style().polish(self.pill)
        if server.model is not None:
            self.model_text.setText(f"Loaded: cnn_gru_top40_final.keras  |  {server.model.count_params():,} parameters  |  "
                                    "phone requests are real; classified flows are sampled from the test set.")

    def _refresh_logs(self):
        if self.pause.isChecked():
            return
        try:
            new = []
            for e in reversed(server.access_log):
                if e["id"] <= self.last_id:
                    break
                new.append(e)
        except RuntimeError:
            return
        new.reverse()
        real = [e for e in new if e["level"] != "DEBUG"]
        self.rate = 0.6 * self.rate + 0.4 * (len(real) / 0.5)
        if new:
            self.last_id = new[-1]["id"]
            self.rx += sum(e["rx"] for e in new)
            self.tx += sum(e["tx"] for e in new)
            for e in real:
                if e["level"] in ("ALERT", "BLOCK") or e["status"] == 403:
                    self.acc_attack += 1
                else:
                    self.acc_normal += 1
                if e.get("probs"):
                    self.latest_probs = e["probs"]
            self.model.append([e for e in new if self._passes(e)])
            if self.follow.isChecked():
                self.view.scrollToBottom()
            if not self.view.currentIndex().isValid():
                self._show_probs(self.latest_probs)
        self.log_status.setText(f"showing {len(self.model.rows):,} log lines  |  also written to webapp/logs/traffic.log")

    def _refresh_side(self):
        with server.lock:
            c = dict(server.counters)
            by_class = list(c["by_class"])
            blocked = dict(server.blocked)
        self.kpi[0].setText(f"{c['total']:,}")
        self.kpi[1].setText(f"{c['attacks']:,}")
        self.kpi[2].setText(str(len(blocked)))
        self.kpi[3].setText(f"{c['blocked_requests']:,}")
        self.kpi[4].setText(f"{self.rate:.1f}")
        self.kpi[5].setText(f"{(self.rx + self.tx) / 1024:.1f}")

        with server.lock:
            top = sorted(server.sources.items(), key=lambda kv: (-kv[1]["attacks"], -kv[1]["flows"]))[:5]
        self.src_table.setRowCount(len(top))
        for r, (ip, info) in enumerate(top):
            if ip in blocked:
                status, color = "BLOCKED", "#8e1a26"
            elif info["attacks"]:
                status, color = "suspicious", WARN
            else:
                status, color = "ok", OK
            vals = [ip, info["origin"], str(info["flows"]), str(info["attacks"]), status]
            for col, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if col in (2, 3):
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if col == 4:
                    it.setForeground(QBrush(QColor(color)))
                self.src_table.setItem(r, col, it)

        want = [f"{ip}  -  {info['reason']}" for ip, info in blocked.items()]
        if [self.blist.item(i).text() for i in range(self.blist.count())] != want:
            self.blist.clear()
            self.blist.addItems(want)

        for i, v in enumerate(by_class):
            if self.det_set.at(i) != v:
                self.det_set.replace(i, v)
        self.det_axis.setRange(0, max(5, int(max(by_class) * 1.2) + 1))

    def tick_timeline(self):
        self.buckets.append((self.acc_normal, self.acc_attack))
        self.acc_normal = self.acc_attack = 0
        n = len(self.buckets)
        xs = range(-n + 1, 1)
        self.tl_normal.replace([QPointF(x, b[0]) for x, b in zip(xs, self.buckets)])
        self.tl_attack.replace([QPointF(x, b[1]) for x, b in zip(xs, self.buckets)])
        self.tl_y.setRange(0, max(5, int(max(max(b) for b in self.buckets) * 1.2) + 1))


# ------------------------------------------------------------------
class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NIDS Command Center - CNN-GRU Intrusion Detection")
        self.resize(1600, 1060)
        self.dash = Dashboard(server.load_results())
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.dash.setMinimumSize(1250, 1000)
        scroll.setWidget(self.dash)
        self.setCentralWidget(scroll)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.dash.refresh)
        self.timer.start(500)
        self.tl_timer = QTimer(self)
        self.tl_timer.timeout.connect(self.dash.tick_timeline)
        self.tl_timer.start(1000)
        self.dash.refresh()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    try:
        server.start_server()
    except OSError as exc:
        sys.exit(f"Cannot start the phone server on port {server.PORT}: {exc}\n"
                 "Is another copy of the app already running? (or set NIDS_PORT)")
    threading.Thread(target=server.load_engine, daemon=True).start()
    win = Main()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
