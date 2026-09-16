"""
MathLens - PyQt6 Desktop: захоплення екрана -> OCR -> SymPy (+ Gemini за запитом).
Однофайлова збірка. Запуск: python mathlens.py
"""
from __future__ import annotations

import os
import re
import sys
import shutil
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

# ------------------------------------------------------------------ dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ------------------------------------------------------------------ config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
DEFAULT_HOTKEY = os.getenv("HOTKEY", "F8").strip()
DEFAULT_INTERVAL_S = int(os.getenv("AUTO_INTERVAL_S", "3"))
OCR_LANG = os.getenv("OCR_LANG", "eng+ukr").strip()
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()
OCR_MIN_CONF = 30
SOLVER_CACHE_SIZE = 256
DEFAULT_FRAME = (220, 220, 720, 320)


# ------------------------------------------------------------------ imports
import mss
import pytesseract
import sympy as sp
from PIL import Image
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor,
)

from PyQt6.QtCore import (
    Qt, QRect, QPoint, pyqtSignal, QThread, QObject, QTimer,
    QRunnable, QThreadPool,
)
from PyQt6.QtGui import QPainter, QColor, QPen, QFont
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QCheckBox, QSlider, QLabel,
    QInputDialog, QDialog, QVBoxLayout, QHBoxLayout, QTextBrowser,
    QMessageBox, QFileDialog, QLineEdit, QFrame,
)

try:
    import keyboard
    _HAS_KEYBOARD = True
except Exception:
    _HAS_KEYBOARD = False


# ------------------------------------------------------------------ Tesseract
def find_tesseract() -> str:
    """Шукає tesseract у PATH або типових місцях встановлення."""
    if TESSERACT_CMD and os.path.isfile(TESSERACT_CMD):
        return TESSERACT_CMD
    found = shutil.which("tesseract")
    if found:
        return found
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/opt/homebrew/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/usr/bin/tesseract",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return ""


_TESS_PATH = find_tesseract()
if _TESS_PATH:
    pytesseract.pytesseract.tesseract_cmd = _TESS_PATH


def set_tesseract_path(path: str):
    global _TESS_PATH
    _TESS_PATH = path
    pytesseract.pytesseract.tesseract_cmd = path


# ==================================================================== OCR
@dataclass
class OCRItem:
    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float
    line_id: Tuple[int, int, int]


@dataclass
class OCRLine:
    text: str
    left: int
    top: int
    width: int
    height: int


def capture_region(x: int, y: int, w: int, h: int) -> Image.Image:
    if w <= 0 or h <= 0:
        raise ValueError("Некоректні розміри області")
    with mss.mss() as sct:
        monitor = {"left": int(x), "top": int(y),
                   "width": int(w), "height": int(h)}
        shot = sct.grab(monitor)
        return Image.frombytes("RGB", shot.size, shot.rgb)


def ocr_image(img: Image.Image, lang: str, min_conf: int) -> List[OCRItem]:
    data = pytesseract.image_to_data(
        img, lang=lang, output_type=pytesseract.Output.DICT
    )
    items: List[OCRItem] = []
    for i in range(len(data["text"])):
        raw = (data["text"][i] or "").strip()
        if not raw:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < min_conf:
            continue
        items.append(OCRItem(
            text=raw,
            left=int(data["left"][i]), top=int(data["top"][i]),
            width=int(data["width"][i]), height=int(data["height"][i]),
            confidence=conf,
            line_id=(int(data["block_num"][i]),
                     int(data["par_num"][i]),
                     int(data["line_num"][i])),
        ))
    return items


def group_lines(items: List[OCRItem]) -> List[OCRLine]:
    buckets: Dict[Tuple[int, int, int], List[OCRItem]] = {}
    for it in items:
        buckets.setdefault(it.line_id, []).append(it)

    lines: List[OCRLine] = []
    for words in buckets.values():
        words.sort(key=lambda w: w.left)
        text = " ".join(w.text for w in words)
        left = min(w.left for w in words)
        top = min(w.top for w in words)
        right = max(w.left + w.width for w in words)
        bottom = max(w.top + w.height for w in words)
        lines.append(OCRLine(text, left, top, right - left, bottom - top))
    lines.sort(key=lambda l: (l.top, l.left))
    return lines


def read_screen(x: int, y: int, w: int, h: int,
                lang: str, min_conf: int) -> List[OCRLine]:
    img = capture_region(x, y, w, h)
    return group_lines(ocr_image(img, lang=lang, min_conf=min_conf))


# ============================================================== SOLVER
_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application, convert_xor,
)
_SAFE_RE = re.compile(r"^[\s0-9a-zA-Z+\-*/^().,=]+$")
_EQ_RE = re.compile(
    r"[0-9a-zA-Z()\[\]{}\^+\-*/\s]{1,60}=\s*[0-9a-zA-Z()\[\]{}\^+\-*/\s]{1,60}"
)
_ARITH_RE = re.compile(r"\d[\d\s]*[-+*/^]\s*[\d\s()+\-*/^.]*\d")

_CACHE: "OrderedDict[str, Optional[dict]]" = OrderedDict()

_UNICODE_MAP = {"×": "*", "·": "*", "÷": "/", "−": "-", "–": "-", "—": "-",
                "²": "^2", "³": "^3"}


def _normalize(text: str) -> str:
    s = text.strip()
    for k, v in _UNICODE_MAP.items():
        s = s.replace(k, v)
    s = re.sub(r"\s+", "", s)
    s = s.rstrip("?.")
    s = s.rstrip("=")
    return s


def _cache_get(key: str):
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    return "__MISS__"


def _cache_put(key: str, value):
    _CACHE[key] = value
    _CACHE.move_to_end(key)
    while len(_CACHE) > SOLVER_CACHE_SIZE:
        _CACHE.popitem(last=False)


def _fmt(v):
    if getattr(v, "is_Integer", False):
        return str(v)
    if getattr(v, "is_Rational", False):
        return sp.sstr(v)
    if getattr(v, "is_Float", False):
        return f"{float(v):.6g}"
    return sp.sstr(v)


def try_solve(text: str) -> Optional[dict]:
    norm = _normalize(text)
    if not norm or not _SAFE_RE.match(norm):
        return None
    cached = _cache_get(norm)
    if cached != "__MISS__":
        return cached
    result: Optional[dict] = None
    try:
        if "=" in norm:
            result = _solve_equation(norm, text)
        else:
            result = _solve_arithmetic(norm, text)
    except Exception:
        result = None
    _cache_put(norm, result)
    return result


def _solve_arithmetic(norm: str, original: str) -> Optional[dict]:
    expr = parse_expr(norm, transformations=_TRANSFORMS, evaluate=True)
    if expr.free_symbols:
        return None
    val = sp.simplify(expr)
    return {"kind": "arith", "expression": original.strip(),
            "solution": f"= {_fmt(val)}"}


def _solve_equation(norm: str, original: str) -> Optional[dict]:
    lhs, rhs = norm.split("=", 1)
    if not lhs or not rhs:
        return None
    expr = parse_expr(f"({lhs}) - ({rhs})", transformations=_TRANSFORMS)
    syms = sorted(expr.free_symbols, key=lambda s: s.name)
    if not syms:
        if sp.simplify(expr) == 0:
            return {"kind": "identity", "expression": original.strip(),
                    "solution": "тотожність"}
        return None
    x = syms[0]
    sols = sp.solve(expr, x)
    if not sols:
        return None
    sol_str = ", ".join(f"{x} = {_fmt(s)}" for s in sols)
    return {"kind": "equation", "expression": original.strip(),
            "solution": sol_str}


def extract_candidates(line: str) -> List[str]:
    line = line.strip()
    if not line:
        return []
    cands: List[str] = []
    for m in _EQ_RE.finditer(line):
        cands.append(m.group(0).strip())
    for m in _ARITH_RE.finditer(line):
        cands.append(m.group(0).strip())
    if not cands:
        cands.append(line)
    seen, out = set(), []
    for c in sorted(cands, key=len, reverse=True):
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def solve_line(line: str) -> Optional[dict]:
    for cand in extract_candidates(line):
        res = try_solve(cand)
        if res:
            return res
    return None


# ============================================================== AI
_AI_PROMPT = (
    "Ти - стислий математичний репетитор.\n"
    "Поясни розв'язок виразу: \"{expression}\".\n"
    "Формат відповіді:\n"
    "1. Короткий результат.\n"
    "2. 2-3 кроки пояснення без вітань і зайвих слів.\n"
    "Мова: українська."
)
_ai_client = None


def _get_ai_client():
    global _ai_client
    if _ai_client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY не задано у .env")
        from google import genai
        _ai_client = genai.Client(api_key=GEMINI_API_KEY)
    return _ai_client


def explain(expression: str) -> str:
    client = _get_ai_client()
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=_AI_PROMPT.format(expression=expression),
    )
    text = getattr(resp, "text", None)
    if not text:
        raise RuntimeError("Порожня відповідь від Gemini")
    return text.strip()


# ============================================================== СТИЛІ
STYLE_PANEL_BG   = "#000000"
STYLE_TEXT       = "#ffffff"
STYLE_TEXT_DIM   = "#888888"
STYLE_BORDER     = "#333333"
STYLE_BORDER_HI  = "#ffffff"

PANEL_QSS = """
QWidget#ControlPanel {
    background: #000000;
    color: #ffffff;
}
QLabel {
    color: #ffffff;
    font-size: 12px;
}
QLabel#TitleLabel {
    color: #ffffff;
    font-size: 16px;
    font-weight: bold;
    letter-spacing: 2px;
}
QLabel#StatusLabel {
    color: #ffffff;
    font-size: 11px;
    padding: 4px 6px;
    border: 1px solid #333333;
}
QLabel#HintLabel {
    color: #888888;
    font-size: 10px;
}
QPushButton {
    background: #000000;
    color: #ffffff;
    border: 1px solid #ffffff;
    padding: 6px 12px;
    font-size: 12px;
    border-radius: 0px;
}
QPushButton:hover {
    background: #ffffff;
    color: #000000;
}
QPushButton:pressed {
    background: #cccccc;
}
QPushButton:disabled {
    color: #555555;
    border-color: #333333;
}
QPushButton#PrimaryButton {
    background: #ffffff;
    color: #000000;
    border: 1px solid #ffffff;
    padding: 10px 16px;
    font-size: 13px;
    font-weight: bold;
    letter-spacing: 1px;
}
QPushButton#PrimaryButton:hover {
    background: #000000;
    color: #ffffff;
}
QCheckBox {
    color: #ffffff;
    font-size: 12px;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #ffffff;
    background: #000000;
}
QCheckBox::indicator:checked {
    background: #ffffff;
}
QSlider::groove:horizontal {
    height: 2px;
    background: #333333;
}
QSlider::handle:horizontal {
    background: #ffffff;
    width: 12px;
    margin: -6px 0;
    border-radius: 0px;
}
QLineEdit {
    background: #000000;
    color: #ffffff;
    border: 1px solid #333333;
    padding: 4px 6px;
    font-size: 11px;
}
QTextBrowser {
    background: #0a0a0a;
    color: #ffffff;
    border: 1px solid #333333;
    font-size: 12px;
}
QDialog {
    background: #000000;
}
"""


# ============================================================== CAPTURE FRAME
class CaptureFrame(QWidget):
    """Тільки рамка захвату. Тягнеться за будь-яке місце, ресайзиться з країв."""
    roi_changed = pyqtSignal(QRect)

    MARGIN = 14
    MIN_W = 80
    MIN_H = 60

    def __init__(self):
        super().__init__(None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self.setMinimumSize(self.MIN_W, self.MIN_H)
        self.resize(DEFAULT_FRAME[2], DEFAULT_FRAME[3])
        self.move(DEFAULT_FRAME[0], DEFAULT_FRAME[1])

        self._drag_offset: Optional[QPoint] = None
        self._resize_edge: Optional[str] = None
        self._start_geom = QRect()
        self._start_pos = QPoint()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect()
        # Напівпрозора чорна підкладка (щоб рамка читалась на будь-якому фоні)
        p.fillRect(r, QColor(0, 0, 0, 30))
        # Пунктирна біла рамка
        pen = QPen(QColor(255, 255, 255), 1, Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(r.adjusted(1, 1, -2, -2))
        # Кутові квадрати (видимі "ручки")
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255))
        s = 8
        w, h = self.width(), self.height()
        for cx, cy in [(0, 0), (w - s, 0), (0, h - s), (w - s, h - s)]:
            p.drawRect(cx, cy, s, s)

    def roi_rect(self) -> QRect:
        return QRect(self.x(), self.y(), self.width(), self.height())

    # ---------------------------------------------------- edge detection
    def _edge_at(self, pos: QPoint) -> Optional[str]:
        m = self.MARGIN
        w, h = self.width(), self.height()
        x, y = pos.x(), pos.y()
        left, right = x < m, x > w - m
        top, bottom = y < m, y > h - m
        if top and left: return "tl"
        if top and right: return "tr"
        if bottom and left: return "bl"
        if bottom and right: return "br"
        if left: return "l"
        if right: return "r"
        if top: return "t"
        if bottom: return "b"
        return None

    def _update_cursor(self, pos: QPoint):
        e = self._edge_at(pos)
        cursors = {
            "tl": Qt.CursorShape.SizeFDiagCursor,
            "br": Qt.CursorShape.SizeFDiagCursor,
            "tr": Qt.CursorShape.SizeBDiagCursor,
            "bl": Qt.CursorShape.SizeBDiagCursor,
            "l":  Qt.CursorShape.SizeHorCursor,
            "r":  Qt.CursorShape.SizeHorCursor,
            "t":  Qt.CursorShape.SizeVerCursor,
            "b":  Qt.CursorShape.SizeVerCursor,
        }
        self.setCursor(cursors.get(e, Qt.CursorShape.SizeAllCursor))

    def mousePressEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        pos = e.position().toPoint()
        self._resize_edge = self._edge_at(pos)
        if self._resize_edge:
            self._start_geom = QRect(self.geometry())
            self._start_pos = e.globalPosition().toPoint()
        else:
            self._drag_offset = (e.globalPosition().toPoint()
                                 - self.frameGeometry().topLeft())

    def mouseMoveEvent(self, e):
        buttons = e.buttons() & Qt.MouseButton.LeftButton
        if self._resize_edge and buttons:
            self._do_resize(e.globalPosition().toPoint())
        elif self._drag_offset is not None and buttons:
            self.move(e.globalPosition().toPoint() - self._drag_offset)
        else:
            self._update_cursor(e.position().toPoint())

    def mouseReleaseEvent(self, e):
        self._resize_edge = None
        self._drag_offset = None

    def _do_resize(self, gp: QPoint):
        d = gp - self._start_pos
        g = QRect(self._start_geom)
        edge = self._resize_edge
        if "l" in edge: g.setLeft(g.left() + d.x())
        if "r" in edge: g.setRight(g.right() + d.x())
        if "t" in edge: g.setTop(g.top() + d.y())
        if "b" in edge: g.setBottom(g.bottom() + d.y())
        if g.width() < self.MIN_W:
            if "l" in edge: g.setLeft(g.right() - self.MIN_W)
            else: g.setRight(g.left() + self.MIN_W)
        if g.height() < self.MIN_H:
            if "t" in edge: g.setTop(g.bottom() - self.MIN_H)
            else: g.setBottom(g.top() + self.MIN_H)
        self.setGeometry(g)

    def moveEvent(self, e):
        super().moveEvent(e)
        self.roi_changed.emit(self.roi_rect())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.roi_changed.emit(self.roi_rect())


# ============================================================== CONTROL PANEL
class ControlPanel(QWidget):
    """Окреме вікно з налаштуваннями."""
    scan_requested = pyqtSignal()
    auto_changed = pyqtSignal(bool, int)
    hotkey_changed = pyqtSignal(str)
    quit_requested = pyqtSignal()
    tesseract_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__(None)
        self.setObjectName("ControlPanel")
        self.setWindowTitle("MathLens")
        self.setFixedWidth(420)
        self.setStyleSheet(PANEL_QSS)

        self._hotkey = DEFAULT_HOTKEY
        self._build_ui()
        self._refresh_tesseract_status()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # Заголовок
        title = QLabel("MATHLENS")
        title.setObjectName("TitleLabel")
        root.addWidget(title)

        sub = QLabel("Screen OCR + SymPy + optional Gemini")
        sub.setObjectName("HintLabel")
        root.addWidget(sub)

        root.addWidget(self._hsep())

        # Tesseract
        row_t = QHBoxLayout()
        self.tess_status = QLabel("Tesseract: -")
        self.tess_status.setObjectName("StatusLabel")
        row_t.addWidget(self.tess_status, 1)
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self._on_browse_tesseract)
        row_t.addWidget(btn_browse)
        root.addLayout(row_t)

        # Hotkey
        row_h = QHBoxLayout()
        self.hotkey_btn = QPushButton(f"Hotkey: {self._hotkey}")
        self.hotkey_btn.clicked.connect(self._on_hotkey_clicked)
        row_h.addWidget(self.hotkey_btn, 1)
        root.addLayout(row_h)

        root.addWidget(self._hsep())

        # Auto
        row_a = QHBoxLayout()
        self.auto_chk = QCheckBox("Auto scan")
        self.auto_chk.stateChanged.connect(self._emit_auto)
        row_a.addWidget(self.auto_chk)

        self.interval_slider = QSlider(Qt.Orientation.Horizontal)
        self.interval_slider.setRange(1, 10)
        self.interval_slider.setValue(DEFAULT_INTERVAL_S)
        self.interval_slider.valueChanged.connect(self._emit_auto)
        row_a.addWidget(self.interval_slider, 1)

        self.interval_lbl = QLabel(f"{DEFAULT_INTERVAL_S}s")
        self.interval_lbl.setFixedWidth(30)
        row_a.addWidget(self.interval_lbl)
        root.addLayout(row_a)

        root.addWidget(self._hsep())

        # Scan button
        self.scan_btn = QPushButton("ANALYZE")
        self.scan_btn.setObjectName("PrimaryButton")
        self.scan_btn.clicked.connect(self.scan_requested.emit)
        root.addWidget(self.scan_btn)

        # Status
        self.status_lbl = QLabel("Ready")
        self.status_lbl.setObjectName("StatusLabel")
        root.addWidget(self.status_lbl)

        # Quit
        btn_quit = QPushButton("Quit")
        btn_quit.clicked.connect(self.quit_requested.emit)
        root.addWidget(btn_quit)

        hint = QLabel("Hotkey scans current frame. Drag the white frame to position. "
                      "Resize from any edge.")
        hint.setObjectName("HintLabel")
        hint.setWordWrap(True)
        root.addWidget(hint)

    @staticmethod
    def _hsep() -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.Shape.HLine)
        f.setStyleSheet("color:#333333; background:#333333; max-height:1px;")
        return f

    # --------------------------------------------------------- status
    def set_status(self, text: str):
        self.status_lbl.setText(text)

    def _refresh_tesseract_status(self):
        if _TESS_PATH and os.path.isfile(_TESS_PATH):
            self.tess_status.setText(f"Tesseract: OK")
            self.tess_status.setToolTip(_TESS_PATH)
        else:
            self.tess_status.setText("Tesseract: NOT FOUND")
            self.tess_status.setToolTip(
                "Install Tesseract or click Browse to point to tesseract.exe")

    def _on_browse_tesseract(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select tesseract executable", "",
            "Tesseract (tesseract.exe tesseract);;All files (*)")
        if path:
            set_tesseract_path(path)
            self._refresh_tesseract_status()
            self.tesseract_changed.emit(path)

    # --------------------------------------------------------- hotkey
    def _on_hotkey_clicked(self):
        text, ok = QInputDialog.getText(
            self, "Hotkey", "Combination (F8, ctrl+alt+m, ...):",
            text=self._hotkey)
        if ok and text.strip():
            self._hotkey = text.strip()
            self.hotkey_btn.setText(f"Hotkey: {self._hotkey}")
            self.hotkey_changed.emit(self._hotkey)

    # --------------------------------------------------------- auto
    def _emit_auto(self, *_):
        self.interval_lbl.setText(f"{self.interval_slider.value()}s")
        self.auto_changed.emit(self.auto_chk.isChecked(),
                               self.interval_slider.value())


# ============================================================== UI: BADGE
class BadgeWindow(QWidget):
    clicked = pyqtSignal(dict)

    def __init__(self, data: dict, label: str, is_solved: bool):
        super().__init__(None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.data = data
        self._label = label
        self._is_solved = is_solved
        f = QFont(); f.setPointSize(9); f.setBold(True)
        self.setFont(f)
        fm = self.fontMetrics()
        self.resize(fm.horizontalAdvance(label) + 20, fm.height() + 10)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect().adjusted(0, 0, -1, -1)
        if self._is_solved:
            bg, fg, border = QColor(0, 0, 0, 235), QColor(255, 255, 255), QColor(255, 255, 255)
        else:
            bg, fg, border = QColor(255, 255, 255, 240), QColor(0, 0, 0), QColor(255, 255, 255)
        p.setBrush(bg)
        p.setPen(QPen(border, 1))
        p.drawRoundedRect(r, 4, 4)
        p.setPen(fg)
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._label)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.data)


class OverlayManager:
    def __init__(self, on_click):
        self._badges: List[BadgeWindow] = []
        self._roi = QRect()
        self._on_click = on_click

    def clear(self):
        for b in self._badges:
            b.hide(); b.deleteLater()
        self._badges.clear()

    def set_roi(self, roi: QRect):
        self._roi = QRect(roi)
        for b in self._badges:
            b.move(self._badge_pos(b.data))

    def show_items(self, roi: QRect, items: List[dict]):
        self.clear()
        self._roi = QRect(roi)
        for it in items:
            label, solved = self._label_for(it)
            b = BadgeWindow(it, label, solved)
            b.clicked.connect(self._on_click)
            b.move(self._badge_pos(it))
            b.show()
            self._badges.append(b)

    def _badge_pos(self, item: dict) -> QPoint:
        x, y, _, h = item["rect"]
        return QPoint(self._roi.x() + x, self._roi.y() + y + h + 4)

    @staticmethod
    def _label_for(item: dict) -> Tuple[str, bool]:
        sol = item.get("solution")
        if sol:
            label = sol if len(sol) <= 30 else sol[:28] + "..."
            return label, True
        return "ASK AI", False


# ============================================================== AI WORKER
class AIWorker(QThread):
    done = pyqtSignal(str)
    err = pyqtSignal(str)

    def __init__(self, expression: str, parent=None):
        super().__init__(parent)
        self.expression = expression

    def run(self):
        try:
            self.done.emit(explain(self.expression))
        except Exception as e:
            self.err.emit(str(e))


class DetailDialog(QDialog):
    def __init__(self, item: dict, parent=None):
        super().__init__(parent)
        self.item = item
        self._worker: Optional[AIWorker] = None
        self.setWindowTitle("Details")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setStyleSheet(PANEL_QSS)
        self.resize(560, 420)

        v = QVBoxLayout(self)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(10)

        t = QLabel("EXPRESSION")
        t.setObjectName("TitleLabel")
        v.addWidget(t)

        lbl = QLabel(f"<code>{item['text']}</code>")
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lbl.setWordWrap(True)
        v.addWidget(lbl)

        sol = item.get("solution")
        s = QLabel(f"LOCAL RESULT: {sol}" if sol
                   else "LOCAL RESULT: not solved")
        s.setObjectName("StatusLabel")
        v.addWidget(s)

        self.browser = QTextBrowser()
        v.addWidget(self.browser, 1)

        h = QHBoxLayout()
        self.ai_btn = QPushButton("Ask Gemini")
        self.ai_btn.setEnabled(bool(GEMINI_API_KEY))
        self.ai_btn.clicked.connect(self._request_ai)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        h.addWidget(self.ai_btn); h.addStretch(1); h.addWidget(close_btn)
        v.addLayout(h)

        if not GEMINI_API_KEY:
            self.browser.setMarkdown("> GEMINI_API_KEY is not set in .env")

    def _request_ai(self):
        self.ai_btn.setEnabled(False)
        self.browser.setMarkdown("*Requesting Gemini...*")
        self._worker = AIWorker(self.item["text"], parent=self)
        self._worker.done.connect(self._on_done)
        self._worker.err.connect(self._on_err)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _on_done(self, text: str):
        self.browser.setMarkdown(text); self.ai_btn.setEnabled(True)

    def _on_err(self, msg: str):
        self.browser.setMarkdown(f"**AI error:** {msg}")
        self.ai_btn.setEnabled(True)


# ============================================================== SCAN TASK
class ScanSignals(QObject):
    done = pyqtSignal(QRect, list)
    error = pyqtSignal(str)


class ScanTask(QRunnable):
    def __init__(self, roi: QRect):
        super().__init__()
        self.roi = QRect(roi)
        self.signals = ScanSignals()

    def run(self):
        try:
            lines = read_screen(self.roi.x(), self.roi.y(),
                                self.roi.width(), self.roi.height(),
                                lang=OCR_LANG, min_conf=OCR_MIN_CONF)
            items: List[dict] = []
            for line in lines:
                if not line.text.strip():
                    continue
                result = solve_line(line.text)
                items.append({
                    "text": result["expression"] if result else line.text.strip(),
                    "rect": (line.left, line.top, line.width, line.height),
                    "solution": result["solution"] if result else None,
                    "kind": result["kind"] if result else None,
                })
            self.signals.done.emit(self.roi, items)
        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))


class HotkeyBridge(QObject):
    triggered = pyqtSignal()


# ============================================================== CONTROLLER
class App(QObject):
    def __init__(self, qapp: QApplication):
        super().__init__()
        self.qapp = qapp
        self.pool = QThreadPool.globalInstance()

        self.frame = CaptureFrame()
        self.panel = ControlPanel()
        self.overlay = OverlayManager(on_click=self._show_details)

        self.frame.roi_changed.connect(self.overlay.set_roi)
        self.panel.scan_requested.connect(self.start_scan)
        self.panel.auto_changed.connect(self._on_auto_changed)
        self.panel.hotkey_changed.connect(self._register_hotkey)
        self.panel.quit_requested.connect(self.qapp.quit)

        self._auto_timer = QTimer(self)
        self._auto_timer.timeout.connect(self.start_scan)

        self._hotkey_handle = None
        self._bridge = HotkeyBridge()
        self._bridge.triggered.connect(self.start_scan)
        self._register_hotkey(DEFAULT_HOTKEY)

        self.frame.show()
        self.panel.show()

        if not _TESS_PATH:
            self.panel.set_status("Tesseract not found - install or browse")
        else:
            self.panel.set_status("Ready")

    # --------------------------------------------------------- hotkey
    def _register_hotkey(self, hk: str):
        if not _HAS_KEYBOARD:
            self.panel.set_status("keyboard module unavailable")
            return
        try:
            if self._hotkey_handle is not None:
                keyboard.remove_hotkey(self._hotkey_handle)
            self._hotkey_handle = keyboard.add_hotkey(
                hk, self._bridge.triggered.emit)
        except Exception as e:
            QMessageBox.warning(
                self.panel, "Hotkey",
                f"Failed to register '{hk}':\n{e}\n"
                "Try running as administrator (Windows) or with sudo (Linux).")

    # --------------------------------------------------------- auto
    def _on_auto_changed(self, enabled: bool, interval: int):
        self._auto_timer.stop()
        if enabled:
            self._auto_timer.start(max(1, interval) * 1000)
            self.panel.set_status(f"Auto every {interval}s")
        else:
            self.panel.set_status("Ready")

    # --------------------------------------------------------- scan
    def start_scan(self):
        if not _TESS_PATH:
            self.panel.set_status("Tesseract not found")
            QMessageBox.warning(
                self.panel, "Tesseract",
                "Tesseract OCR is not installed or not in PATH.\n\n"
                "Windows: install from https://github.com/UB-Mannheim/tesseract/wiki\n"
                "macOS: brew install tesseract tesseract-lang\n"
                "Linux: sudo apt install tesseract-ocr tesseract-ocr-ukr\n\n"
                "Or click 'Browse' in the panel to point to tesseract executable.")
            return

        roi = self.frame.roi_rect()
        if roi.width() < 30 or roi.height() < 20:
            self.panel.set_status("Frame too small")
            return
        self.panel.set_status("Scanning...")
        task = ScanTask(roi)
        task.signals.done.connect(self._on_scan_done)
        task.signals.error.connect(self._on_scan_error)
        self.pool.start(task)

    def _on_scan_done(self, roi: QRect, items: List[dict]):
        self.overlay.show_items(roi, items)
        solved = sum(1 for it in items if it.get("solution"))
        if not items:
            self.panel.set_status("Nothing found")
        else:
            self.panel.set_status(f"{solved}/{len(items)} solved locally")

    def _on_scan_error(self, msg: str):
        self.panel.set_status("Error")
        QMessageBox.warning(self.panel, "Scan error", msg)

    def _show_details(self, item: dict):
        dlg = DetailDialog(item, parent=None)
        dlg.show(); dlg.exec()


# ============================================================== ENTRY
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("MathLens")
    controller = App(app)   # noqa: F841
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
