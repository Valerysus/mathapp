"""
MathLens — PyQt6 Desktop: захоплення екрана → OCR → SymPy (+ Gemini за запитом).
Однофайлова збірка. Запуск: python mathlens.py
"""
from __future__ import annotations

import os
import re
import sys
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
DEFAULT_FRAME = (200, 200, 720, 340)
OCR_MIN_CONF = 30
SOLVER_CACHE_SIZE = 256

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
    QMessageBox,
)

try:
    import keyboard
    _HAS_KEYBOARD = True
except Exception:
    _HAS_KEYBOARD = False


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
    "Ти — стислий математичний репетитор.\n"
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


# ============================================================== UI: FRAME
class SelectionFrame(QWidget):
    roi_changed = pyqtSignal(QRect)
    scan_requested = pyqtSignal()
    auto_changed = pyqtSignal(bool, int)
    hotkey_changed = pyqtSignal(str)
    quit_requested = pyqtSignal()

    HEADER_H = 44
    MARGIN = 8

    def __init__(self, hotkey: str, interval: int):
        super().__init__(None, Qt.WindowType.FramelessWindowHint
                              | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self.setMinimumSize(520, 220)
        self.resize(DEFAULT_FRAME[2], DEFAULT_FRAME[3])
        self.move(DEFAULT_FRAME[0], DEFAULT_FRAME[1])

        self._drag_offset: Optional[QPoint] = None
        self._resize_edge: Optional[str] = None
        self._start_geom = QRect()
        self._start_pos = QPoint()
        self._hotkey = hotkey

        self._build_controls(interval)
        self._layout_header()

    def _build_controls(self, interval: int):
        self.hotkey_btn = QPushButton(f"🎯 {self._hotkey}", self)
        self.hotkey_btn.clicked.connect(self._on_hotkey_clicked)

        self.auto_chk = QCheckBox("Авто", self)
        self.auto_chk.stateChanged.connect(self._emit_auto)

        self.interval_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.interval_slider.setRange(1, 10)
        self.interval_slider.setValue(int(interval))
        self.interval_slider.setFixedWidth(90)
        self.interval_slider.valueChanged.connect(self._emit_auto)

        self.interval_lbl = QLabel(f"{interval}s", self)
        self.interval_lbl.setFixedWidth(28)

        self.scan_btn = QPushButton("Аналізувати", self)
        self.scan_btn.clicked.connect(self.scan_requested.emit)

        self.status_lbl = QLabel("● Готовий", self)

        self.close_btn = QPushButton("✕", self)
        self.close_btn.setFixedWidth(28)
        self.close_btn.clicked.connect(self.quit_requested.emit)

        self.scan_btn.setStyleSheet(
            "QPushButton{background:#2563eb;color:#fff;border:0;"
            "padding:5px 10px;border-radius:5px;font-size:12px;}"
            "QPushButton:hover{background:#1d4ed8;}"
        )
        self.hotkey_btn.setStyleSheet(
            "QPushButton{background:#0f172a;color:#cbd5e1;"
            "border:1px solid #334155;padding:4px 8px;"
            "border-radius:5px;font-size:12px;}"
        )
        self.close_btn.setStyleSheet(
            "QPushButton{background:#7f1d1d;color:#fff;border:0;"
            "border-radius:5px;padding:4px;font-size:12px;}"
        )
        self.auto_chk.setStyleSheet("QCheckBox{color:#cbd5e1;font-size:12px;}")
        self.interval_lbl.setStyleSheet("color:#cbd5e1;font-size:12px;")
        self.status_lbl.setStyleSheet("color:#22c55e;font-size:12px;")

    def _layout_header(self):
        y = (self.HEADER_H - 26) // 2
        x = 10
        for w, gap in [
            (self.hotkey_btn, 8), (self.auto_chk, 8),
            (self.interval_slider, 4), (self.interval_lbl, 10),
            (self.scan_btn, 10),
        ]:
            w.adjustSize()
            w.move(x, y + (26 - w.sizeHint().height()) // 2)
            w.show()
            x += w.sizeHint().width() + gap
        self.close_btn.move(self.width() - 34, y)
        self.status_lbl.adjustSize()
        self.status_lbl.move(
            max(x, self.width() - 210),
            y + (26 - self.status_lbl.sizeHint().height()) // 2,
        )

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(0, 0, self.width(), self.HEADER_H, QColor(15, 23, 42, 240))
        p.fillRect(0, self.HEADER_H - 1, self.width(), 1, QColor(37, 99, 235))
        body = QRect(0, self.HEADER_H, self.width(),
                     self.height() - self.HEADER_H)
        p.fillRect(body, QColor(37, 99, 235, 30))
        p.setPen(QPen(QColor(37, 99, 235), 2))
        p.drawRect(self.rect().adjusted(1, 1, -2, -2))

    def roi_rect(self) -> QRect:
        return QRect(self.x(), self.y() + self.HEADER_H,
                     self.width(), self.height() - self.HEADER_H)

    def set_status(self, text: str, color: str = "#22c55e"):
        self.status_lbl.setText(f"● {text}")
        self.status_lbl.setStyleSheet(f"color:{color};font-size:12px;")
        self.status_lbl.adjustSize()
        self._layout_header()

    def _on_hotkey_clicked(self):
        text, ok = QInputDialog.getText(
            self, "Гаряча клавіша",
            "Комбінація (F8, ctrl+alt+m, ...):", text=self._hotkey)
        if ok and text.strip():
            self._hotkey = text.strip()
            self.hotkey_btn.setText(f"🎯 {self._hotkey}")
            self.hotkey_changed.emit(self._hotkey)
            self._layout_header()

    def _emit_auto(self, *_):
        self.interval_lbl.setText(f"{self.interval_slider.value()}s")
        self.auto_changed.emit(self.auto_chk.isChecked(),
                               self.interval_slider.value())

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
        self.setCursor(cursors.get(e, Qt.CursorShape.ArrowCursor))

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
        minw, minh = self.minimumWidth(), self.minimumHeight()
        if g.width() < minw:
            if "l" in edge: g.setLeft(g.right() - minw)
            else: g.setRight(g.left() + minw)
        if g.height() < minh:
            if "t" in edge: g.setTop(g.bottom() - minh)
            else: g.setBottom(g.top() + minh)
        self.setGeometry(g)

    def moveEvent(self, e):
        super().moveEvent(e)
        self.roi_changed.emit(self.roi_rect())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._layout_header()
        self.roi_changed.emit(self.roi_rect())


# ============================================================== UI: BADGE
class BadgeWindow(QWidget):
    clicked = pyqtSignal(dict)
    COLOR_OK = "#16a34a"
    COLOR_AI = "#d97706"

    def __init__(self, data: dict, label: str, color: str):
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
        self._color = color
        f = QFont(); f.setPointSize(9); f.setBold(True)
        self.setFont(f)
        fm = self.fontMetrics()
        self.resize(fm.horizontalAdvance(label) + 20, fm.height() + 10)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect().adjusted(0, 0, -1, -1)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(self._color))
        p.drawRoundedRect(r, 6, 6)
        p.setPen(QColor("white"))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, self._label)

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
            label, color = self._label_for(it)
            b = BadgeWindow(it, label, color)
            b.clicked.connect(self._on_click)
            b.move(self._badge_pos(it))
            b.show()
            self._badges.append(b)

    def _badge_pos(self, item: dict) -> QPoint:
        x, y, _, h = item["rect"]
        return QPoint(self._roi.x() + x, self._roi.y() + y + h + 4)

    @staticmethod
    def _label_for(item: dict) -> Tuple[str, str]:
        sol = item.get("solution")
        if sol:
            label = sol if len(sol) <= 24 else sol[:22] + "…"
            return label, BadgeWindow.COLOR_OK
        return "🤖 AI", BadgeWindow.COLOR_AI


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
        self.setWindowTitle("Деталі виразу")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.resize(560, 420)

        v = QVBoxLayout(self)
        lbl = QLabel(f"<b>Вираз:</b> <code>{item['text']}</code>")
        lbl.setTextFormat(Qt.TextFormat.RichText); lbl.setWordWrap(True)
        v.addWidget(lbl)

        sol = item.get("solution")
        s = QLabel(f"<b>Локальний розв'язок:</b> {sol}" if sol
                   else "<i>Локально не розв'язано. Спробуйте AI.</i>")
        s.setTextFormat(Qt.TextFormat.RichText)
        v.addWidget(s)

        self.browser = QTextBrowser()
        v.addWidget(self.browser, 1)

        h = QHBoxLayout()
        self.ai_btn = QPushButton("🤖 Пояснити з AI")
        self.ai_btn.setEnabled(bool(GEMINI_API_KEY))
        self.ai_btn.clicked.connect(self._request_ai)
        close_btn = QPushButton("Закрити")
        close_btn.clicked.connect(self.accept)
        h.addWidget(self.ai_btn); h.addStretch(1); h.addWidget(close_btn)
        v.addLayout(h)

        if not GEMINI_API_KEY:
            self.browser.setMarkdown(
                "> **AI недоступний:** задайте `GEMINI_API_KEY` у `.env`.")

    def _request_ai(self):
        self.ai_btn.setEnabled(False)
        self.browser.setMarkdown("_Запит до Gemini…_")
        self._worker = AIWorker(self.item["text"], parent=self)
        self._worker.done.connect(self._on_done)
        self._worker.err.connect(self._on_err)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _on_done(self, text: str):
        self.browser.setMarkdown(text); self.ai_btn.setEnabled(True)

    def _on_err(self, msg: str):
        self.browser.setMarkdown(f"**Помилка AI:** {msg}")
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
    def __init__(self, app: QApplication):
        super().__init__()
        self.app = app
        self.pool = QThreadPool.globalInstance()

        self.frame = SelectionFrame(DEFAULT_HOTKEY, DEFAULT_INTERVAL_S)
        self.overlay = OverlayManager(on_click=self._show_details)

        self.frame.roi_changed.connect(self.overlay.set_roi)
        self.frame.scan_requested.connect(self.start_scan)
        self.frame.auto_changed.connect(self._on_auto_changed)
        self.frame.hotkey_changed.connect(self._register_hotkey)
        self.frame.quit_requested.connect(self.app.quit)

        self._auto_timer = QTimer(self)
        self._auto_timer.timeout.connect(self.start_scan)

        self._hotkey_handle = None
        self._bridge = HotkeyBridge()
        self._bridge.triggered.connect(self.start_scan)
        self._register_hotkey(DEFAULT_HOTKEY)

        self.frame.show()
        self.frame.set_status("Готовий", "#22c55e")

    def _register_hotkey(self, hk: str):
        if not _HAS_KEYBOARD:
            self.frame.set_status("keyboard недоступний", "#f59e0b")
            return
        try:
            if self._hotkey_handle is not None:
                keyboard.remove_hotkey(self._hotkey_handle)
            self._hotkey_handle = keyboard.add_hotkey(
                hk, self._bridge.triggered.emit)
        except Exception as e:
            QMessageBox.warning(
                self.frame, "Хоткей",
                f"Не вдалося зареєструвати '{hk}':\n{e}\n"
                "На macOS/Linux може знадобитись sudo.")

    def _on_auto_changed(self, enabled: bool, interval: int):
        self._auto_timer.stop()
        if enabled:
            self._auto_timer.start(max(1, interval) * 1000)
            self.frame.set_status(f"Авто {interval}s", "#38bdf8")
        else:
            self.frame.set_status("Готовий", "#22c55e")

    def start_scan(self):
        roi = self.frame.roi_rect()
        if roi.width() < 30 or roi.height() < 20:
            self.frame.set_status("Замала область", "#f59e0b")
            return
        self.frame.set_status("Сканування…", "#eab308")
        task = ScanTask(roi)
        task.signals.done.connect(self._on_scan_done)
        task.signals.error.connect(self._on_scan_error)
        self.pool.start(task)

    def _on_scan_done(self, roi: QRect, items: List[dict]):
        self.overlay.show_items(roi, items)
        solved = sum(1 for it in items if it.get("solution"))
        if not items:
            self.frame.set_status("Нічого не знайдено", "#94a3b8")
        else:
            self.frame.set_status(
                f"{solved}/{len(items)} розв'язано", "#22c55e")

    def _on_scan_error(self, msg: str):
        self.frame.set_status("Помилка", "#ef4444")
        QMessageBox.warning(self.frame, "Помилка сканування", msg)

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
