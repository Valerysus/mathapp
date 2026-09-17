"""
MathLens v5 - Smart Math OCR, Local Equation Solver & Adaptive AI Explainer
- Strict text filtering (ignores instruction sentences, numbers with dots, text paragraphs)
- High-precision column-aware expression detection (splits multiple equations on one row)
- Cyrillic math variable mapping (х -> x, у -> y, а -> a, : -> /)
- Visual Bounding Boxes around detected problems
- Local answer badge + separate [ ⚡ AI ] button
- Adaptive AI explanation difficulty (Elementary, Medium, Advanced)
- Dual-window architecture (Capture Frame + Taskbar Control Panel)
"""

import os
import sys
import re
import shutil
from pathlib import Path

import mss
from PIL import Image
import pytesseract
import sympy
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)
import keyboard
from dotenv import load_dotenv

from PyQt6.QtCore import Qt, QPoint, QRect, QRectF, pyqtSignal, QThread, QTimer, QObject
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QCursor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QCheckBox,
    QSlider,
    QLabel,
    QLineEdit,
    QTextBrowser,
    QFileDialog,
    QFrame,
)

# ----------------------------------------------------------------------
# Configuration & Auto-Detection
# ----------------------------------------------------------------------
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

DEFAULT_HOTKEY = os.getenv("HOTKEY", "F8").strip() or "F8"
DEFAULT_INTERVAL = int(os.getenv("AUTO_INTERVAL_S", "3"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
GEMINI_FALLBACK_MODEL = "gemini-1.5-flash"
OCR_LANG = os.getenv("OCR_LANG", "eng+ukr").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6JSN12zl5DZ5p7ELzRtAy2Yx65DXP7PWYuAq9jWBg_ivg").strip()


def find_tesseract_cmd() -> str:
    custom = os.getenv("TESSERACT_CMD", "").strip()
    if custom and os.path.isfile(custom):
        return custom

    which_path = shutil.which("tesseract")
    if which_path:
        return which_path

    candidates = [
        "D:/programs/tesseract.exe",
        "D:/programs/Tesseract-OCR/tesseract.exe",
        "D:/Tesseract-OCR/tesseract.exe",
        "C:/Program Files/Tesseract-OCR/tesseract.exe",
        "C:/Program Files (x86)/Tesseract-OCR/tesseract.exe",
        "C:/Tesseract-OCR/tesseract.exe",
        "E:/Tesseract-OCR/tesseract.exe",
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe"),
        os.path.join(os.environ.get("USERPROFILE", ""), "AppData", "Local", "Programs", "Tesseract-OCR", "tesseract.exe"),
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return ""


TESSERACT_CMD = find_tesseract_cmd()


def save_tesseract_cmd(path: str):
    global TESSERACT_CMD
    TESSERACT_CMD = path
    lines = []
    if ENV_PATH.exists():
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()

    found = False
    new_lines = []
    for line in lines:
        if line.strip().startswith("TESSERACT_CMD="):
            new_lines.append(f"TESSERACT_CMD={path}\n")
            found = True
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f"TESSERACT_CMD={path}\n")

    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


# ----------------------------------------------------------------------
# Math Parsing, Normalization & Cyrillic Variable Mapping
# ----------------------------------------------------------------------
TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

CYRILLIC_TO_LATIN = {
    'а': 'a', 'в': 'b', 'с': 'c', 'е': 'e', 'і': 'i',
    'к': 'k', 'м': 'm', 'н': 'n', 'о': 'o', 'р': 'p',
    'т': 't', 'х': 'x', 'у': 'y',
    'А': 'A', 'В': 'B', 'С': 'C', 'Е': 'E', 'І': 'I',
    'К': 'K', 'М': 'M', 'Н': 'N', 'О': 'O', 'Р': 'P',
    'Т': 'T', 'Х': 'X', 'У': 'Y',
}


def normalize_math_symbols(raw_text: str) -> str:
    text = raw_text.strip()
    for cyr, lat in CYRILLIC_TO_LATIN.items():
        text = text.replace(cyr, lat)

    text = text.replace("×", "*").replace("✕", "*").replace("·", "*").replace("•", "*")
    text = text.replace("÷", "/").replace(":", "/")
    text = text.replace("—", "-").replace("–", "-").replace("−", "-")
    text = text.replace("²", "^2").replace("³", "^3").replace("√", "sqrt")
    text = re.sub(r"(?<=\d)[Oo](?=\d)", "0", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def solve_locally(expr_str: str) -> dict:
    """
    Solves arithmetic and algebraic equations locally using SymPy.
    """
    cleaned = normalize_math_symbols(expr_str)
    if cleaned.endswith("="):
        cleaned = cleaned[:-1].strip()
    cleaned = cleaned.replace("^", "**")

    if not cleaned or re.search(r"[^\w\s\+\-\*\/\(\)\^\.\=\,\>\<]", cleaned):
        return {"is_solved": False, "result_str": "Needs AI"}

    try:
        if "=" in cleaned:
            parts = cleaned.split("=")
            if len(parts) == 2:
                lhs_str, rhs_str = parts[0].strip(), parts[1].strip()
                if not lhs_str or not rhs_str:
                    return {"is_solved": False, "result_str": "Needs AI"}

                lhs = parse_expr(lhs_str, transformations=TRANSFORMATIONS)
                rhs = parse_expr(rhs_str, transformations=TRANSFORMATIONS)
                eq = sympy.Eq(lhs, rhs)
                syms = sorted(list(eq.free_symbols), key=lambda s: s.name)

                if syms:
                    sols = sympy.solve(eq, syms)
                    if isinstance(sols, list):
                        sol_strs = []
                        for s in sols:
                            if isinstance(s, tuple):
                                sol_strs.append(", ".join(str(item) for item in s))
                            else:
                                sol_strs.append(f"{syms[0].name} = {s}")
                        res = ", ".join(sol_strs)
                        return {"is_solved": True, "result_str": res}
                    else:
                        return {"is_solved": True, "result_str": str(sols)}
                else:
                    return {"is_solved": True, "result_str": "True" if bool(lhs == rhs) else "False"}
        else:
            parsed = parse_expr(cleaned, transformations=TRANSFORMATIONS)
            if not parsed.free_symbols:
                val = parsed.evalf()
                if abs(val - round(float(val))) < 1e-9:
                    return {"is_solved": True, "result_str": f"= {int(round(float(val)))}"}
                return {"is_solved": True, "result_str": f"≈ {float(val):.4g}"}
            else:
                simplified = sympy.simplify(parsed)
                return {"is_solved": True, "result_str": f"= {simplified}"}
    except Exception:
        return {"is_solved": False, "result_str": "Needs AI"}

    return {"is_solved": False, "result_str": "Needs AI"}


def get_complexity_level(expr: str) -> tuple[str, str]:
    """Determines math problem complexity for adaptive Gemini prompting."""
    e = expr.lower()
    if re.search(r"(sin|cos|tan|cot|log|ln|sqrt|\^3|\^[4-9])", e):
        return "Advanced", "Advanced difficulty: Provide a clear, thorough step-by-step mathematical proof/derivation."
    elif re.search(r"(\^2|\*\*2)", e) or ("/" in e and any(c.isalpha() for c in e)):
        return "Medium", "Medium algebra: Provide a structured 2-3 step algebraic solution showing the core steps."
    else:
        return "Elementary", "Elementary school level: Provide an ultra-short, simple 1-2 sentence solution (e.g. explain the inverse operation) suitable for elementary school without intro chatter."


# ----------------------------------------------------------------------
# Screen Capture & Strict Math Token Filtering
# ----------------------------------------------------------------------
def capture_screen_roi(x: int, y: int, width: int, height: int, dpr: float = 1.0) -> Image.Image:
    with mss.mss() as sct:
        monitor = {
            "top": int(y * dpr),
            "left": int(x * dpr),
            "width": max(10, int(width * dpr)),
            "height": max(10, int(height * dpr)),
        }
        sct_img = sct.grab(monitor)
        return Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")


def is_math_token(token: str) -> bool:
    """Rejects general text words, task labels (e.g. 336.), and accepts math symbols."""
    t = token.strip()
    if not t:
        return False

    # Ignore Cyrillic or Latin words with length >= 2 (like 'Розв\'яжи', 'рівняння', 'solve')
    if re.search(r"[a-zA-Zа-яіїєґА-ЯІЇЄҐ]{2,}", t):
        # Allow known math functions
        if t.lower() in ("sqrt", "sin", "cos", "tan", "log", "ln"):
            return True
        return False

    # Ignore numbered list markers like '336.', '1)', '2.'
    if re.match(r"^\d+[\.\)]$", t):
        return False

    # Accept numbers, operators, single-character variables
    if re.search(r"[\d\+\-\*\/\:\·\•\×\=\^\(\)]", t):
        return True
    if re.match(r"^[a-zA-Zа-яіїєґ]$", t):
        return True

    return False


def is_valid_math_expression(expr_str: str) -> bool:
    """Verifies that an assembled token group represents a real math problem."""
    # Must have at least one digit or two distinct variables
    has_digit = bool(re.search(r"\d", expr_str))
    has_operator = bool(re.search(r"[\+\-\*\/\:\·\•\×\=\^]", expr_str))
    if not has_operator:
        return False
    if not has_digit:
        vars_found = set(re.findall(r"[a-zA-Zа-яіїєґ]", expr_str))
        if len(vars_found) < 2:
            return False
    return True


def extract_math_blocks(image: Image.Image, min_conf: int = 15) -> list[dict]:
    tess = TESSERACT_CMD if (TESSERACT_CMD and os.path.isfile(TESSERACT_CMD)) else find_tesseract_cmd()
    if not tess:
        raise pytesseract.TesseractNotFoundError()

    pytesseract.pytesseract.tesseract_cmd = tess
    custom_cfg = r"--oem 3 --psm 6"

    try:
        data = pytesseract.image_to_data(
            image, lang=OCR_LANG, config=custom_cfg, output_type=pytesseract.Output.DICT
        )
    except pytesseract.TesseractNotFoundError:
        raise
    except Exception as e:
        if "tesseract is not installed" in str(e).lower():
            raise pytesseract.TesseractNotFoundError()
        raise e

    n_boxes = len(data["text"])
    lines_dict = {}

    for i in range(n_boxes):
        raw_word = data["text"][i].strip()
        conf = int(data["conf"][i]) if str(data["conf"][i]).isdigit() else -1
        if not raw_word or conf < min_conf:
            continue

        if not is_math_token(raw_word):
            continue

        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        tok = {
            "text": raw_word,
            "l": data["left"][i],
            "t": data["top"][i],
            "r": data["left"][i] + data["width"][i],
            "b": data["top"][i] + data["height"][i],
            "conf": conf,
        }
        if key not in lines_dict:
            lines_dict[key] = []
        lines_dict[key].append(tok)

    results = []

    # Process each line, splitting multiple equations separated by columns/large gaps
    for _, tokens in lines_dict.items():
        if not tokens:
            continue

        # Sort tokens horizontally
        tokens.sort(key=lambda item: item["l"])

        groups = []
        cur_group = []

        for tok in tokens:
            if not cur_group:
                cur_group.append(tok)
                continue

            prev = cur_group[-1]
            gap = tok["l"] - prev["r"]
            avg_h = max(12, prev["b"] - prev["t"])
            has_equals = any(t["text"] == "=" for t in cur_group)

            # Split if there is a column gap (> 25px or 1.4x line height) or a second '=' sign
            if gap > max(24, avg_h * 1.4) or (has_equals and tok["text"] == "="):
                groups.append(cur_group)
                cur_group = [tok]
            else:
                cur_group.append(tok)

        if cur_group:
            groups.append(cur_group)

        for g in groups:
            expr_str = " ".join(t["text"] for t in g)
            if is_valid_math_expression(expr_str):
                min_l = min(t["l"] for t in g)
                min_t = min(t["t"] for t in g)
                max_r = max(t["r"] for t in g)
                max_b = max(t["b"] for t in g)

                results.append({
                    "text": expr_str,
                    "x": min_l,
                    "y": min_t,
                    "w": max_r - min_l,
                    "h": max_b - min_t,
                })

    return results


# ----------------------------------------------------------------------
# Adaptive Gemini AI Explainer
# ----------------------------------------------------------------------
_gemini_client = None
_gemini_type = None


def get_gemini_client():
    global _gemini_client, _gemini_type
    if _gemini_client is not None:
        return _gemini_client, _gemini_type

    api_key = os.getenv("GEMINI_API_KEY", "").strip() or GEMINI_API_KEY
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured.")

    try:
        from google import genai
        _gemini_client = genai.Client(api_key=api_key)
        _gemini_type = "google-genai"
        return _gemini_client, _gemini_type
    except ImportError:
        pass

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        _gemini_client = genai
        _gemini_type = "google-generativeai"
        return _gemini_client, _gemini_type
    except ImportError:
        pass

    raise ImportError("Please install google-genai or google-generativeai.")


def ask_gemini_explanation(expression: str) -> tuple[str, str]:
    """
    Returns (complexity_level, explanation_text) adapting depth to math level.
    """
    level_name, level_instruction = get_complexity_level(expression)

    try:
        client, ctype = get_gemini_client()
    except Exception as e:
        return level_name, f"AI Setup Error: {str(e)}"

    prompt = (
        f"You are a math tutor. Explain the solution for: \"{expression}\".\n"
        f"Level context: {level_instruction}\n"
        "Format:\n"
        "1. Final answer.\n"
        "2. Step-by-step solution matching the requested depth.\n"
        "Language: Ukrainian (or English if formula only)."
    )

    if ctype == "google-genai":
        try:
            resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            return level_name, resp.text.strip() if resp.text else "No explanation."
        except Exception:
            try:
                resp = client.models.generate_content(model=GEMINI_FALLBACK_MODEL, contents=prompt)
                return level_name, resp.text.strip() if resp.text else "No explanation."
            except Exception as err:
                return level_name, f"Gemini Error: {str(err)}"
    elif ctype == "google-generativeai":
        try:
            model = client.GenerativeModel(GEMINI_FALLBACK_MODEL)
            resp = model.generate_content(prompt)
            return level_name, resp.text.strip() if resp.text else "No explanation."
        except Exception as err:
            return level_name, f"Gemini Error: {str(err)}"

    return level_name, "Unknown client error."


# ----------------------------------------------------------------------
# GUI: Window 1 (Pure Transparent Capture ROI Frame)
# ----------------------------------------------------------------------
class CaptureFrame(QWidget):
    region_changed = pyqtSignal(QRect)
    close_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(600, 320)
        self.move(180, 180)

        self.border_width = 4
        self.resizing = False
        self.moving = False
        self.drag_position = QPoint()
        self.active_edge = None

        top_bar = QWidget(self)
        top_bar.setFixedHeight(24)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(8, 2, 8, 2)

        lbl_drag = QLabel("⠿ ROI FRAME (drag here to move)", top_bar)
        lbl_drag.setStyleSheet("color: #000000; font-size: 11px; font-weight: bold;")
        lbl_drag.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
        top_layout.addWidget(lbl_drag)
        top_layout.addStretch()

        btn_close = QPushButton("✕", top_bar)
        btn_close.setFixedSize(20, 18)
        btn_close.setStyleSheet("""
            QPushButton {
                background: #000000;
                color: #ffffff;
                border: 1px solid #ffffff;
                border-radius: 2px;
                font-weight: bold;
                font-size: 10px;
                padding: 0;
            }
            QPushButton:hover {
                background: #d32f2f;
                border: 1px solid #d32f2f;
            }
        """)
        btn_close.clicked.connect(self.close_clicked.emit)
        top_layout.addWidget(btn_close)

        self.top_bar = top_bar
        self.setMouseTracking(True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.top_bar.setGeometry(
            self.border_width,
            self.border_width,
            self.width() - 2 * self.border_width,
            24,
        )
        self.region_changed.emit(self.get_capture_rect())

    def get_capture_rect(self) -> QRect:
        bar_h = self.top_bar.height()
        tl = self.mapToGlobal(QPoint(self.border_width, self.border_width + bar_h))
        w = max(10, self.width() - 2 * self.border_width)
        h = max(10, self.height() - 2 * self.border_width - bar_h)
        return QRect(tl.x(), tl.y(), w, h)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Header bar (solid white strip)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.drawRect(
            self.border_width,
            self.border_width,
            self.width() - 2 * self.border_width,
            self.top_bar.height(),
        )

        # Subtle dark scrim
        painter.setBrush(QBrush(QColor(0, 0, 0, 15)))
        pen = QPen(QColor("#ffffff"), self.border_width)
        painter.setPen(pen)
        painter.drawRect(
            self.border_width // 2,
            self.border_width // 2,
            self.width() - self.border_width,
            self.height() - self.border_width,
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            edge = self._detect_edge(event.pos())
            if edge:
                self.resizing = True
                self.active_edge = edge
            else:
                self.moving = True
                self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        pos = event.pos()
        if not self.resizing and not self.moving:
            edge = self._detect_edge(pos)
            self._update_cursor(edge)
        elif self.moving:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            self.region_changed.emit(self.get_capture_rect())
        elif self.resizing:
            self._resize_by_edge(event.globalPosition().toPoint())
            self.region_changed.emit(self.get_capture_rect())

    def mouseReleaseEvent(self, event):
        self.resizing = False
        self.moving = False
        self.active_edge = None
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        self.region_changed.emit(self.get_capture_rect())

    def moveEvent(self, event):
        super().moveEvent(event)
        self.region_changed.emit(self.get_capture_rect())

    def _detect_edge(self, pos: QPoint):
        b = self.border_width + 6
        w, h = self.width(), self.height()
        left, right, top, bottom = pos.x() < b, pos.x() > w - b, pos.y() < b, pos.y() > h - b
        if top and left: return "top_left"
        if top and right: return "top_right"
        if bottom and left: return "bottom_left"
        if bottom and right: return "bottom_right"
        if left: return "left"
        if right: return "right"
        if top: return "top"
        if bottom: return "bottom"
        return None

    def _update_cursor(self, edge):
        cmap = {
            "top_left": Qt.CursorShape.SizeFDiagCursor, "bottom_right": Qt.CursorShape.SizeFDiagCursor,
            "top_right": Qt.CursorShape.SizeBDiagCursor, "bottom_left": Qt.CursorShape.SizeBDiagCursor,
            "left": Qt.CursorShape.SizeHorCursor, "right": Qt.CursorShape.SizeHorCursor,
            "top": Qt.CursorShape.SizeVerCursor, "bottom": Qt.CursorShape.SizeVerCursor,
        }
        self.setCursor(QCursor(cmap.get(edge, Qt.CursorShape.ArrowCursor)))

    def _resize_by_edge(self, global_pt: QPoint):
        geo = self.geometry()
        min_w, min_h = 180, 120
        if "right" in self.active_edge: geo.setRight(max(global_pt.x(), geo.left() + min_w))
        if "bottom" in self.active_edge: geo.setBottom(max(global_pt.y(), geo.top() + min_h))
        if "left" in self.active_edge: geo.setLeft(min(global_pt.x(), geo.right() - min_w))
        if "top" in self.active_edge: geo.setTop(min(global_pt.y(), geo.bottom() - min_h))
        self.setGeometry(geo)


# ----------------------------------------------------------------------
# GUI: Window 2 (Control Panel with Windows Taskbar Presence)
# ----------------------------------------------------------------------
class ControlPanelWindow(QMainWindow):
    scan_requested = pyqtSignal()
    auto_scan_toggled = pyqtSignal(bool, int)
    hotkey_changed = pyqtSignal(str)
    toggle_frame_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MathLens - Control Panel")
        self.resize(440, 320)
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #121212; color: #ffffff; font-family: 'Segoe UI', Arial; font-size: 12px; }
            QFrame.card { background-color: #1c1c1c; border: 1px solid #333333; border-radius: 6px; }
            QPushButton { background-color: #242424; color: #ffffff; border: 1px solid #555555; border-radius: 4px; padding: 6px 12px; font-weight: bold; }
            QPushButton:hover { background-color: #333333; border: 1px solid #ffffff; }
            QPushButton.primary { background-color: #ffffff; color: #000000; border: 1px solid #ffffff; }
            QPushButton.primary:hover { background-color: #e0e0e0; }
            QLineEdit { background-color: #181818; border: 1px solid #444444; color: #ffffff; padding: 5px 8px; border-radius: 3px; }
            QLineEdit:focus { border: 1px solid #ffffff; }
            QSlider::groove:horizontal { height: 4px; background: #333333; }
            QSlider::handle:horizontal { background: #ffffff; width: 14px; margin: -5px 0; border-radius: 7px; }
            QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #555555; background: #181818; }
            QCheckBox::indicator:checked { background: #ffffff; }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        card_act = QFrame()
        card_act.setProperty("class", "card")
        act_l = QHBoxLayout(card_act)
        self.btn_scan = QPushButton(f"Analyze Now ({DEFAULT_HOTKEY})")
        self.btn_scan.setProperty("class", "primary")
        self.btn_scan.setFixedHeight(34)
        self.btn_scan.clicked.connect(self.scan_requested.emit)
        act_l.addWidget(self.btn_scan)

        self.btn_toggle = QPushButton("Hide/Show Frame")
        self.btn_toggle.setFixedHeight(34)
        self.btn_toggle.clicked.connect(self.toggle_frame_requested.emit)
        act_l.addWidget(self.btn_toggle)
        layout.addWidget(card_act)

        card_st = QFrame()
        card_st.setProperty("class", "card")
        st_l = QHBoxLayout(card_st)
        st_l.addWidget(QLabel("Status:"))
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("font-weight: bold; color: #ffffff;")
        st_l.addWidget(self.lbl_status)
        st_l.addStretch()
        layout.addWidget(card_st)

        card_set = QFrame()
        card_set.setProperty("class", "card")
        set_l = QVBoxLayout(card_set)
        set_l.setSpacing(10)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Hotkey:"))
        self.txt_hotkey = QLineEdit(DEFAULT_HOTKEY)
        self.txt_hotkey.setFixedWidth(50)
        self.txt_hotkey.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.txt_hotkey.editingFinished.connect(lambda: self.hotkey_changed.emit(self.txt_hotkey.text().strip().upper()))
        row1.addWidget(self.txt_hotkey)

        row1.addSpacing(20)
        self.chk_auto = QCheckBox("Auto-Scan")
        self.chk_auto.toggled.connect(lambda c: self.auto_scan_toggled.emit(c, self.slider.value()))
        row1.addWidget(self.chk_auto)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(1, 10)
        self.slider.setValue(DEFAULT_INTERVAL)
        self.slider.setFixedWidth(80)
        self.slider.valueChanged.connect(self._on_slider)
        row1.addWidget(self.slider)

        self.lbl_interval = QLabel(f"{DEFAULT_INTERVAL}s")
        row1.addWidget(self.lbl_interval)
        set_l.addLayout(row1)

        row2 = QVBoxLayout()
        row2.addWidget(QLabel("Tesseract Path (tesseract.exe):"))
        p_box = QHBoxLayout()
        self.txt_tess = QLineEdit(TESSERACT_CMD)
        self.txt_tess.setPlaceholderText("Select tesseract.exe path...")
        p_box.addWidget(self.txt_tess)

        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self._browse_tess)
        p_box.addWidget(self.btn_browse)
        row2.addLayout(p_box)
        set_l.addLayout(row2)

        layout.addWidget(card_set)
        layout.addStretch()

        foot = QHBoxLayout()
        btn_quit = QPushButton("Quit Application")
        btn_quit.clicked.connect(QApplication.instance().quit)
        foot.addStretch()
        foot.addWidget(btn_quit)
        layout.addLayout(foot)

    def set_status(self, text: str, is_error: bool = False):
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(f"font-weight: bold; color: {'#ff5555' if is_error else '#ffffff'};")

    def _on_slider(self, val: int):
        self.lbl_interval.setText(f"{val}s")
        if self.chk_auto.isChecked():
            self.auto_scan_toggled.emit(True, val)

    def _browse_tess(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select tesseract.exe", "C:/", "Executables (*.exe);;All Files (*.*)")
        if path:
            self.txt_tess.setText(path)
            save_tesseract_cmd(path)
            self.set_status("Tesseract path updated!", False)

    def closeEvent(self, event):
        QApplication.instance().quit()
        event.accept()


# ----------------------------------------------------------------------
# GUI: Visual Bounding Boxes & Dual Badges (Answer + AI Button)
# ----------------------------------------------------------------------
class ExplanationDialog(QDialog):
    def __init__(self, expr: str, local_res: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Solution & AI Steps")
        self.resize(520, 380)
        self.setStyleSheet("""
            QDialog { background-color: #121212; color: #ffffff; font-family: 'Segoe UI', Arial; }
            QTextBrowser { background-color: #1c1c1c; color: #e0e0e0; border: 1px solid #333333; padding: 12px; font-size: 13px; border-radius: 4px; }
            QPushButton { background-color: #ffffff; color: #000000; font-weight: bold; padding: 6px 16px; border-radius: 4px; border: none; }
            QPushButton:hover { background-color: #cccccc; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        header_box = QHBoxLayout()
        lbl_expr = QLabel(f"Expression: {expr}")
        lbl_expr.setStyleSheet("font-size: 14px; font-weight: bold;")
        header_box.addWidget(lbl_expr)
        header_box.addStretch()

        self.lbl_diff = QLabel("Detecting...")
        self.lbl_diff.setStyleSheet("color: #aaaaaa; font-size: 11px; border: 1px solid #444; border-radius: 3px; padding: 2px 6px;")
        header_box.addWidget(self.lbl_diff)
        layout.addLayout(header_box)

        if local_res and "Needs" not in local_res:
            lbl_ans = QLabel(f"Local Solver Result: {local_res}")
            lbl_ans.setStyleSheet("color: #81c784; font-weight: bold;")
            layout.addWidget(lbl_ans)

        self.browser = QTextBrowser()
        self.browser.setMarkdown("Analyzing problem with Gemini AI...")
        layout.addWidget(self.browser)

        foot = QHBoxLayout()
        foot.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        foot.addWidget(btn_close)
        layout.addLayout(foot)

        # Fetch adaptive explanation
        level_name, explanation = ask_gemini_explanation(expr)
        self.lbl_diff.setText(f"Level: {level_name}")
        self.browser.setMarkdown(explanation)


class MathItemWidget(QWidget):
    """
    Renders the local answer badge and a separate [ ⚡ AI ] button right next to it.
    """
    def __init__(self, expr_text: str, result_info: dict, parent=None):
        super().__init__(parent)
        self.expr_text = expr_text
        self.result_info = result_info
        is_solved = result_info.get("is_solved", False)
        res_str = result_info.get("result_str", "Needs AI")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        if is_solved:
            # Answer badge
            self.lbl_ans = QLabel(f"[ {res_str} ]")
            self.lbl_ans.setStyleSheet("""
                QLabel {
                    background-color: #000000;
                    color: #ffffff;
                    font-family: 'Consolas', monospace;
                    font-weight: bold;
                    font-size: 12px;
                    border: 1px solid #ffffff;
                    border-radius: 3px;
                    padding: 2px 6px;
                }
            """)
            layout.addWidget(self.lbl_ans)

            # Separate small AI button
            self.btn_ai = QPushButton("⚡ AI")
            self.btn_ai.setStyleSheet("""
                QPushButton {
                    background-color: #242424;
                    color: #ffffff;
                    font-size: 11px;
                    font-weight: bold;
                    border: 1px solid #666666;
                    border-radius: 3px;
                    padding: 2px 6px;
                }
                QPushButton:hover {
                    background-color: #ffffff;
                    color: #000000;
                    border: 1px solid #ffffff;
                }
            """)
            self.btn_ai.setCursor(Qt.CursorShape.PointingHandCursor)
            self.btn_ai.clicked.connect(self._open_dialog)
            layout.addWidget(self.btn_ai)
        else:
            # Fallback button for complex problems
            self.btn_ai = QPushButton("⚡ Solve with AI")
            self.btn_ai.setStyleSheet("""
                QPushButton {
                    background-color: #1a1a1a;
                    color: #ffffff;
                    font-size: 11px;
                    font-weight: bold;
                    border: 1px dashed #ffffff;
                    border-radius: 3px;
                    padding: 2px 8px;
                }
                QPushButton:hover {
                    background-color: #ffffff;
                    color: #000000;
                }
            """)
            self.btn_ai.setCursor(Qt.CursorShape.PointingHandCursor)
            self.btn_ai.clicked.connect(self._open_dialog)
            layout.addWidget(self.btn_ai)

    def _open_dialog(self):
        dlg = ExplanationDialog(self.expr_text, self.result_info.get("result_str", ""), self.window())
        dlg.exec()


class HUDOverlay(QWidget):
    """
    Transparent HUD overlay that outlines math problems with bounding boxes
    and positions Answer badges + AI buttons neatly around them.
    """
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self.items_data = []
        self.widgets = []
        self.hide()

    def sync_with_roi(self, rect: QRect):
        self.setGeometry(rect)

    def clear_results(self):
        for w in self.widgets:
            w.deleteLater()
        self.widgets.clear()
        self.items_data.clear()
        self.update()

    def display_results(self, items: list[dict], dpr: float = 1.0):
        self.clear_results()
        self.items_data = items

        for item in items:
            # Convert physical image pixels to Qt logical coordinates
            lx = int(item["x"] / dpr)
            ly = int(item["y"] / dpr)
            lw = int(item["w"] / dpr)
            lh = int(item["h"] / dpr)

            item["lx"] = lx
            item["ly"] = ly
            item["lw"] = lw
            item["lh"] = lh

            widget = MathItemWidget(item["text"], item["solution"], self)
            widget.adjustSize()

            # Place widget above the bounding box if space allows, otherwise below
            bw = widget.sizeHint().width()
            bh = widget.sizeHint().height()

            if ly - bh - 4 >= 0:
                pos_x = min(self.width() - bw - 4, max(2, lx))
                pos_y = ly - bh - 4
            else:
                pos_x = min(self.width() - bw - 4, max(2, lx))
                pos_y = ly + lh + 4

            widget.setGeometry(pos_x, pos_y, bw, bh)
            widget.show()
            self.widgets.append(widget)

        self.show()
        self.update()

    def paintEvent(self, event):
        """Draws crisp bounding box outlines around every detected math expression."""
        if not self.items_data:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # High-contrast rounded rectangle outline around the problem
        pen = QPen(QColor(255, 255, 255, 230), 1.5, Qt.PenStyle.SolidLine)
        brush = QBrush(QColor(0, 0, 0, 30))
        painter.setPen(pen)
        painter.setBrush(brush)

        for item in self.items_data:
            lx = item.get("lx", 0)
            ly = item.get("ly", 0)
            lw = item.get("lw", 0)
            lh = item.get("lh", 0)
            # Add small padding around the text
            painter.drawRoundedRect(QRectF(lx - 3, ly - 2, lw + 6, lh + 4), 3, 3)


# ----------------------------------------------------------------------
# Application Controller & Worker
# ----------------------------------------------------------------------
class ScanWorker(QThread):
    finished_scan = pyqtSignal(list, float)
    status_changed = pyqtSignal(str, bool)

    def __init__(self, rect: QRect, dpr: float):
        super().__init__()
        self.rect = rect
        self.dpr = dpr

    def run(self):
        self.status_changed.emit("Scanning...", False)
        try:
            img = capture_screen_roi(
                self.rect.x(), self.rect.y(), self.rect.width(), self.rect.height(), dpr=self.dpr
            )
            blocks = extract_math_blocks(img)

            solved_count = 0
            for block in blocks:
                sol = solve_locally(block["text"])
                block["solution"] = sol
                if sol.get("is_solved"):
                    solved_count += 1

            self.status_changed.emit(f"{solved_count}/{len(blocks)} solved locally", False)
            self.finished_scan.emit(blocks, self.dpr)

        except pytesseract.TesseractNotFoundError:
            self.status_changed.emit("Tesseract not found! Click 'Browse...' in Control Panel", True)
            self.finished_scan.emit([], self.dpr)
        except Exception as e:
            err = str(e)
            if "tesseract is not installed" in err.lower():
                self.status_changed.emit("Tesseract not found! Click 'Browse...' in Control Panel", True)
            else:
                self.status_changed.emit(f"Error: {err[:35]}", True)
            self.finished_scan.emit([], self.dpr)


class HotkeyBridge(QObject):
    triggered = pyqtSignal()


class MathLensApp:
    def __init__(self):
        self.frame = CaptureFrame()
        self.panel = ControlPanelWindow()
        self.overlay = HUDOverlay()
        self.worker = None

        screen = QApplication.primaryScreen()
        self.dpr = screen.devicePixelRatio() if screen else 1.0

        self.timer = QTimer()
        self.timer.timeout.connect(self.trigger_scan)

        self.frame.region_changed.connect(self.overlay.sync_with_roi)
        self.overlay.sync_with_roi(self.frame.get_capture_rect())

        self.panel.scan_requested.connect(self.trigger_scan)
        self.panel.auto_scan_toggled.connect(self.on_auto_scan)
        self.panel.hotkey_changed.connect(self.setup_hotkey)
        self.panel.toggle_frame_requested.connect(self.toggle_frame)
        self.frame.close_clicked.connect(self.toggle_frame)

        self.hotkey_bridge = HotkeyBridge()
        self.hotkey_bridge.triggered.connect(self.trigger_scan)
        self.setup_hotkey(DEFAULT_HOTKEY)

        self.panel.show()
        self.frame.show()

    def toggle_frame(self):
        if self.frame.isVisible():
            self.frame.hide()
            self.overlay.hide()
        else:
            self.frame.show()
            self.overlay.sync_with_roi(self.frame.get_capture_rect())

    def trigger_scan(self):
        if self.worker and self.worker.isRunning():
            return
        screen = QApplication.primaryScreen()
        self.dpr = screen.devicePixelRatio() if screen else 1.0
        self.worker = ScanWorker(self.frame.get_capture_rect(), self.dpr)
        self.worker.finished_scan.connect(self.overlay.display_results)
        self.worker.status_changed.connect(self.panel.set_status)
        self.worker.start()

    def on_auto_scan(self, enabled: bool, interval: int):
        if enabled:
            self.timer.start(interval * 1000)
            self.panel.set_status("Auto-Scan Active", False)
        else:
            self.timer.stop()
            self.panel.set_status("Ready", False)

    def setup_hotkey(self, key: str):
        try:
            keyboard.clear_all_hotkeys()
            keyboard.add_hotkey(key, lambda: self.hotkey_bridge.triggered.emit())
            self.panel.set_status(f"Hotkey: {key}", False)
        except Exception:
            self.panel.set_status("Hotkey Inactive", True)


def main():
    app = QApplication(sys.argv)
    _ = MathLensApp()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
