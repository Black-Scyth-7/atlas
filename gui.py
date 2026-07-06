"""JARVIS-style holographic dashboard for Atlas (PySide6).

Run this INSTEAD of main.py to get a window:
    python gui.py

A dense holographic HUD, modelled on the reference image: a central ATOM (glowing
nucleus + electron orbits inside concentric tick-rings) surrounded by a grid of
chamfered "data panels" — molecule wireframes, a neural-network graph, bar
charts, radar/sonar gauges, a rotating wireframe cube, and telemetry text — over
a dark, bokeh-lit field. Two of the panels are live: SYSTEM (who's signed in,
model, wake word, mic, state) and TRANSCRIPT (the running conversation). The
atom's colour tracks Atlas's state and its nucleus pulses with the mic/speech
level; the decorative panels animate purely for atmosphere.

How it works: the normal assistant loop (`main.main()`) runs unchanged on a
worker thread and emits lightweight events through `ui_events`. This module
subscribes once and re-emits them as Qt signals, so all widget updates happen on
the GUI thread. The terminal app is untouched — with no GUI running, those emits
are no-ops. Closing the window exits the whole app.
"""

from __future__ import annotations

import math
import os
import random
import subprocess
import sys
import threading
import time
import traceback
from collections import deque

from PySide6.QtCore import QObject, QPointF, QRectF, QThread, QTimer, Qt, Signal
from PySide6.QtGui import (
    QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF, QRadialGradient,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication, QFrame, QGridLayout, QLabel, QLineEdit, QMainWindow,
    QSizePolicy, QTextEdit, QVBoxLayout, QWidget,
)

import ui_events as ux
from config import GuiConfig

# The whole HUD lives in one blue/cyan family so it reads as a single hologram.
# Kept deliberately dark/subdued — only the atom's nucleus is a bright focal point.
CYAN = (58, 138, 190)
BLUE = (46, 108, 160)
DIM = (36, 82, 120)


# ---- worker thread: run the real assistant loop unchanged ---------------
class AssistantThread(QThread):
    """Runs main.main() (wake -> record -> think -> speak) off the GUI thread."""

    def run(self) -> None:
        try:
            import main
            main.main()
        except Exception:                       # surface a crash, don't die silent
            traceback.print_exc()
            ux.status(error="assistant stopped — see console")
            ux.set_state("idle")


# ---- event bridge: ui_events (worker thread) -> Qt signals (GUI thread) --
class Bridge(QObject):
    state = Signal(str)
    user = Signal(str, str)
    delta = Signal(str)
    done = Signal()
    level = Signal(float)
    status = Signal(dict)
    mode = Signal(bool)
    ready = Signal()
    llm = Signal(list)
    tool = Signal(str, str, str)

    def __init__(self) -> None:
        super().__init__()
        ux.subscribe(self._on_event)

    def _on_event(self, kind: str, data: dict) -> None:
        if kind == "state":
            self.state.emit(data.get("state", "idle"))
        elif kind == "user":
            self.user.emit(data.get("text", ""), data.get("lang", ""))
        elif kind == "delta":
            self.delta.emit(data.get("text", ""))
        elif kind == "done":
            self.done.emit()
        elif kind == "level":
            self.level.emit(data.get("rms", 0.0))
        elif kind == "status":
            self.status.emit(dict(data))
        elif kind == "text_mode":
            self.mode.emit(bool(data.get("on")))
        elif kind == "ready":
            self.ready.emit()
        elif kind == "llm":
            self.llm.emit(data.get("trace", []))
        elif kind == "tool":
            self.tool.emit(data.get("name", ""), data.get("args", ""),
                           data.get("status", "run"))


# ---- dark, bokeh-lit background field -----------------------------------
class HudBackground(QWidget):
    """Deep radial-gradient field speckled with soft bokeh + corner brackets."""

    def __init__(self) -> None:
        super().__init__()
        rng = random.Random(7)   # fixed seed -> stable bokeh layout
        self._bokeh = []
        for _ in range(52):
            self._bokeh.append((
                rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0),
                rng.uniform(6, 34), rng.uniform(0.03, 0.13),
                rng.choice([(110, 160, 210), (90, 140, 195),
                            (200, 175, 130), (200, 165, 120)]),
                rng.uniform(0, math.tau),
            ))
        self._phase = 0.0

    def tick(self) -> None:
        self._phase += 0.03

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        bg = QRadialGradient(w * 0.5, h * 0.46, max(w, h) * 0.75)
        bg.setColorAt(0.0, QColor(9, 18, 30))
        bg.setColorAt(0.55, QColor(4, 10, 18))
        bg.setColorAt(1.0, QColor(1, 3, 6))
        p.fillRect(self.rect(), bg)
        p.setPen(Qt.NoPen)
        for fx, fy, r, a, (cr, cg, cb), ph in self._bokeh:
            tw = 0.6 + 0.4 * math.sin(self._phase + ph)
            cx, cy = fx * w, fy * h
            g = QRadialGradient(cx, cy, r)
            c0 = QColor(cr, cg, cb); c0.setAlpha(int(255 * a * tw))
            c1 = QColor(cr, cg, cb); c1.setAlpha(0)
            g.setColorAt(0.0, c0); g.setColorAt(1.0, c1)
            p.setBrush(g)
            p.drawEllipse(cx - r, cy - r, 2 * r, 2 * r)
        pen = QPen(QColor(50, 110, 155, 65)); pen.setWidthF(1.4)
        p.setPen(pen)
        m, ln = 14, 44
        for x, sx in ((m, 1), (w - m, -1)):
            for y, sy in ((m, 1), (h - m, -1)):
                p.drawLine(x, y, x + sx * ln, y)
                p.drawLine(x, y, x, y + sy * ln)
        p.end()


# ---- base chamfered card (shared by decorative + functional panels) -----
def _chamfer_path(w: float, h: float, c: float = 12.0) -> QPainterPath:
    path = QPainterPath()
    path.moveTo(c, 1); path.lineTo(w - c, 1); path.lineTo(w - 1, c)
    path.lineTo(w - 1, h - c); path.lineTo(w - c, h - 1); path.lineTo(c, h - 1)
    path.lineTo(1, h - c); path.lineTo(1, c); path.closeSubpath()
    return path


def _paint_frame(p: QPainter, w: int, h: int, title: str) -> None:
    path = _chamfer_path(w, h)
    p.fillPath(path, QColor(5, 12, 21, 175))
    pen = QPen(QColor(46, 108, 155, 150)); pen.setWidthF(1.2)
    p.setPen(pen); p.setBrush(Qt.NoBrush); p.drawPath(path)
    p.setPen(QPen(QColor(80, 160, 205, 190), 2.0))
    p.drawLine(14, 1, 44, 1)
    if title:
        p.setPen(QColor(80, 150, 195, 190))
        p.setFont(QFont("Consolas", 7, QFont.Bold))
        p.drawText(14, 17, title.upper())


class HudCard(QFrame):
    """A chamfered panel that animates a decorative graphic (draw_content)."""

    def __init__(self, title: str = "", seed: int = 0, min_w: int = 188,
                 min_h: int = 116) -> None:
        super().__init__()
        self._title = title
        self.seed = seed
        self.phase = 0.0
        self.setMinimumSize(min_w, min_h)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._seed()

    def _seed(self) -> None:
        pass

    def tick(self) -> None:
        self.phase += 0.03
        self.update()

    def draw_content(self, p: QPainter, x: float, y: float,
                     w: float, h: float) -> None:
        pass

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        _paint_frame(p, self.width(), self.height(), self._title)
        top = 24 if self._title else 10
        rx, ry = 12, top
        rw, rh = self.width() - 24, self.height() - top - 10
        if rw > 4 and rh > 4:
            p.save()
            p.setClipRect(QRectF(rx, ry, rw, rh))
            self.draw_content(p, rx, ry, rw, rh)
            p.restore()
        p.end()


# ---- decorative panels ---------------------------------------------------
class MoleculeCard(HudCard):
    """A slowly rotating 3D node cluster with bonds — a 'molecule' hologram."""

    def _seed(self) -> None:
        rng = random.Random(self.seed)
        n = rng.randint(9, 13)
        self.nodes = [(rng.uniform(-1, 1), rng.uniform(-1, 1),
                       rng.uniform(-1, 1)) for _ in range(n)]
        self.edges = []
        for i in range(n):
            for j in range(i + 1, n):
                d = math.dist(self.nodes[i], self.nodes[j])
                if d < 0.95:
                    self.edges.append((i, j))

    def tick(self) -> None:
        self.phase += 0.012
        self.update()

    def draw_content(self, p, x, y, w, h) -> None:
        cx, cy = x + w / 2, y + h / 2
        s = min(w, h) * 0.42
        a = self.phase
        ca, sa = math.cos(a), math.sin(a)
        pts = []
        for X, Y, Z in self.nodes:
            xr = X * ca - Z * sa
            zr = X * sa + Z * ca
            pts.append((cx + xr * s, cy + Y * s, zr))
        for i, j in self.edges:
            x1, y1, z1 = pts[i]; x2, y2, z2 = pts[j]
            al = int(30 + 45 * ((z1 + z2) / 2 + 1) / 2)
            pen = QPen(QColor(*CYAN, al)); pen.setWidthF(1.0)
            p.setPen(pen)
            p.drawLine(x1, y1, x2, y2)
        p.setPen(Qt.NoPen)
        for px, py, z in pts:
            depth = (z + 1) / 2
            rad = 1.6 + 2.6 * depth
            g = QRadialGradient(px, py, rad * 2.6)
            g.setColorAt(0.0, QColor(150, 195, 225, int(110 + 80 * depth)))
            g.setColorAt(1.0, QColor(*CYAN, 0))
            p.setBrush(g)
            p.drawEllipse(px - rad * 2.6, py - rad * 2.6, rad * 5.2, rad * 5.2)
            p.setBrush(QColor(160, 200, 225, int(110 + 80 * depth)))
            p.drawEllipse(px - rad, py - rad, 2 * rad, 2 * rad)


class NetworkCard(HudCard):
    """A node-link 'neural network' graph with signals travelling the edges."""

    def _seed(self) -> None:
        rng = random.Random(self.seed + 100)
        n = rng.randint(10, 14)
        self.nodes = [(rng.uniform(0.1, 0.9), rng.uniform(0.12, 0.9))
                      for _ in range(n)]
        self.edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if math.dist(self.nodes[i], self.nodes[j]) < 0.42:
                    self.edges.append((i, j))
        self.pulses = [(rng.randrange(len(self.edges)) if self.edges else 0,
                        rng.uniform(0.4, 1.0), rng.uniform(0, 1))
                       for _ in range(min(5, len(self.edges)))]

    def draw_content(self, p, x, y, w, h) -> None:
        def pos(i):
            fx, fy = self.nodes[i]
            return x + fx * w, y + fy * h
        pen = QPen(QColor(*BLUE, 70)); pen.setWidthF(1.0)
        p.setPen(pen)
        for i, j in self.edges:
            x1, y1 = pos(i); x2, y2 = pos(j)
            p.drawLine(x1, y1, x2, y2)
        p.setPen(Qt.NoPen)
        for ei, sp, off in self.pulses:
            if not self.edges:
                break
            i, j = self.edges[ei % len(self.edges)]
            x1, y1 = pos(i); x2, y2 = pos(j)
            t = (self.phase * sp + off) % 1.0
            px, py = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
            p.setBrush(QColor(150, 195, 225, 190))
            p.drawEllipse(px - 2.2, py - 2.2, 4.4, 4.4)
        for i in range(len(self.nodes)):
            px, py = pos(i)
            g = QRadialGradient(px, py, 7)
            g.setColorAt(0.0, QColor(150, 195, 225, 180))
            g.setColorAt(1.0, QColor(*CYAN, 0))
            p.setBrush(g)
            p.drawEllipse(px - 7, py - 7, 14, 14)
            p.setBrush(QColor(160, 200, 225, 200))
            p.drawEllipse(px - 2.4, py - 2.4, 4.8, 4.8)


def _conf_color(p: float) -> tuple[int, int, int]:
    """Map a probability/confidence 0..1 to a colour: red (uncertain) through
    amber to cyan (confident). Used for the live logit readout."""
    p = max(0.0, min(1.0, p))
    if p < 0.5:
        f = p / 0.5
        return (228, int(96 + 120 * f), 96)
    f = (p - 0.5) / 0.5
    return (int(228 - 150 * f), int(216 - 8 * f), int(96 + 140 * f))


class NeuralActivityCard(HudCard):
    """LIVE, REAL model telemetry — not decoration.

    Plays back the per-token logit trace emitted by llm.py for the reply Atlas
    just generated (see ui_events.llm_activity). Shows three genuine signals:
      • a confidence heat-strip of the most recent tokens (green = the model was
        sure, red = uncertain),
      • the top-k candidate 'race' for the current token — the actual
        alternatives the model weighed, with their probabilities,
      • a running confidence/entropy readout.
    With no trace yet it rests on a dim flatline."""

    def _seed(self) -> None:
        self.trace: list = []
        self._cursor = 0.0          # playback head (tokens)
        self._speed = 0.5           # tokens advanced per frame (~lively)

    def set_trace(self, trace: list) -> None:
        self.trace = trace or []
        self._cursor = 0.0

    def tick(self) -> None:
        self.phase += 0.03
        if self.trace:
            self._cursor = min(self._cursor + self._speed, len(self.trace) - 1)
        self.update()

    def draw_content(self, p, x, y, w, h) -> None:
        if not self.trace:
            self._draw_resting(p, x, y, w, h)
            return
        idx = max(0, min(int(self._cursor), len(self.trace) - 1))
        cur = self.trace[idx]

        # --- header: current token + confidence + entropy ---
        p.setFont(QFont("Consolas", 7, QFont.Bold))
        conf = float(cur.get("p", 0.0))
        p.setPen(QColor(120, 175, 205, 210))
        p.drawText(int(x), int(y + 9),
                   f"tok \"{cur.get('t', '')}\"")
        cr, cg, cb = _conf_color(conf)
        p.setPen(QColor(cr, cg, cb, 235))
        p.drawText(int(x), int(y + 20),
                   f"p={conf*100:4.0f}%  H={float(cur.get('e', 0.0)):.2f}")

        # --- confidence heat-strip: a window of recent tokens up to the head ---
        strip_y = y + 26
        strip_h = 12
        win = 20
        lo = max(0, idx - win + 1)
        cells = self.trace[lo:idx + 1]
        if cells:
            cw = w / win
            for i, tk in enumerate(cells):
                pv = float(tk.get("p", 0.0))
                r, g, b = _conf_color(pv)
                bx = x + i * cw
                newest = (lo + i == idx)
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(r, g, b, 235 if newest else 150))
                p.drawRect(QRectF(bx, strip_y, max(1.0, cw - 1.5), strip_h))
                if newest:
                    p.setBrush(QColor(230, 245, 255, 230))
                    p.drawRect(QRectF(bx, strip_y, max(1.0, cw - 1.5), 2))

        # --- candidate 'race' for the current token ---
        cands = cur.get("k", [])[:5]
        top = y + 44
        avail = (y + h) - top - 2
        if cands and avail > 8:
            rh = min(15.0, avail / len(cands))
            label_w = 44.0
            bar_x = x + label_w
            bar_w = max(10.0, w - label_w - 26)
            p.setFont(QFont("Consolas", 7))
            for i, cand in enumerate(cands):
                tok = str(cand[0]) if cand else ""
                prob = float(cand[1]) if len(cand) > 1 else 0.0
                ry = top + i * rh
                chosen = (tok == cur.get("t", "")) or (i == 0 and tok == "")
                r, g, b = _conf_color(prob)
                # token label
                p.setPen(QColor(200, 225, 240, 230) if chosen
                         else QColor(110, 155, 180, 190))
                p.drawText(int(x), int(ry + rh - 3), f"{tok:<6.6}")
                # track + bar
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(40, 70, 95, 90))
                p.drawRect(QRectF(bar_x, ry + 1, bar_w, rh - 3))
                p.setBrush(QColor(r, g, b, 235 if chosen else 165))
                p.drawRect(QRectF(bar_x, ry + 1, bar_w * max(0.0, min(1.0, prob)),
                                  rh - 3))
                # probability %
                p.setPen(QColor(r, g, b, 220))
                p.drawText(int(bar_x + bar_w + 3), int(ry + rh - 3),
                           f"{prob*100:2.0f}")

    def _draw_resting(self, p, x, y, w, h) -> None:
        """No trace yet — a calm scanning flatline."""
        cy = y + h / 2
        pen = QPen(QColor(*BLUE, 70)); pen.setWidthF(1.0)
        p.setPen(pen)
        prev = None
        for i in range(0, int(w), 4):
            v = math.sin(i * 0.08 + self.phase * 2.0) * 2.0
            pt = (x + i, cy + v)
            if prev:
                p.drawLine(prev[0], prev[1], pt[0], pt[1])
            prev = pt
        sweep = (self.phase * 40) % (w + 20) - 10
        p.setPen(Qt.NoPen)
        g = QRadialGradient(x + sweep, cy, 10)
        g.setColorAt(0.0, QColor(150, 195, 225, 170))
        g.setColorAt(1.0, QColor(*CYAN, 0))
        p.setBrush(g)
        p.drawEllipse(x + sweep - 10, cy - 10, 20, 20)
        p.setFont(QFont("Consolas", 7))
        p.setPen(QColor(80, 130, 165, 150))
        p.drawText(int(x), int(y + h - 3), "awaiting inference…")


class SystemResourceCard(HudCard):
    """LIVE system load — real telemetry, not decoration.

    CPU / RAM / GPU / VRAM utilisation as eased bars. Readings are gathered on a
    daemon background thread (psutil for CPU+RAM, `nvidia-smi` for the GPU) so
    the GUI thread never blocks on a subprocess; the bars glide toward the
    latest values. Bars warm from cyan → amber → red as load climbs."""

    _ROWS = (("CPU", "cpu"), ("RAM", "ram"), ("GPU", "gpu"), ("VRAM", "vram"))
    _POLL_S = 1.5

    def _seed(self) -> None:
        self._target = {k: 0.0 for _, k in self._ROWS}
        self._disp = {k: 0.0 for _, k in self._ROWS}
        self._gpu_ok = True
        threading.Thread(target=self._poll_loop, daemon=True).start()

    # ---- background sampling (never touches Qt) ----
    def _poll_loop(self) -> None:
        try:
            import psutil
        except Exception:
            psutil = None
        if psutil is not None:
            try:
                psutil.cpu_percent(interval=None)      # prime the delta counter
            except Exception:
                pass
        while True:
            if psutil is not None:
                try:
                    self._target["cpu"] = psutil.cpu_percent(interval=None) / 100.0
                    self._target["ram"] = psutil.virtual_memory().percent / 100.0
                except Exception:
                    pass
            self._poll_gpu()
            time.sleep(self._POLL_S)

    def _poll_gpu(self) -> None:
        if not self._gpu_ok:
            return
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2.5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            util, used, total = (float(v) for v in
                                 out.stdout.strip().splitlines()[0].split(","))
            self._target["gpu"] = max(0.0, min(1.0, util / 100.0))
            self._target["vram"] = max(0.0, min(1.0, used / total)) if total else 0.0
        except Exception:
            self._gpu_ok = False       # no usable GPU readout — leave those bars at 0

    # ---- render ----
    def tick(self) -> None:
        self.phase += 0.03
        for k in self._disp:
            self._disp[k] += (self._target[k] - self._disp[k]) * 0.15
        self.update()

    def draw_content(self, p, x, y, w, h) -> None:
        rh = h / len(self._ROWS)
        label_w, val_w = 36.0, 32.0
        bar_x = x + label_w
        bar_w = max(10.0, w - label_w - val_w)
        p.setFont(QFont("Consolas", 7, QFont.Bold))
        for i, (lab, key) in enumerate(self._ROWS):
            v = max(0.0, min(1.0, self._disp[key]))
            cy = y + i * rh + rh / 2
            r, g, b = _conf_color(1.0 - v)      # more load -> warmer/red
            p.setPen(QColor(120, 175, 205, 210))
            p.drawText(int(x), int(cy + 3), lab)
            bh = min(9.0, rh - 8)
            by = cy - bh / 2
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(40, 70, 95, 90))
            p.drawRect(QRectF(bar_x, by, bar_w, bh))
            p.setBrush(QColor(r, g, b, 225))
            p.drawRect(QRectF(bar_x, by, bar_w * v, bh))
            p.setBrush(QColor(220, 245, 255, 210))
            p.drawRect(QRectF(bar_x + bar_w * v - 1.0, by, 2.0, bh))  # leading edge
            p.setPen(QColor(r, g, b, 235))
            p.drawText(int(bar_x + bar_w + 4), int(cy + 3), f"{v * 100:3.0f}%")


class ToolActivityCard(HudCard):
    """LIVE tool-call readout — real, not decoration.

    Shows which tool Atlas is reaching for right now (see ui_events.tool_activity
    / Tools.execute): the current tool name with a status dot that pulses amber
    while running and settles green (ok), red (fail) or orange (denied), its
    argument preview, and a short log of recent calls. Rests when idle."""

    _COLORS = {"run": (230, 180, 90), "ok": (120, 210, 150),
               "fail": (230, 110, 90), "deny": (215, 150, 80)}
    _GLYPH = {"run": "▶", "ok": "✓", "fail": "✗", "deny": "⊘"}

    def _seed(self) -> None:
        self.current: dict | None = None
        self.history: deque = deque(maxlen=4)   # (name, status) most-recent last

    def set_tool(self, name: str, args: str, status: str) -> None:
        self.current = {"name": name, "args": args, "status": status}
        if status in ("ok", "fail", "deny"):
            self.history.append((name, status))

    def _sc(self, status: str) -> tuple[int, int, int]:
        return self._COLORS.get(status, (120, 175, 205))

    def draw_content(self, p, x, y, w, h) -> None:
        if self.current is None:
            p.setFont(QFont("Consolas", 8))
            p.setPen(QColor(80, 130, 165, 150))
            p.drawText(int(x), int(y + h / 2), "idle — no tool calls yet")
            return

        cur = self.current
        r, g, b = self._sc(cur["status"])
        # --- status dot (pulses while running) ---
        pulse = 0.55 + 0.45 * math.sin(self.phase * 4.0) if cur["status"] == "run" else 1.0
        dot_r = 4.0
        dcx, dcy = x + dot_r + 1, y + 9
        gg = QRadialGradient(dcx, dcy, dot_r * 3)
        gg.setColorAt(0.0, QColor(r, g, b, int(200 * pulse)))
        gg.setColorAt(1.0, QColor(r, g, b, 0))
        p.setPen(Qt.NoPen); p.setBrush(gg)
        p.drawEllipse(dcx - dot_r * 3, dcy - dot_r * 3, dot_r * 6, dot_r * 6)
        p.setBrush(QColor(r, g, b, int(120 + 135 * pulse)))
        p.drawEllipse(dcx - dot_r, dcy - dot_r, 2 * dot_r, 2 * dot_r)

        # --- current tool name ---
        p.setFont(QFont("Consolas", 10, QFont.Bold))
        p.setPen(QColor(min(255, r + 40), min(255, g + 40), min(255, b + 40), 240))
        p.drawText(int(dcx + dot_r + 6), int(dcy + 4), str(cur["name"]))

        # --- argument preview ---
        if cur["args"]:
            p.setFont(QFont("Consolas", 7))
            p.setPen(QColor(110, 155, 180, 190))
            p.drawText(int(x), int(y + 24), str(cur["args"]))

        # --- recent-call log ---
        top = y + 38
        if self.history and top < y + h - 6:
            p.setFont(QFont("Consolas", 7, QFont.Bold))
            p.setPen(QColor(73, 113, 134, 190))
            p.drawText(int(x), int(top), "RECENT")
            p.setFont(QFont("Consolas", 7))
            ly = top + 12
            for name, status in reversed(self.history):
                if ly > y + h - 2:
                    break
                cr, cg, cb = self._sc(status)
                p.setPen(QColor(cr, cg, cb, 220))
                p.drawText(int(x), int(ly), self._GLYPH.get(status, "·"))
                p.setPen(QColor(150, 185, 205, 200))
                p.drawText(int(x + 12), int(ly), str(name)[:20])
                ly += 12


class BarsCard(HudCard):
    """An animated bar chart — telemetry bars breathing in and out."""

    def _seed(self) -> None:
        rng = random.Random(self.seed + 200)
        self.n = 16
        self.ph = [rng.uniform(0, math.tau) for _ in range(self.n)]
        self.sp = [rng.uniform(0.7, 1.6) for _ in range(self.n)]

    def draw_content(self, p, x, y, w, h) -> None:
        gap = 3
        bw = (w - gap * (self.n - 1)) / self.n
        p.setPen(Qt.NoPen)
        for i in range(self.n):
            v = 0.15 + 0.85 * (0.5 + 0.5 * math.sin(self.phase * self.sp[i] + self.ph[i]))
            bh = v * h
            bx = x + i * (bw + gap)
            by = y + h - bh
            al = int(120 + 110 * v)
            p.setBrush(QColor(*CYAN, al))
            p.drawRect(QRectF(bx, by, bw, bh))
            p.setBrush(QColor(220, 245, 255, 230))
            p.drawRect(QRectF(bx, by, bw, 2))


class RadarCard(HudCard):
    """A sonar sweep: concentric rings, a rotating fading wedge, and blips."""

    def _seed(self) -> None:
        rng = random.Random(self.seed + 300)
        self.blips = [(rng.uniform(0, 360), rng.uniform(0.25, 0.92))
                      for _ in range(rng.randint(4, 7))]

    def draw_content(self, p, x, y, w, h) -> None:
        cx, cy = x + w / 2, y + h / 2
        R = min(w, h) * 0.46
        pen = QPen(QColor(*BLUE, 90)); pen.setWidthF(1.0)
        p.setPen(pen); p.setBrush(Qt.NoBrush)
        for f in (0.4, 0.7, 1.0):
            p.drawEllipse(QPointF(cx, cy), R * f, R * f)
        p.drawLine(cx - R, cy, cx + R, cy)
        p.drawLine(cx, cy - R, cx, cy + R)
        sweep = (self.phase * 55) % 360
        for k in range(20):
            a = math.radians(sweep - k * 3)
            al = max(0, 150 - k * 8)
            pen = QPen(QColor(*CYAN, al)); pen.setWidthF(1.6)
            p.setPen(pen)
            p.drawLine(cx, cy, cx + R * math.cos(a), cy + R * math.sin(a))
        p.setPen(Qt.NoPen)
        for ang, rf in self.blips:
            diff = (sweep - ang) % 360
            glow = max(0.0, 1.0 - diff / 70.0) if diff < 70 else 0.0
            bx = cx + R * rf * math.cos(math.radians(ang))
            by = cy + R * rf * math.sin(math.radians(ang))
            al = int(60 + 195 * glow)
            r = 2.0 + 2.5 * glow
            p.setBrush(QColor(210, 245, 255, al))
            p.drawEllipse(bx - r, by - r, 2 * r, 2 * r)


class CubeCard(HudCard):
    """A rotating 3D wireframe cube."""

    def _seed(self) -> None:
        self.verts = [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1)
                      for sz in (-1, 1)]
        self.edges = []
        for i in range(8):
            for j in range(i + 1, 8):
                if sum(a != b for a, b in zip(self.verts[i], self.verts[j])) == 1:
                    self.edges.append((i, j))

    def tick(self) -> None:
        self.phase += 0.018
        self.update()

    def draw_content(self, p, x, y, w, h) -> None:
        cx, cy = x + w / 2, y + h / 2
        s = min(w, h) * 0.34
        ax, ay = self.phase * 0.7, self.phase
        cx1, sx1 = math.cos(ax), math.sin(ax)
        cy1, sy1 = math.cos(ay), math.sin(ay)
        pts = []
        for X, Y, Z in self.verts:
            y1 = Y * cx1 - Z * sx1
            z1 = Y * sx1 + Z * cx1
            x1 = X * cy1 + z1 * sy1
            z2 = -X * sy1 + z1 * cy1
            persp = 1.0 / (2.4 - z2 * 0.5)
            pts.append((cx + x1 * s * persp * 1.4, cy + y1 * s * persp * 1.4, z2))
        pen = QPen(QColor(*CYAN, 150)); pen.setWidthF(1.3)
        p.setPen(pen)
        for i, j in self.edges:
            x1, y1, _ = pts[i]; x2, y2, _ = pts[j]
            p.drawLine(x1, y1, x2, y2)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(155, 200, 225, 200))
        for px, py, _ in pts:
            p.drawEllipse(px - 1.8, py - 1.8, 3.6, 3.6)


class TelemetryCard(HudCard):
    """Scrolling hex/decimal telemetry lines with a live sparkline footer."""

    def _seed(self) -> None:
        self.rng = random.Random(self.seed + 400)
        self.rows = [self._row() for _ in range(7)]
        self.spark = deque([0.5] * 40, maxlen=40)
        self._c = 0

    def _row(self) -> str:
        return (f"0x{self.rng.randrange(0xFFFF):04X}  "
                f"{self.rng.uniform(0, 99):5.2f}  "
                f"{''.join(self.rng.choice('▁▂▃▄▅▆▇') for _ in range(6))}")

    def tick(self) -> None:
        self.phase += 0.03
        self._c += 1
        if self._c % 8 == 0:                     # jitter one row for a live feel
            self.rows[self.rng.randrange(len(self.rows))] = self._row()
        self.spark.append(0.5 + 0.45 * math.sin(self.phase * 1.3)
                          + self.rng.uniform(-0.08, 0.08))
        self.update()

    def draw_content(self, p, x, y, w, h) -> None:
        p.setFont(QFont("Consolas", 8))
        lh = 14
        for i, row in enumerate(self.rows):
            yy = y + 12 + i * lh
            if yy > y + h - 20:
                break
            p.setPen(QColor(110, 165, 195, 200) if i % 2 else QColor(80, 135, 170, 180))
            p.drawText(int(x), int(yy), row)
        # sparkline footer
        base = y + h - 4
        amp = 16
        pen = QPen(QColor(*CYAN, 180)); pen.setWidthF(1.3)
        p.setPen(pen)
        pts = list(self.spark)
        step = w / max(1, len(pts) - 1)
        prev = None
        for i, v in enumerate(pts):
            px = x + i * step
            py = base - max(0.0, min(1.0, v)) * amp
            if prev is not None:
                p.drawLine(prev[0], prev[1], px, py)
            prev = (px, py)


# ---- the living core (centre-piece) -------------------------------------
class LivingCore(QWidget):
    """An organic 'living' core rather than a rigid atom: a morphing membrane,
    swirling thought-motes, and firing synapses around a bright nucleus.

    A single smoothed `activity` level (driven by state + audio) governs how
    alive it looks — idle is a slow calm breath; THINKING drives fast erratic
    morphing of the membrane, churning motes, rapid neuron-like firing, and a
    flickering core, so it clearly reads as actively 'thinking'. Colour comes
    from the state; the nucleus is the one bright focal point. Kept centred."""

    def __init__(self, cfg: GuiConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.setMinimumSize(360, 360)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._state = "idle"
        self._level = 0.0        # smoothed audio level
        self._target = 0.0       # latest audio level, decays each frame
        self._activity = 0.14    # smoothed "how alive / how hard it's thinking"
        self._t = 0.0            # organic time (flows faster when active)
        self._pulse = 0.0        # phase for emanating thought-rings
        self._flicker = 0.0      # core flicker seed
        self._rng = random.Random(11)
        # thought-motes: each swirls on its own drifting orbit
        self.motes = []
        for _ in range(24):
            self.motes.append({
                "a0": self._rng.uniform(0, math.tau),
                "r0": self._rng.uniform(0.42, 0.92),
                "aspd": self._rng.uniform(0.15, 0.5) * self._rng.choice((-1, 1)),
                "rphase": self._rng.uniform(0, math.tau),
                "rspd": self._rng.uniform(0.6, 1.7),
                "ramp": self._rng.uniform(0.05, 0.17),
                "sz": self._rng.uniform(1.4, 3.0),
            })
        self.sparks = []         # active synapse firings: [i, j, life]

    def set_state(self, state: str) -> None:
        if state in self.cfg.state_colors:
            self._state = state

    def push_level(self, rms: float) -> None:
        self._target = max(self._target, min(1.0, rms * 3.5))

    def tick(self) -> None:
        self._level += (self._target - self._level) * self.cfg.level_smoothing
        self._target *= self.cfg.level_decay
        # how alive should it look right now?
        st = self._state
        if st == "thinking":
            base = 1.0
        elif st == "listening":
            base = 0.45
        elif st == "speaking":
            base = 0.35 + 0.6 * self._level
        else:
            base = 0.14
        self._activity += (base - self._activity) * 0.06
        act = self._activity
        self._t += 0.010 * (0.5 + 1.7 * act)     # time quickens when thinking (medium)
        self._pulse += 0.006 + 0.028 * act
        self._flicker += (self._rng.uniform(-1, 1) - self._flicker) * 0.25
        # synapses fire — decay old ones, spawn new ones at a rate set by activity
        for sp in self.sparks:
            sp[2] -= 0.035 + 0.05 * act
        self.sparks = [sp for sp in self.sparks if sp[2] > 0][:28]
        for _ in range(2):
            if self._rng.random() < act * 0.32 and len(self.motes) > 1:
                i = self._rng.randrange(len(self.motes))
                j = self._rng.randrange(len(self.motes))
                if i != j:
                    self.sparks.append([i, j, 1.0])
        self.update()

    def _mote_xy(self, m: dict, R: float, act: float) -> tuple[float, float]:
        a = m["a0"] + self._t * m["aspd"] * (0.4 + 1.2 * act)
        rr = R * (m["r0"] + m["ramp"] * math.sin(self._t * m["rspd"] + m["rphase"])
                  * (0.3 + 1.1 * act))
        return rr * math.cos(a), rr * math.sin(a)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        R = min(w, h) * self.cfg.orb_max_frac
        lvl = self._level
        act = self._activity
        t = self._t
        r, g, b = self.cfg.orb_color or self.cfg.state_colors.get(
            self._state, (70, 170, 240))
        hi = (min(255, r + 80), min(255, g + 80), min(255, b + 90))
        p.translate(cx, cy)

        # --- outer soft glow (breathes with activity + audio) ---
        glow_r = R * (1.5 + 0.3 * act + 0.3 * lvl)
        grad = QRadialGradient(0, 0, glow_r)
        c0 = QColor(r, g, b); c0.setAlpha(int(55 + 70 * act + 60 * lvl))
        cmid = QColor(r, g, b); cmid.setAlpha(22)
        c1 = QColor(r, g, b); c1.setAlpha(0)
        grad.setColorAt(0.0, c0); grad.setColorAt(0.5, cmid); grad.setColorAt(1.0, c1)
        p.setPen(Qt.NoPen); p.setBrush(grad)
        p.drawEllipse(-glow_r, -glow_r, 2 * glow_r, 2 * glow_r)

        # --- faint containment ring (subtle HUD framing, gently rotating) ---
        p.save(); p.rotate(math.degrees(t) * 0.15)
        ring_r = R * 1.32
        for i in range(60):
            a = math.radians(i * 6)
            r1 = ring_r + (R * 0.05 if i % 5 == 0 else R * 0.025)
            pen = QPen(QColor(r, g, b, 55 if i % 5 == 0 else 28))
            pen.setWidthF(1.4 if i % 5 == 0 else 1.0)
            p.setPen(pen)
            p.drawLine(ring_r * math.cos(a), ring_r * math.sin(a),
                       r1 * math.cos(a), r1 * math.sin(a))
        p.restore()

        # --- emanating thought-rings (more/brighter while thinking) ---
        p.setBrush(Qt.NoBrush)
        for k in range(3):
            frac = (self._pulse + k / 3.0) % 1.0
            rr = R * (0.55 + frac * 1.05)
            al = int((1.0 - frac) * (30 + 130 * act))
            if al > 3:
                pen = QPen(QColor(r, g, b, al)); pen.setWidthF(1.2)
                p.setPen(pen)
                p.drawEllipse(-rr, -rr, 2 * rr, 2 * rr)

        # --- morphing organic membrane (agitated when thinking) ---
        N = 128
        amp = R * (0.05 + 0.17 * act + 0.05 * lvl)
        base_r = R * 0.7
        poly = QPolygonF()
        for k in range(N):
            ang = math.tau * k / N
            d = (math.sin(2 * ang + t * 0.9) * 0.5
                 + math.sin(3 * ang - t * 1.3) * 0.34
                 + math.sin(5 * ang + t * 1.7) * 0.22
                 + math.sin(7 * ang - t * 2.1) * 0.14)
            rr = base_r + amp * d
            poly.append(QPointF(rr * math.cos(ang), rr * math.sin(ang)))
        path = QPainterPath(); path.addPolygon(poly); path.closeSubpath()
        mg = QRadialGradient(0, 0, base_r * 1.25)
        mg.setColorAt(0.0, QColor(r, g, b, int(55 + 70 * act)))
        mg.setColorAt(0.62, QColor(r, g, b, 26))
        mg.setColorAt(1.0, QColor(r, g, b, 0))
        p.setPen(Qt.NoPen); p.setBrush(mg); p.drawPath(path)
        pen = QPen(QColor(*hi, int(110 + 110 * act))); pen.setWidthF(1.6)
        p.setPen(pen); p.setBrush(Qt.NoBrush); p.drawPath(path)

        # --- positions of the thought-motes this frame ---
        pos = [self._mote_xy(m, R, act) for m in self.motes]

        # --- firing synapses (bright, brief; frequent while thinking) ---
        for i, j, life in self.sparks:
            x1, y1 = pos[i]; x2, y2 = pos[j]
            pen = QPen(QColor(*hi, int(200 * life))); pen.setWidthF(0.8 + 1.4 * life)
            p.setPen(pen)
            p.drawLine(x1, y1, x2, y2)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(220, 240, 255, int(210 * life)))
            p.drawEllipse(x2 - 2, y2 - 2, 4, 4)

        # --- thought-motes swirling around the core ---
        p.setPen(Qt.NoPen)
        for m, (x, y) in zip(self.motes, pos):
            rad = m["sz"] * (0.8 + 0.6 * act)
            gg = QRadialGradient(x, y, rad * 3.2)
            gg.setColorAt(0.0, QColor(195, 222, 245, int(90 + 110 * act)))
            gg.setColorAt(1.0, QColor(r, g, b, 0))
            p.setBrush(gg)
            p.drawEllipse(x - rad * 3.2, y - rad * 3.2, rad * 6.4, rad * 6.4)
            p.setBrush(QColor(205, 230, 248, int(130 + 100 * act)))
            p.drawEllipse(x - rad, y - rad, 2 * rad, 2 * rad)

        # --- nucleus: the one bright focal point; flickers while thinking ---
        flick = 1.0 + act * (0.12 * math.sin(t * 7.0) + 0.06 * self._flicker)
        core_r = R * (0.15 + 0.10 * lvl) * flick
        cgrad = QRadialGradient(0, 0, core_r)
        inner = QColor(min(255, r + 120), min(255, g + 120), min(255, b + 120))
        cgrad.setColorAt(0.0, QColor(245, 251, 255))
        cgrad.setColorAt(0.4, inner)
        cgrad.setColorAt(1.0, QColor(r, g, b, 150))
        p.setBrush(cgrad)
        p.drawEllipse(-core_r, -core_r, 2 * core_r, 2 * core_r)
        p.end()


# ---- functional chamfered panel (holds child widgets) -------------------
class HudPanel(QFrame):
    def __init__(self, title: str = "") -> None:
        super().__init__()
        self._title = title
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 28 if title else 12, 14, 12)
        lay.setSpacing(4)
        self._lay = lay

    def add(self, w: QWidget) -> None:
        self._lay.addWidget(w)

    def add_stretch(self) -> None:
        self._lay.addStretch(1)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        _paint_frame(p, self.width(), self.height(), self._title)
        p.end()


# ---- main window ---------------------------------------------------------
class AtlasWindow(QMainWindow):
    _LABELS = {"idle": "IDLE", "listening": "LISTENING",
               "thinking": "THINKING", "speaking": "SPEAKING"}

    def __init__(self, cfg: GuiConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._status = {}
        self._atlas_open = False
        self.setWindowTitle("ATLAS")
        self.resize(cfg.window_w, cfg.window_h)
        self.setStyleSheet("QMainWindow { background: #02050a; }")

        self.bg = HudBackground()
        self.setCentralWidget(self.bg)
        root = QVBoxLayout(self.bg)
        root.setContentsMargins(16, 12, 16, 14)
        root.setSpacing(8)

        title = QLabel("A  T  L  A  S")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Consolas", 20, QFont.Bold))
        title.setStyleSheet(
            "color: #5aa6cc; letter-spacing: 10px; background: transparent;")
        root.addWidget(title)

        grid = QGridLayout()
        grid.setSpacing(10)
        root.addLayout(grid, stretch=1)

        # decorative panels flanking the atom (2 columns each side, 3 rows)
        self._cards = [
            SystemResourceCard("System load", seed=1),
            BarsCard("Signal", seed=2),
            RadarCard("Scan α", seed=3),
            ToolActivityCard("Tool activity", seed=4),
            NeuralActivityCard("LLM · logits", seed=5),
            TelemetryCard("Telemetry", seed=6),
            MoleculeCard("Structure γ", seed=7),
            CubeCard("Lattice", seed=8),
            RadarCard("Scan β", seed=9),
        ]
        c = self._cards
        self.neural = c[4]          # the live logit panel (real model telemetry)
        self.tools_card = c[3]      # the live tool-call panel
        # left block (cols 0,1)
        grid.addWidget(c[0], 0, 0); grid.addWidget(c[3], 0, 1)
        grid.addWidget(c[1], 1, 0); grid.addWidget(c[4], 1, 1)
        grid.addWidget(c[2], 2, 0)
        self.sys_panel = self._build_system_panel()
        grid.addWidget(self.sys_panel, 2, 1)

        # centre atom (col 2, spans all 3 rows)
        self.orb = LivingCore(cfg)
        grid.addWidget(self.orb, 0, 2, 3, 1)

        # right block (cols 3,4)
        grid.addWidget(c[6], 0, 3); grid.addWidget(c[8], 0, 4)
        grid.addWidget(c[5], 1, 3); grid.addWidget(c[7], 1, 4)
        self.conv_panel, self.transcript = self._build_transcript_panel()
        grid.addWidget(self.conv_panel, 2, 3, 1, 2)

        grid.setColumnStretch(2, 1)
        for col in (0, 1, 3, 4):
            grid.setColumnMinimumWidth(col, 176)
        for rw in range(3):
            grid.setRowStretch(rw, 1)

        # state label under the atom
        self.state_label = QLabel("● IDLE")
        self.state_label.setAlignment(Qt.AlignCenter)
        self.state_label.setFont(QFont("Consolas", 14, QFont.Bold))
        self.state_label.setStyleSheet("background: transparent;")
        root.addWidget(self.state_label)

        # type-in box: hidden until F1
        self.input = QLineEdit()
        self.input.setFont(QFont("Consolas", 12))
        self.input.setPlaceholderText("type a command, then Enter  (F1 to exit)")
        self.input.setStyleSheet(
            "QLineEdit { background: rgba(10,20,32,180); color: #eaf6ff; border: "
            "1px solid #2a8; border-radius: 8px; padding: 7px; }")
        self.input.returnPressed.connect(self._submit)
        self.input.hide()
        root.addWidget(self.input)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(max(10, 1000 // max(1, cfg.fps)))

        self.bridge = Bridge()
        self.bridge.state.connect(self._on_state)
        self.bridge.user.connect(self._on_user)
        self.bridge.delta.connect(self._on_delta)
        self.bridge.done.connect(self._on_done)
        self.bridge.level.connect(self.orb.push_level)
        self.bridge.status.connect(self._on_status)
        self.bridge.mode.connect(self._on_mode)
        self.bridge.ready.connect(self._reveal)
        self.bridge.llm.connect(self.neural.set_trace)
        self.bridge.tool.connect(self.tools_card.set_tool)
        self._on_state("idle")

    def _reveal(self) -> None:
        """Show the window once Atlas is online + ready (idempotent)."""
        if not self.isVisible():
            self.show()
        self.raise_()
        self.activateWindow()

    # ---- functional panels ----
    def _build_system_panel(self) -> HudPanel:
        panel = HudPanel("System")
        self._sys_rows: dict[str, QLabel] = {}
        for key, label in (("user", "USER"), ("authority", "AUTHORITY"),
                           ("model", "MODEL"), ("wake_word", "WAKE WORD"),
                           ("mic", "MIC"), ("state", "STATE")):
            row = QWidget(); row.setStyleSheet("background: transparent;")
            rl = QVBoxLayout(row); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(0)
            k = QLabel(label)
            k.setFont(QFont("Consolas", 7, QFont.Bold))
            k.setStyleSheet("color: #497186; letter-spacing: 2px; background: transparent;")
            v = QLabel("—")
            v.setFont(QFont("Consolas", 10))
            v.setStyleSheet("color: #86adbf; background: transparent;")
            rl.addWidget(k); rl.addWidget(v)
            panel.add(row)
            self._sys_rows[key] = v
        panel.add_stretch()
        self.err_label = QLabel("")
        self.err_label.setWordWrap(True)
        self.err_label.setFont(QFont("Consolas", 8))
        self.err_label.setStyleSheet("color: #ff8a8a; background: transparent;")
        panel.add(self.err_label)
        return panel

    def _build_transcript_panel(self):
        panel = HudPanel("Transcript")
        te = QTextEdit()
        te.setReadOnly(True)
        te.setFont(QFont("Consolas", 10))
        te.setStyleSheet(
            "QTextEdit { background: transparent; color: #97bccb; border: none; }")
        panel.add(te)
        return panel, te

    def _tick(self) -> None:
        self.bg.tick(); self.bg.update()
        self.orb.tick()
        for card in self._cards:
            card.tick()

    # ---- slots (GUI thread) ----
    def _on_state(self, state: str) -> None:
        self.orb.set_state(state)
        r, g, b = self.cfg.state_colors.get(state, (140, 200, 220))
        self.state_label.setText(f"● {self._LABELS.get(state, state.upper())}")
        self.state_label.setStyleSheet(
            f"color: rgb({r},{g},{b}); background: transparent;")
        if "state" in self._sys_rows:
            self._sys_rows["state"].setText(self._LABELS.get(state, state.upper()))
            self._sys_rows["state"].setStyleSheet(
                f"color: rgb({r},{g},{b}); background: transparent;")

    def _append(self, prefix: str, text: str, color: str) -> None:
        self.transcript.moveCursor(QTextCursor.End)
        if self.transcript.toPlainText():
            self.transcript.insertPlainText("\n")
        self.transcript.setTextColor(QColor(color))
        self.transcript.insertPlainText(prefix)
        self.transcript.setTextColor(QColor("#a6c4d2"))
        self.transcript.insertPlainText(text)
        self._trim()
        self.transcript.moveCursor(QTextCursor.End)

    def _on_user(self, text: str, _lang: str) -> None:
        self._atlas_open = False
        self._append("you:   ", text, "#4a9ec8")

    def _on_delta(self, text: str) -> None:
        if not self._atlas_open:
            self._append("atlas: ", "", "#5a86bf")
            self._atlas_open = True
        self.transcript.moveCursor(QTextCursor.End)
        self.transcript.setTextColor(QColor("#a6c4d2"))
        self.transcript.insertPlainText(text)
        self.transcript.moveCursor(QTextCursor.End)

    def _on_done(self) -> None:
        self._atlas_open = False

    def _on_mode(self, on: bool) -> None:
        self.input.setVisible(on)
        if on:
            self.activateWindow(); self.raise_(); self.input.setFocus()
        else:
            self.input.clear()

    def _submit(self) -> None:
        text = self.input.text().strip()
        if text:
            ux.submit_text(text)
            self.input.clear()

    def _on_status(self, data: dict) -> None:
        self._status.update(data)
        s = self._status
        for key in ("user", "authority", "model", "wake_word"):
            if key in self._sys_rows and key in s:
                self._sys_rows[key].setText(str(s.get(key, "—")))
        if "mic" in self._sys_rows and "mic" in s:
            self._sys_rows["mic"].setText("ONLINE ●" if s.get("mic") else "OFF ○")
        if s.get("error"):
            self.err_label.setText(f"⚠ {s['error']}")
            self._reveal()      # surface the window if the assistant crashed

    def _trim(self) -> None:
        doc = self.transcript.document()
        extra = doc.blockCount() - self.cfg.transcript_max_lines
        if extra > 0:
            cur = QTextCursor(doc)
            cur.movePosition(QTextCursor.Start)
            for _ in range(extra):
                cur.select(QTextCursor.BlockUnderCursor)
                cur.removeSelectedText()
                cur.deleteChar()

    def closeEvent(self, event) -> None:
        os._exit(0)


def main() -> None:
    cfg = GuiConfig()
    app = QApplication(sys.argv)
    win = AtlasWindow(cfg)
    # The window stays hidden while models load + identity is verified; it
    # reveals itself when the assistant emits `ready` (or if it crashes). The
    # console still shows load progress in the meantime.
    worker = AssistantThread()
    worker.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
