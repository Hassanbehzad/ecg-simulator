"""
ECG Signal Simulator & Viewer — Complete Single File
=====================================================
Team: Hammad (Signal), Hassan (Visualizer), Talha (Controls), Bilal (Integration)

Install deps:
    pip install numpy matplotlib

Run:
    python ecg_simulator.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, CheckButtons
import csv
import threading


# ============================================================
#  MODULE 1 — ecg_generator.py  (Hammad — Signal Engineer)
# ============================================================

def get_ecg_signal(heart_rate=75, duration=5, fs=360, noise_level=0.0):
    """
    Generate a synthetic ECG signal using Gaussian (bell-curve) functions.

    Each of the 5 waves (P, Q, R, S, T) is modeled as a Gaussian:
        amplitude * exp( -(t - center)^2 / (2 * width^2) )

    Negative amplitude = downward deflection (Q and S waves).

    Args:
        heart_rate  : beats per minute (int, 40–180)
        duration    : total signal length in seconds
        fs          : sampling frequency in Hz (360 Hz = standard ECG)
        noise_level : 0.0 = clean signal, ~0.08 = mild artifact

    Returns:
        t   : np.ndarray — time axis in seconds
        ecg : np.ndarray — amplitude in millivolts (mV)
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    ecg = np.zeros_like(t)

    # (amplitude_mV, gaussian_width_s, time_offset_from_beat_start_s)
    pqrst = {
        'P': ( 0.25, 0.090, 0.00),   # Small bump — atrial depolarisation
        'Q': (-0.10, 0.030, 0.14),   # Small dip — start of ventricular
        'R': ( 1.60, 0.025, 0.20),   # Tall sharp spike — ventricular depol.
        'S': (-0.30, 0.030, 0.26),   # Negative dip — end of QRS complex
        'T': ( 0.35, 0.120, 0.42),   # Broad hump — ventricular repolarisation
    }

    beat_period = 60.0 / heart_rate          # seconds per beat
    beat_starts = np.arange(0, duration, beat_period)

    for beat_t in beat_starts:
        for wave, (amp, width, offset) in pqrst.items():
            center = beat_t + offset
            ecg += amp * np.exp(-((t - center) ** 2) / (2 * width ** 2))

    if noise_level > 0:
        ecg += noise_level * np.random.randn(len(t))

    return t, ecg


# ============================================================
#  MODULE 2 — visualizer.py  (Hassan — Visualization Lead)
# ============================================================

class ECGVisualizer:
    """
    Live scrolling ECG plot built with Matplotlib FuncAnimation.
    Handles waveform rendering, PQRST labels, and grid styling.
    """

    FS          = 360          # sampling frequency (Hz)
    WINDOW_SEC  = 5            # seconds visible in the plot
    BG_COLOR    = '#0d1117'    # dark background
    GRID_COLOR  = '#1f2a1f'    # subtle green grid
    TRACE_COLOR = '#00e676'    # bright green ECG trace
    LABEL_COLOR = '#ffd54f'    # amber wave labels
    TEXT_COLOR  = '#b0bec5'    # muted axis text

    def __init__(self):
        self.heart_rate  = 75
        self.noise_level = 0.0
        self.paused      = False
        self._anim       = None

        self._build_figure()

    # ── Figure setup ──────────────────────────────────────────

    def _build_figure(self):
        self.fig = plt.figure(
            figsize=(14, 8),
            facecolor=self.BG_COLOR,
            num="ECG Signal Simulator"
        )
        self.fig.canvas.manager.set_window_title("ECG Signal Simulator — BIOS R&D")

        # Main ECG axes (top portion, room for controls below)
        self.ax = self.fig.add_axes([0.06, 0.38, 0.90, 0.54])
        self._style_axes()

        # Initial waveform
        t, ecg = get_ecg_signal(self.heart_rate, self.WINDOW_SEC, self.FS, self.noise_level)
        self.line, = self.ax.plot(t, ecg, color=self.TRACE_COLOR, linewidth=1.4, zorder=3)
        self.ax.set_xlim(0, self.WINDOW_SEC)
        self.ax.set_ylim(-0.7, 2.1)

        # Annotate first beat
        self._annotate_waves(t, ecg)

        # Info text (top-left corner of plot)
        self.info_text = self.ax.text(
            0.01, 0.95, self._info_str(),
            transform=self.ax.transAxes,
            fontsize=10, color=self.TEXT_COLOR,
            verticalalignment='top', fontfamily='monospace'
        )

    def _style_axes(self):
        self.ax.set_facecolor(self.BG_COLOR)
        self.ax.tick_params(colors=self.TEXT_COLOR, labelsize=9)
        for spine in self.ax.spines.values():
            spine.set_edgecolor('#263238')
        self.ax.set_xlabel('Time (s)', color=self.TEXT_COLOR, fontsize=10)
        self.ax.set_ylabel('Amplitude (mV)', color=self.TEXT_COLOR, fontsize=10)
        self.ax.set_title(
            'ECG Signal Simulator — Live Viewer',
            color='#eceff1', fontsize=13, fontweight='bold', pad=12
        )
        self.ax.grid(True, color=self.GRID_COLOR, linewidth=0.6, zorder=0)
        # Horizontal baseline at 0 mV
        self.ax.axhline(0, color='#37474f', linewidth=0.8, zorder=1)

    def _annotate_waves(self, t, ecg):
        """Place P, R, T labels on the first beat."""
        self._wave_annotations = []
        waves = {'P': (0.00, 0.15), 'R': (0.20, 0.12), 'T': (0.42, 0.15)}
        beat_period = 60.0 / self.heart_rate
        for label, (offset, dy) in waves.items():
            idx = int((beat_period * 0 + offset) * self.FS)
            if 0 < idx < len(ecg):
                ann = self.ax.annotate(
                    label,
                    xy=(t[idx], ecg[idx]),
                    xytext=(t[idx], ecg[idx] + dy),
                    color=self.LABEL_COLOR, fontsize=9, fontweight='bold',
                    ha='center',
                    arrowprops=dict(arrowstyle='->', color=self.LABEL_COLOR,
                                    lw=0.8, mutation_scale=10)
                )
                self._wave_annotations.append(ann)

    def _info_str(self):
        status = "PAUSED" if self.paused else "LIVE"
        return (f"HR: {self.heart_rate} BPM  |  "
                f"Noise: {'ON' if self.noise_level > 0 else 'OFF'}  |  "
                f"Fs: {self.FS} Hz  |  [{status}]")

    def _refresh_annotations(self, t, ecg):
        for ann in self._wave_annotations:
            ann.remove()
        self._annotate_waves(t, ecg)

    # ── Animation ─────────────────────────────────────────────

    def update(self, frame):
        """FuncAnimation callback — redraws trace each frame."""
        if self.paused:
            return self.line,
        t, ecg = get_ecg_signal(self.heart_rate, self.WINDOW_SEC, self.FS, self.noise_level)
        self.line.set_ydata(ecg)
        self.info_text.set_text(self._info_str())
        return self.line,

    def start(self):
        self._anim = animation.FuncAnimation(
            self.fig, self.update, interval=800, blit=False, cache_frame_data=False
        )
        plt.show()

    # ── Public setters (called by Controls) ───────────────────

    def set_heart_rate(self, hr):
        self.heart_rate = int(hr)

    def set_noise(self, on: bool):
        self.noise_level = 0.08 if on else 0.0

    def toggle_pause(self):
        self.paused = not self.paused

    def export_csv(self):
        t, ecg = get_ecg_signal(self.heart_rate, 10, self.FS, self.noise_level)
        filename = f"ecg_export_hr{self.heart_rate}.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time_s', 'amplitude_mV'])
            writer.writerows(zip(np.round(t, 5), np.round(ecg, 5)))
        print(f"[ECG] Exported {len(t)} samples → {filename}")


# ============================================================
#  MODULE 3 — controls.py  (Talha — Controls & Features)
# ============================================================

class ECGControls:
    """
    Matplotlib-native control panel embedded below the ECG plot.
    Uses Slider, Button, and CheckButtons widgets — no Tkinter needed,
    so everything runs in one window with zero threading issues.
    """

    BG      = '#0d1117'
    AX_BG   = '#161b22'
    TEXT    = '#b0bec5'
    GREEN   = '#00e676'
    AMBER   = '#ffd54f'
    RED     = '#ef5350'

    def __init__(self, fig, viewer: ECGVisualizer):
        self.fig    = fig
        self.viewer = viewer
        self._build_controls()

    def _make_ax(self, rect, label=''):
        ax = self.fig.add_axes(rect, facecolor=self.AX_BG)
        if label:
            ax.set_title(label, color=self.TEXT, fontsize=8, pad=3)
        return ax

    def _build_controls(self):
        # ── Heart Rate Slider ──
        ax_hr = self._make_ax([0.10, 0.24, 0.55, 0.04], 'Heart Rate (BPM)')
        self.slider_hr = Slider(
            ax_hr, '', 40, 180, valinit=75, valstep=1,
            color=self.GREEN, initcolor=self.GREEN
        )
        self.slider_hr.label.set_color(self.TEXT)
        self.slider_hr.valtext.set_color(self.GREEN)
        self.slider_hr.valtext.set_fontsize(11)
        self.slider_hr.on_changed(self._on_hr_change)

        # BPM range labels
        self.fig.text(0.09, 0.257, '40', color=self.TEXT, fontsize=8, ha='right')
        self.fig.text(0.67, 0.257, '180', color=self.TEXT, fontsize=8)

        # ── Noise Toggle (CheckButton) ──
        ax_noise = self._make_ax([0.70, 0.20, 0.12, 0.10])
        self.check_noise = CheckButtons(ax_noise, ['Add Noise'], [False])
        self.check_noise.on_clicked(self._on_noise_toggle)
        # Style checkbutton text
        for txt in self.check_noise.labels:
            txt.set_color(self.TEXT)
            txt.set_fontsize(9)

        # ── Pause / Play Button ──
        ax_pause = self._make_ax([0.10, 0.10, 0.14, 0.08])
        self.btn_pause = Button(ax_pause, '|| Pause',
                                color=self.AX_BG, hovercolor='#1f2a1f')
        self.btn_pause.label.set_color(self.AMBER)
        self.btn_pause.label.set_fontsize(10)
        self.btn_pause.on_clicked(self._on_pause)

        # ── Export CSV Button ──
        ax_export = self._make_ax([0.28, 0.10, 0.16, 0.08])
        self.btn_export = Button(ax_export, '[S] Export CSV',
                                 color=self.AX_BG, hovercolor='#1f2a1f')
        self.btn_export.label.set_color(self.GREEN)
        self.btn_export.label.set_fontsize(10)
        self.btn_export.on_clicked(lambda e: self.viewer.export_csv())

        # ── Reset Button ──
        ax_reset = self._make_ax([0.48, 0.10, 0.12, 0.08])
        self.btn_reset = Button(ax_reset, '↺  Reset',
                                color=self.AX_BG, hovercolor='#1f2a1f')
        self.btn_reset.label.set_color(self.RED)
        self.btn_reset.label.set_fontsize(10)
        self.btn_reset.on_clicked(self._on_reset)

        # ── Status bar ──
        self.status = self.fig.text(
            0.10, 0.04,
            'Ready  —  adjust heart rate slider or toggle noise',
            color=self.TEXT, fontsize=9, fontfamily='monospace'
        )

    # ── Callbacks ─────────────────────────────────────────────

    def _on_hr_change(self, val):
        self.viewer.set_heart_rate(val)
        self.status.set_text(f'Heart rate set to {int(val)} BPM')

    def _on_noise_toggle(self, label):
        on = self.check_noise.get_status()[0]
        self.viewer.set_noise(on)
        self.status.set_text(f'Noise artifact {"enabled" if on else "disabled"}')

    def _on_pause(self, event):
        self.viewer.toggle_pause()
        paused = self.viewer.paused
        self.btn_pause.label.set_text('> Play' if paused else '|| Pause')
        self.status.set_text('Paused' if paused else 'Resumed — live update active')

    def _on_reset(self, event):
        self.slider_hr.reset()
        self.viewer.set_heart_rate(75)
        self.viewer.set_noise(False)
        if self.viewer.paused:
            self.viewer.toggle_pause()
            self.btn_pause.label.set_text('|| Pause')
        self.status.set_text('Reset to defaults — HR 75 BPM, no noise')


# ============================================================
#  MODULE 4 — main.py  (Bilal — Integration & Entry Point)
# ============================================================

def main():
    """
    Entry point — wires all modules together into one running app.

    Architecture:
        get_ecg_signal()  ──►  ECGVisualizer  ──►  ECGControls
                                    │
                              FuncAnimation
                              (live plot updates)
    """
    print("=" * 55)
    print("  ECG Signal Simulator & Viewer — BIOS R&D")
    print("  Team: Hammad | Hassan | Talha | Bilal")
    print("=" * 55)
    print("  Controls:")
    print("    - Drag slider  : change heart rate (40–180 BPM)")
    print("    - Add Noise    : inject EMG/artifact noise")
    print("    - Pause/Play   : freeze the live trace")
    print("    - Export CSV   : save 10s signal to file")
    print("    - Reset        : restore defaults")
    print("=" * 55)

    # 1. Build visualizer (owns the figure)
    viewer = ECGVisualizer()

    # 2. Attach controls to the same figure
    controls = ECGControls(viewer.fig, viewer)

    # 3. Start live animation (blocks until window closed)
    viewer.start()


if __name__ == '__main__':
    main()
