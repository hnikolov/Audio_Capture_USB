"""
USB Audio Capture — Python / PortAudio
=======================================
Captures audio from USB devices like Sonicake Pocket Master, NUX Mighty Plug Pro, etc.
Uses sounddevice (PortAudio backend) which accesses WASAPI, WDM, and ASIO drivers
— much broader device support than browser-based WebAudio.

Requirements:
    pip install sounddevice numpy soundfile

Usage:
    python audio_capture_usb.py
"""

import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: 'sounddevice' is required.  Install with:  pip install sounddevice")
    sys.exit(1)

try:
    import soundfile as sf
except ImportError:
    print("ERROR: 'soundfile' is required.  Install with:  pip install soundfile")
    sys.exit(1)


# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATES = [44100, 48000, 96000]
DEFAULT_SR = 48000
BLOCK_SIZES = [64, 128, 256, 512, 1024]
DEFAULT_BLOCK_SIZE = 256
WAVEFORM_WIDTH = 700
WAVEFORM_HEIGHT = 120
METER_WIDTH = 500
METER_HEIGHT = 16

# ── Shared state (written by audio callback, read by GUI) ────────────────────
audio_buf_lock = threading.Lock()
audio_buf = np.zeros(DEFAULT_BLOCK_SIZE, dtype=np.float32)
peak_level = 0.0
rms_level = 0.0

rec_queue: queue.Queue[np.ndarray] = queue.Queue()
rec_active = False
rec_gain = 1.0


# ── Audio callback ────────────────────────────────────────────────────────────
def audio_callback(indata, frames, time_info, status):
    global audio_buf, peak_level, rms_level
    if status:
        print(f"[sounddevice] {status}")
    data = indata.copy()
    mono = data[:, 0] if data.ndim > 1 else data.ravel()
    with audio_buf_lock:
        audio_buf = mono.copy()
        peak_level = float(np.max(np.abs(mono)))
        rms_level = float(np.sqrt(np.mean(mono ** 2)))
    if rec_active:
        rec_queue.put((data * rec_gain).copy())


# ── Application ──────────────────────────────────────────────────────────────
class AudioCaptureApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("USB Audio Capture (Python / PortAudio)")
        self.root.resizable(True, True)
        self.root.minsize(780, 620)

        self.stream: sd.InputStream | None = None
        self.is_listening = False
        self.is_recording = False
        self.rec_frames: list[np.ndarray] = []
        self.rec_start_time = 0.0
        self.gain = 1.0
        self.rec_sr = DEFAULT_SR
        self.rec_channels = 1
        self.recording_counter = 0
        self._countdown_remaining = 0
        self._countdown_after_id = None

        self._build_ui()
        self._refresh_devices()
        self._tick()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        pad = dict(padx=8, pady=4)
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # Info
        info = ttk.LabelFrame(main, text="Info", padding=8)
        info.pack(fill=tk.X, **pad)
        ttk.Label(info, wraplength=700, text=(
            "This app uses PortAudio to access USB audio devices directly — "
            "including WASAPI and WDM devices that browsers cannot see. "
            "If your Sonicake Pocket Master appears in the device list below, it can be captured."
        )).pack(anchor=tk.W)

        # Host APIs
        api_frame = ttk.LabelFrame(main, text="Available Host APIs", padding=8)
        api_frame.pack(fill=tk.X, **pad)
        api_names = ", ".join(a["name"] for a in sd.query_hostapis())
        ttk.Label(api_frame, text=api_names, wraplength=700).pack(anchor=tk.W)

        # Device selection
        dev_frame = ttk.LabelFrame(main, text="1. Select Audio Input", padding=8)
        dev_frame.pack(fill=tk.X, **pad)

        row0 = ttk.Frame(dev_frame)
        row0.pack(fill=tk.X, pady=2)
        ttk.Label(row0, text="Input Device:").pack(side=tk.LEFT)
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(row0, textvariable=self.device_var,
                                         state="readonly", width=80)
        self.device_combo.pack(side=tk.LEFT, padx=(8, 4), fill=tk.X, expand=True)
        ttk.Button(row0, text="Refresh", command=self._refresh_devices).pack(side=tk.LEFT)

        row1 = ttk.Frame(dev_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="Sample Rate:").pack(side=tk.LEFT)
        self.sr_var = tk.StringVar(value=str(DEFAULT_SR))
        ttk.Combobox(row1, textvariable=self.sr_var,
                      values=[str(s) for s in SAMPLE_RATES],
                      state="readonly", width=10).pack(side=tk.LEFT, padx=8)

        row_bs = ttk.Frame(dev_frame)
        row_bs.pack(fill=tk.X, pady=2)
        ttk.Label(row_bs, text="Block Size:").pack(side=tk.LEFT)
        self.bs_var = tk.StringVar(value=str(DEFAULT_BLOCK_SIZE))
        ttk.Combobox(row_bs, textvariable=self.bs_var,
                      values=[str(b) for b in BLOCK_SIZES],
                      state="readonly", width=10).pack(side=tk.LEFT, padx=8)
        ttk.Label(row_bs, text="(smaller = more responsive meter/waveform, higher CPU)",
                  foreground="gray").pack(side=tk.LEFT)

        row_excl = ttk.Frame(dev_frame)
        row_excl.pack(fill=tk.X, pady=2)
        self.wasapi_excl_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row_excl,
                         text="WASAPI Exclusive Mode (bit-perfect capture, locks device)",
                         variable=self.wasapi_excl_var).pack(side=tk.LEFT)

        # Levels
        lvl_frame = ttk.LabelFrame(main, text="2. Levels", padding=8)
        lvl_frame.pack(fill=tk.X, **pad)

        btn_row = ttk.Frame(lvl_frame)
        btn_row.pack(fill=tk.X, pady=2)
        self.btn_listen = ttk.Button(btn_row, text="▶ Start Listening",
                                     command=self._start_listening)
        self.btn_listen.pack(side=tk.LEFT, padx=(0, 4))
        self.btn_stop = ttk.Button(btn_row, text="■ Stop",
                                   command=self._stop_listening, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=4)
        self.status_label = ttk.Label(btn_row, text="Idle", foreground="gray")
        self.status_label.pack(side=tk.LEFT, padx=12)

        meter_row = ttk.Frame(lvl_frame)
        meter_row.pack(fill=tk.X, pady=2)
        ttk.Label(meter_row, text="Peak", width=5).pack(side=tk.LEFT)
        self.meter_canvas = tk.Canvas(meter_row, width=METER_WIDTH, height=METER_HEIGHT,
                                      bg="#e0e0e0", highlightthickness=0)
        self.meter_canvas.pack(side=tk.LEFT, padx=8)
        self.meter_bar = self.meter_canvas.create_rectangle(0, 0, 0, METER_HEIGHT, fill="#28a745")
        self.db_label = ttk.Label(meter_row, text="-∞ dB", width=10, font=("Consolas", 9))
        self.db_label.pack(side=tk.LEFT)

        rms_row = ttk.Frame(lvl_frame)
        rms_row.pack(fill=tk.X, pady=2)
        ttk.Label(rms_row, text="RMS", width=5).pack(side=tk.LEFT)
        self.rms_canvas = tk.Canvas(rms_row, width=METER_WIDTH, height=METER_HEIGHT,
                                    bg="#e0e0e0", highlightthickness=0)
        self.rms_canvas.pack(side=tk.LEFT, padx=8)
        self.rms_bar = self.rms_canvas.create_rectangle(0, 0, 0, METER_HEIGHT, fill="#17a2b8")
        self.rms_db_label = ttk.Label(rms_row, text="-∞ dB", width=10, font=("Consolas", 9))
        self.rms_db_label.pack(side=tk.LEFT)

        # Crest factor = Peak / RMS (in dB).  Indicates transient content.
        # Typical crest factors by signal type:
        #   Clean electric guitar :  8–15 dB  (very dynamic, sharp pick attacks)
        #   Distorted guitar      :  3–6  dB  (compression from clipping)
        #   Bass guitar            :  8–12 dB  (pluck transients)
        #   Drums / percussion     : 15–25 dB  (extreme transients)
        #   Vocals (dynamic)       : 10–18 dB
        #   Pop / rock master      :  6–10 dB  (moderate loudness war)
        #   Metal master (loud)    :  3–6  dB  (heavily limited / brickwalled)
        #   Classical / jazz       : 15–25 dB  (wide dynamic range)
        #   Sine wave              :  3.0  dB  (theoretical minimum for continuous)
        #
        # For a guitar input signal being captured:
        #   < 6  dB → heavily compressed / clipping — reduce gain or check signal
        #   6–15 dB → normal clean-to-crunch guitar range (green)
        #   > 15 dB → very dynamic / percussive playing or quiet signal with noise floor
        crest_row = ttk.Frame(lvl_frame)
        crest_row.pack(fill=tk.X, pady=2)
        ttk.Label(crest_row, text="Crest", width=5).pack(side=tk.LEFT)
        self.crest_canvas = tk.Canvas(crest_row, width=METER_WIDTH, height=METER_HEIGHT,
                                      bg="#e0e0e0", highlightthickness=0)
        self.crest_canvas.pack(side=tk.LEFT, padx=8)
        self.crest_bar = self.crest_canvas.create_rectangle(0, 0, 0, METER_HEIGHT, fill="#17a2b8")
        self.crest_label = ttk.Label(crest_row, text="— dB", width=10, font=("Consolas", 9))
        self.crest_label.pack(side=tk.LEFT)

        self.waveform_canvas = tk.Canvas(lvl_frame, width=WAVEFORM_WIDTH,
                                         height=WAVEFORM_HEIGHT, bg="#1a1a2e",
                                         highlightthickness=0)
        self.waveform_canvas.pack(fill=tk.X, pady=4)

        gain_row = ttk.Frame(lvl_frame)
        gain_row.pack(fill=tk.X, pady=2)
        ttk.Label(gain_row, text="Gain:").pack(side=tk.LEFT)
        self.gain_label = ttk.Label(gain_row, text="1.0x", width=5, font=("Consolas", 9))
        self.gain_slider = ttk.Scale(gain_row, from_=0, to=3, orient=tk.HORIZONTAL,
                                     command=self._on_gain)
        self.gain_slider.set(1.0)
        self.gain_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.gain_label.pack(side=tk.LEFT)

        # Record
        rec_frame = ttk.LabelFrame(main, text="3. Record", padding=8)
        rec_frame.pack(fill=tk.X, **pad)

        delay_row = ttk.Frame(rec_frame)
        delay_row.pack(fill=tk.X, pady=2)
        ttk.Label(delay_row, text="Countdown Delay:").pack(side=tk.LEFT)
        self.delay_var = tk.StringVar(value="3")
        ttk.Combobox(delay_row, textvariable=self.delay_var,
                      values=["0", "1", "2", "3", "5", "10"],
                      state="readonly", width=5).pack(side=tk.LEFT, padx=8)
        ttk.Label(delay_row, text="seconds (time to get ready before recording starts)",
                  foreground="gray").pack(side=tk.LEFT)

        rec_row = ttk.Frame(rec_frame)
        rec_row.pack(fill=tk.X, pady=2)
        self.btn_rec = ttk.Button(rec_row, text="⏺ Record",
                                  command=self._start_recording, state=tk.DISABLED)
        self.btn_rec.pack(side=tk.LEFT, padx=(0, 4))
        self.btn_stop_rec = ttk.Button(rec_row, text="⏹ Stop Recording",
                                       command=self._stop_recording, state=tk.DISABLED)
        self.btn_stop_rec.pack(side=tk.LEFT, padx=4)
        self.rec_timer_label = ttk.Label(rec_row, text="", foreground="red",
                                         font=("Consolas", 11, "bold"))
        self.rec_timer_label.pack(side=tk.LEFT, padx=12)

        # Log
        log_frame = ttk.LabelFrame(main, text="Log", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True, **pad)
        self.log_text = tk.Text(log_frame, height=8, font=("Consolas", 8),
                                state=tk.DISABLED, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL,
                                  command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self._log("Ready. Connect your USB audio device and click 'Start Listening'.")

    # ── Logging ───────────────────────────────────────────────────────────────
    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{ts}] {msg}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    # ── Device enumeration ────────────────────────────────────────────────────
    def _refresh_devices(self):
        sd._terminate()
        sd._initialize()
        devices = sd.query_devices()
        self._input_devices = []
        entries = []
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                api_name = sd.query_hostapis(d["hostapi"])["name"]
                label = f"[{i}] {d['name']}  ({api_name}, {d['max_input_channels']}ch)"
                entries.append(label)
                self._input_devices.append((i, d))
        self.device_combo["values"] = entries
        if entries:
            self.device_combo.current(0)
        self._log(f"Found {len(entries)} input device(s).")
        for e in entries:
            self._log(f"  • {e}")

    def _selected_device_index(self) -> int | None:
        sel = self.device_combo.current()
        if sel < 0 or sel >= len(self._input_devices):
            return None
        return self._input_devices[sel][0]

    # ── Start / stop listening ────────────────────────────────────────────────
    def _start_listening(self):
        dev_idx = self._selected_device_index()
        if dev_idx is None:
            messagebox.showwarning("No device", "Select an audio input device first.")
            return

        sr = int(self.sr_var.get())
        blocksize = int(self.bs_var.get())
        dev_info = sd.query_devices(dev_idx)
        channels = min(dev_info["max_input_channels"], 2)

        extra_kw = {}
        if self.wasapi_excl_var.get():
            api_info = sd.query_hostapis(dev_info["hostapi"])
            if "WASAPI" in api_info["name"]:
                try:
                    extra_kw["extra_settings"] = sd.WasapiSettings(exclusive=True)
                except AttributeError:
                    self._log("WARNING: sd.WasapiSettings not available.")
            else:
                self._log("NOTE: WASAPI exclusive mode only applies to WASAPI devices. Ignored.")

        try:
            self.stream = sd.InputStream(
                device=dev_idx,
                samplerate=sr,
                channels=channels,
                blocksize=blocksize,
                dtype="float32",
                latency="low",
                callback=audio_callback,
                **extra_kw,
            )
            self.stream.start()
        except Exception as e:
            self._log(f"ERROR opening device: {e}")
            messagebox.showerror("Error", str(e))
            return

        self.rec_sr = sr
        self.rec_channels = channels
        self.is_listening = True
        self.btn_listen.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)
        self.btn_rec.configure(state=tk.NORMAL)

        actual_sr = int(self.stream.samplerate)
        latency = self.stream.latency
        lat_str = f"{latency * 1000:.1f}ms" if not isinstance(latency, tuple) \
            else f"{latency[0] * 1000:.1f}ms"
        excl = " [WASAPI exclusive]" if "extra_settings" in extra_kw else ""
        self.status_label.configure(text=f"Listening — {dev_info['name']}", foreground="blue")
        self._log(f"Listening: {dev_info['name']} @ {actual_sr} Hz, {channels}ch, "
                  f"block={blocksize}, latency={lat_str}{excl}")
        if actual_sr != sr:
            self._log(f"WARNING: device negotiated {actual_sr} Hz instead of {sr} Hz.")

    def _stop_listening(self):
        if self.is_recording:
            self._stop_recording()
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_listening = False
        self.btn_listen.configure(state=tk.NORMAL)
        self.btn_stop.configure(state=tk.DISABLED)
        self.btn_rec.configure(state=tk.DISABLED)
        self.btn_stop_rec.configure(state=tk.DISABLED)
        self.status_label.configure(text="Idle", foreground="gray")
        self._log("Stopped listening.")

    # ── Recording ─────────────────────────────────────────────────────────────
    def _start_recording(self):
        if not self.is_listening:
            return
        delay = int(self.delay_var.get())
        if delay > 0:
            self._begin_countdown(delay)
        else:
            self._begin_recording()

    def _begin_countdown(self, seconds: int):
        self._countdown_remaining = seconds
        self.btn_rec.configure(state=tk.DISABLED)
        self.btn_stop_rec.configure(state=tk.NORMAL)
        self.status_label.configure(text="Get ready…", foreground="orange")
        self._log(f"Countdown: {seconds}s before recording starts…")
        self._countdown_tick()

    def _countdown_tick(self):
        if self._countdown_remaining <= 0:
            self._countdown_after_id = None
            self._begin_recording()
            return
        self.rec_timer_label.configure(text=f"-{self._countdown_remaining}s")
        self._countdown_remaining -= 1
        self._countdown_after_id = self.root.after(1000, self._countdown_tick)

    def _begin_recording(self):
        global rec_active, rec_gain
        self.rec_frames = []
        while not rec_queue.empty():
            try:
                rec_queue.get_nowait()
            except queue.Empty:
                break
        rec_gain = self.gain
        rec_active = True
        self.is_recording = True
        self.rec_start_time = time.time()
        self.btn_rec.configure(state=tk.DISABLED)
        self.btn_stop_rec.configure(state=tk.NORMAL)
        self.status_label.configure(text="Recording…", foreground="red")
        self._log("Recording started.")

    def _stop_recording(self):
        global rec_active
        # Cancel countdown if still running
        if self._countdown_after_id is not None:
            self.root.after_cancel(self._countdown_after_id)
            self._countdown_after_id = None
            self._countdown_remaining = 0
            self.btn_rec.configure(state=tk.NORMAL)
            self.btn_stop_rec.configure(state=tk.DISABLED)
            self.rec_timer_label.configure(text="")
            self.status_label.configure(text="Listening", foreground="blue")
            self._log("Countdown cancelled.")
            return
        if not self.is_recording:
            return
        rec_active = False
        self.is_recording = False
        self.btn_rec.configure(state=tk.NORMAL)
        self.btn_stop_rec.configure(state=tk.DISABLED)
        self.rec_timer_label.configure(text="")
        self.status_label.configure(text="Listening", foreground="blue")

        while not rec_queue.empty():
            try:
                self.rec_frames.append(rec_queue.get_nowait())
            except queue.Empty:
                break

        if not self.rec_frames:
            self._log("No audio captured.")
            return

        data = np.concatenate(self.rec_frames, axis=0)
        actual_sr = int(self.stream.samplerate) if self.stream else self.rec_sr
        duration = len(data) / actual_sr
        self.recording_counter += 1

        default_name = f"sample-{self.recording_counter}.wav"
        filepath = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("FLAC files", "*.flac")],
            initialfile=default_name,
            title="Save Recording",
        )
        if filepath:
            sf.write(filepath, data, actual_sr)
            size_kb = Path(filepath).stat().st_size / 1024
            self._log(f"Saved: {filepath} — {duration:.1f}s, {size_kb:.1f} KB, sr={actual_sr}")
        else:
            self._log(f"Recording discarded ({duration:.1f}s).")

    # ── Gain ──────────────────────────────────────────────────────────────────
    def _on_gain(self, val):
        global rec_gain
        self.gain = float(val)
        rec_gain = self.gain
        if hasattr(self, "gain_label"):
            self.gain_label.configure(text=f"{self.gain:.1f}x")

    # ── GUI update loop (~30 fps) ────────────────────────────────────────────
    def _tick(self):
        if self.is_listening:
            with audio_buf_lock:
                buf = audio_buf.copy()
                pk = peak_level
                rms = rms_level

            buf = buf * self.gain

            if self.is_recording:
                elapsed = time.time() - self.rec_start_time
                mm = int(elapsed) // 60
                ss = int(elapsed) % 60
                self.rec_timer_label.configure(text=f"{mm:02d}:{ss:02d}")

            # Peak meter
            pk = pk * self.gain
            db = 20 * np.log10(pk) if pk > 0 else -np.inf
            pct = max(0.0, min(1.0, (db + 60) / 60))
            bar_w = int(pct * METER_WIDTH)
            self.meter_canvas.coords(self.meter_bar, 0, 0, bar_w, METER_HEIGHT)
            if pct < 0.6:
                color = "#28a745"
            elif pct < 0.85:
                color = "#ffc107"
            else:
                color = "#dc3545"
            self.meter_canvas.itemconfig(self.meter_bar, fill=color)
            self.db_label.configure(
                text=f"{db:.1f} dB" if np.isfinite(db) else "-∞ dB"
            )

            # RMS meter
            rms = rms * self.gain
            rms_db = 20 * np.log10(rms) if rms > 0 else -np.inf
            rms_pct = max(0.0, min(1.0, (rms_db + 60) / 60))
            rms_bar_w = int(rms_pct * METER_WIDTH)
            self.rms_canvas.coords(self.rms_bar, 0, 0, rms_bar_w, METER_HEIGHT)
            # Color by loudness zone: quiet→cyan, good→green, hot→yellow, clip→red
            if rms_db < -40:
                rms_color = "#6c757d"   # gray — too quiet
            elif rms_db < -20:
                rms_color = "#17a2b8"   # cyan — quiet but usable
            elif rms_db < -6:
                rms_color = "#28a745"   # green — ideal range
            elif rms_db < -3:
                rms_color = "#ffc107"   # yellow — hot
            else:
                rms_color = "#dc3545"   # red — clipping risk
            self.rms_canvas.itemconfig(self.rms_bar, fill=rms_color)
            self.rms_db_label.configure(
                text=f"{rms_db:.1f} dB" if np.isfinite(rms_db) else "-∞ dB"
            )

            # Crest factor (Peak-to-RMS ratio in dB)
            if np.isfinite(db) and np.isfinite(rms_db) and rms_db > -60:
                crest_db = db - rms_db
            else:
                crest_db = 0.0
            # Visualize on a 0–30 dB scale (covers most real signals)
            crest_pct = max(0.0, min(1.0, crest_db / 30.0))
            crest_bar_w = int(crest_pct * METER_WIDTH)
            self.crest_canvas.coords(self.crest_bar, 0, 0, crest_bar_w, METER_HEIGHT)
            # Color tuned for guitar input:
            #   < 3 dB  — red:    near-square-wave, hard clipping or broken signal
            #   3–6 dB  — yellow: heavily compressed / distorted guitar
            #   6–15 dB — green:  normal clean-to-crunch guitar range
            #   > 15 dB — cyan:   very dynamic / percussive attacks or noise floor
            if crest_db < 3:
                crest_color = "#dc3545"   # red — abnormally flat / clipping
            elif crest_db < 6:
                crest_color = "#ffc107"   # yellow — heavy compression
            elif crest_db < 15:
                crest_color = "#28a745"   # green — healthy guitar dynamics
            else:
                crest_color = "#17a2b8"   # cyan — very dynamic / percussive
            self.crest_canvas.itemconfig(self.crest_bar, fill=crest_color)
            self.crest_label.configure(
                text=f"{crest_db:.1f} dB" if (np.isfinite(db) and np.isfinite(rms_db)
                                               and rms_db > -60) else "— dB"
            )

            self._draw_waveform(buf)

        self.root.after(33, self._tick)

    def _draw_waveform(self, buf: np.ndarray):
        c = self.waveform_canvas
        w = c.winfo_width() or WAVEFORM_WIDTH
        h = c.winfo_height() or WAVEFORM_HEIGHT
        c.delete("wave")
        n = len(buf)
        if n == 0:
            return
        # Auto-scale: normalize to peak so quiet signals fill the canvas
        # peak = np.max(np.abs(buf))
        # if peak > 0: scale = min(1.0 / peak, 50.0)  # cap at 50x to avoid noise explosion
        # else:        scale = 1.0

        scale = 5.0
        
        step = max(1, n // w)
        points = []
        for i in range(0, n, step):
            x = (i / n) * w
            y = (0.5 - buf[i] * scale * 0.5) * h
            points.append(x)
            points.append(y)
        if len(points) >= 4:
            c.create_line(points, fill="#00d4ff", width=1.5, tags="wave", smooth=False)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    def on_close(self):
        self._stop_listening()
        self.root.destroy()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    root = tk.Tk()
    app = AudioCaptureApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
