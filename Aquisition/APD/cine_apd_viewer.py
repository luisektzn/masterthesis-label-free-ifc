"""Local Phantom CINE / acquisition NPZ viewer. Run with the project Python."""
from pathlib import Path
import time
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from pims import Cine
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


def load_apd(path):
    with np.load(path, allow_pickle=False) as data:
        t = np.asarray(data['time_s'], dtype=float)
        y = np.asarray(data['photodiode_signal'], dtype=float)
        sync = np.asarray(data['camera_frame_signal'], dtype=float) if 'camera_frame_signal' in data else None
        metadata = json.loads(str(data['metadata_json'].item())) if 'metadata_json' in data else {}
    if t.ndim != 1 or y.shape != t.shape or len(t) < 2:
        raise ValueError('APD: time_s und photodiode_signal muessen gleich lange 1D-Arrays sein.')
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)) or np.any(np.diff(t) <= 0):
        raise ValueError('APD: ungueltige Werte oder nicht aufsteigende Zeitachse.')
    if sync is not None and (sync.shape != t.shape or not np.all(np.isfinite(sync))):
        raise ValueError('Ungueltiges camera_frame_signal.')
    return t, y, sync, metadata


def shared_trigger_times(cine, metadata, correction_s=0.0):
    """Same clock mapping as apd_phantom_cine_event_clips.ipynb."""
    trigger = metadata.get('trigger_time_from_acquisition_start_s')
    if trigger is None:
        raise ValueError('metadata_json enthaelt keine trigger_time_from_acquisition_start_s. '
                         'Gemeinsame Triggerzuordnung nicht moeglich; bekannten manuellen Bildstart verwenden.')
    trigger = float(trigger)
    if not np.isfinite(trigger) or not np.isfinite(correction_s):
        raise ValueError('Triggerzeit und Zeitkorrektur muessen endlich sein.')
    cine_trigger = cine.trigger_time
    trigger_datetime = cine_trigger['datetime']
    trigger_fraction = float(cine_trigger['second_fraction'])
    stamps = cine.frame_time_stamps
    relative = np.fromiter(
        ((dt-trigger_datetime).total_seconds() + float(frac)-trigger_fraction
         for dt, frac in stamps), dtype=np.float64, count=len(stamps))
    if len(relative) != len(cine) or not len(relative):
        raise ValueError('CINE-Zeitstempelanzahl passt nicht zur Bildanzahl.')
    if not np.all(np.isfinite(relative)) or np.any(np.diff(relative) <= 0):
        raise ValueError('CINE-Bildzeitstempel sind ungueltig oder nicht streng aufsteigend.')
    return trigger + relative + correction_s


def require_overlap(times, apd_times, period_s):
    start = max(apd_times[0], times[0])
    stop = min(apd_times[-1], times[-1] + period_s)
    if stop <= start:
        raise ValueError('CINE und APD ueberlappen zeitlich nicht. Dateipaar und Zeitkorrektur pruefen.')
    return float(start), float(stop)


def sync_clock(t, signal, count, fps, active_low=False):
    """Conservative matching: require one measured edge per stored frame."""
    if signal is None:
        raise ValueError('Kein Kamerasynchronisationssignal vorhanden.')
    lo, hi = float(signal.min()), float(signal.max())
    if hi - lo < 1e-9:
        raise ValueError('Kamerasynchronisationssignal ist konstant.')
    for _ in range(20):
        mask = signal < (lo + hi) / 2
        if mask.all() or not mask.any():
            break
        lo, hi = float(signal[mask].mean()), float(signal[~mask].mean())
    active = signal < (lo + hi) / 2 if active_low else signal > (lo + hi) / 2
    edges = t[np.flatnonzero(~active[:-1] & active[1:]) + 1]
    if len(edges) != count:
        raise ValueError(f'{len(edges)} vollstaendige F-Sync-Flanken, aber {count} Bilder. '
                         'Zuordnung nicht eindeutig; ersten Bildzeitpunkt manuell setzen.')
    if count > 1 and not np.allclose(np.diff(edges), 1 / fps, rtol=0.05,
                                    atol=2 * np.median(np.diff(t))):
        raise ValueError('F-Sync-Abstaende passen nicht zur CINE-Bildrate.')
    return edges


class Viewer:
    def __init__(self, root):
        self.root = root
        root.title('CINE + APD synchroner Viewer')
        window_width = min(1250, root.winfo_screenwidth() - 80)
        window_height = min(900, root.winfo_screenheight() - 120)
        root.geometry(f'{window_width}x{window_height}')
        self.cine = None
        self.playing = False
        self.index = 0
        self.paths = [tk.StringVar(), tk.StringVar()]
        for row, title in enumerate(('CINE-Datei', 'APD-Datei (.npz)')):
            bar = ttk.Frame(root); bar.pack(fill='x', padx=8, pady=3)
            ttk.Label(bar, text=title, width=20).pack(side='left')
            ttk.Entry(bar, textvariable=self.paths[row]).pack(side='left', fill='x', expand=True)
            ttk.Button(bar, text='Auswaehlen', command=lambda r=row: self.choose(r)).pack(side='left')
        bar = ttk.Frame(root); bar.pack(fill='x', padx=8)
        ttk.Button(bar, text='Dateipaar laden', command=self.load).pack(side='left')
        self.low = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text='F-Sync aktiv LOW (nur F-Sync-Modus)', variable=self.low).pack(side='left')
        self.mode = tk.StringVar(value='Trigger')
        ttk.Combobox(bar, textvariable=self.mode, values=['Trigger', 'F-Sync', 'Manuell'], state='readonly', width=10).pack(side='left')
        ttk.Label(bar, text=' Erstes Bild / Korrektur [s]:').pack(side='left')
        self.offset = tk.StringVar(value='0.0')
        ttk.Entry(bar, textvariable=self.offset, width=13).pack(side='left')
        ttk.Button(bar, text='Zeitzuordnung anwenden', command=self.align).pack(side='left')
        self.status = tk.StringVar(value='Dateien aus derselben Aufnahme auswaehlen. Zeiten beziehen sich auf die NI-Aufnahme.')
        ttk.Label(root, textvariable=self.status, wraplength=1200).pack(fill='x', padx=8, pady=5)
        self.fig = Figure(figsize=(11, 7), constrained_layout=True)
        self.video_ax, self.signal_ax = self.fig.subplots(2, 1, gridspec_kw={'height_ratios': [3, 2]})
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.toolbar = NavigationToolbar2Tk(self.canvas, root, pack_toolbar=False)
        self.canvas.mpl_connect('button_press_event', self.click)
        self.slider = tk.Scale(root, from_=0, to=0, orient='horizontal', label='Bildindex (ab 0)', command=self.seek)
        self.slider.pack(fill='x', padx=8)
        bar = ttk.Frame(root); bar.pack(fill='x', padx=8, pady=6)
        for label, action in [('Zurueck', lambda: self.step(-1)), ('Play / Pause', self.toggle), ('Weiter', lambda: self.step(1))]:
            ttk.Button(bar, text=label, command=action).pack(side='left')
        ttk.Label(bar, text=' Geschwindigkeit:').pack(side='left')
        self.speed = tk.StringVar(value='0.01')
        ttk.Combobox(bar, textvariable=self.speed, values=['0.001', '0.01', '0.1', '0.25', '0.5', '1', '2'], width=8, state='readonly').pack(side='left')
        self.speed.trace_add('write', lambda *_: self.pause())
        ttk.Label(bar, text=' Signalfenster [ms]:').pack(side='left')
        self.window = tk.StringVar(value='20')
        ttk.Entry(bar, textvariable=self.window, width=8).pack(side='left')
        ttk.Button(bar, text='Ansicht aktualisieren', command=lambda: self.show(self.index)).pack(side='left')
        export_bar = ttk.Frame(root); export_bar.pack(fill='x', padx=8, pady=(0, 6))
        ttk.Label(export_bar, text='AVI-Bereich (Bildindex):').pack(side='left')
        self.export_start = tk.StringVar(value='0')
        self.export_stop = tk.StringVar(value='0')
        ttk.Entry(export_bar, textvariable=self.export_start, width=8).pack(side='left')
        ttk.Button(export_bar, text='Start = aktuelles Bild', command=self.set_export_start).pack(side='left')
        ttk.Label(export_bar, text=' bis ').pack(side='left')
        ttk.Entry(export_bar, textvariable=self.export_stop, width=8).pack(side='left')
        ttk.Button(export_bar, text='Ende = aktuelles Bild', command=self.set_export_stop).pack(side='left')
        ttk.Label(export_bar, text=' Video-FPS:').pack(side='left')
        self.export_fps = tk.StringVar(value='30')
        ttk.Entry(export_bar, textvariable=self.export_fps, width=6).pack(side='left')
        ttk.Button(export_bar, text='CINE + APD als AVI speichern', command=self.export_avi).pack(side='left', padx=(8, 0))
        self.toolbar.pack(fill='x', padx=8)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(30, self.tick)

    def choose(self, row):
        path = filedialog.askopenfilename(filetypes=[('Phantom CINE', '*.cine')] if row == 0 else [('APD NumPy', '*.npz')])
        if path:
            self.paths[row].set(path)

    def pause(self):
        self.playing = False

    def load(self):
        self.pause()
        candidate = None
        try:
            t, y, sync, metadata = load_apd(self.paths[1].get())
            candidate = Cine(str(Path(self.paths[0].get())))
            fps = float(candidate.frame_rate)
            if len(candidate) < 1 or not np.isfinite(fps) or fps <= 0:
                raise ValueError('CINE enthaelt keine Bilder oder keine gueltige Bildrate.')
            first_image = np.array(candidate[0], copy=True)
        except Exception as exc:
            if candidate is not None:
                candidate.close()
            messagebox.showerror('Laden fehlgeschlagen', str(exc)); return
        if self.cine is not None:
            self.cine.close()
        self.cine, self.t, self.y, self.sync, self.fps = candidate, t, y, sync, fps
        self.metadata = metadata
        self.header_fps = fps
        self.times = None
        self.video_ax.clear(); self.signal_ax.clear()
        self.artist = self.video_ax.imshow(first_image, cmap='gray')
        self.video_ax.set_axis_off()
        self.signal_ax.set(xlabel='NI-Zeit [s]', ylabel='APD [V]')
        self.trace, = self.signal_ax.plot([], [], lw=0.8)
        self.cursor = self.signal_ax.axvline(t[0], color='red', lw=1)
        self.signal_ax.grid(alpha=0.25)
        self.slider.configure(to=len(candidate) - 1)
        self.align()

    def align(self):
        self.pause()
        if self.cine is None:
            return
        try:
            offset = float(self.offset.get())
            if not np.isfinite(offset):
                raise ValueError('Zeitwert muss endlich sein.')
            if self.mode.get() == 'Trigger':
                base = shared_trigger_times(self.cine, self.metadata)
                note = 'Gemeinsamer Trigger + CINE-Zeitstempel (wie Event-Clip-Notebook). '
            elif self.mode.get() == 'F-Sync':
                base = sync_clock(self.t, self.sync, len(self.cine), self.header_fps, self.low.get())
                note = 'F-Sync: erste vollstaendige Flanke = erstes gespeichertes Bild; gleiche Aufnahme und Bildbereich vorausgesetzt.'
            else:
                base = np.arange(len(self.cine)) / self.header_fps
                note = 'Manuelle Zuordnung: Zeitpunkt des ersten gespeicherten Bildes auf der NI-Zeitachse.'
            self.times = base + offset
            period = float(np.median(np.diff(self.times))) if len(self.times) > 1 else 1/self.header_fps
            start, stop = require_overlap(self.times, self.t, period)
            self.fps = 1/period
            self.status.set(note + f' | {self.fps:,.3f} fps | APD-Ueberlappung {start:.6f} bis {stop:.6f} s')
            overlap_start = min(len(self.times)-1, int(np.searchsorted(self.times, start, side='left')))
            overlap_stop = min(len(self.times)-1, int(np.searchsorted(self.times, stop, side='left')) - 1)
            self.export_start.set(str(overlap_start))
            self.export_stop.set(str(max(overlap_start, overlap_stop)))
            self.show(overlap_start)
        except Exception as exc:
            self.times = None
            self.status.set(str(exc))
            self.canvas.draw_idle()
            messagebox.showwarning('Synchronisation pruefen', str(exc))

    def seek(self, value):
        if self.cine is not None and self.times is not None:
            i = int(float(value))
            if i != self.index:
                self.pause(); self.show(i)

    def step(self, delta):
        self.pause(); self.show(self.index + delta)

    def set_export_start(self):
        self.export_start.set(str(self.index))

    def set_export_stop(self):
        self.export_stop.set(str(self.index))

    def export_avi(self):
        """Save the selected synchronized plot sequence, without the Tk controls, as MJPEG AVI."""
        self.pause()
        if self.cine is None or self.times is None:
            messagebox.showwarning('Kein Export', 'Zuerst ein Dateipaar laden und synchronisieren.')
            return
        try:
            first = int(self.export_start.get())
            last = int(self.export_stop.get())
            video_fps = float(self.export_fps.get())
            if first < 0 or last >= len(self.cine) or last < first:
                raise ValueError(f'Bildbereich muss zwischen 0 und {len(self.cine)-1} liegen; Ende inklusive.')
            if not np.isfinite(video_fps) or video_fps <= 0:
                raise ValueError('Video-FPS muss positiv und endlich sein.')
        except Exception as exc:
            messagebox.showerror('Ungueltige Exporteinstellung', str(exc))
            return

        default_stem = Path(self.paths[0].get()).stem + f'_apd_bild_{first}-{last}.avi'
        path = filedialog.asksaveasfilename(
            title='Synchronisierte CINE- und APD-Sequenz speichern',
            defaultextension='.avi',
            initialfile=default_stem,
            filetypes=[('AVI-Video', '*.avi')],
        )
        if not path:
            return

        original_index = self.index
        frame_count = last - first + 1
        try:
            import imageio.v2 as imageio
            self.status.set(f'Exportiere 0/{frame_count} Bilder ...')
            self.root.update_idletasks()
            with imageio.get_writer(path, fps=video_fps, codec='mjpeg', quality=8,
                                    macro_block_size=None) as writer:
                for number, index in enumerate(range(first, last + 1), start=1):
                    self.show(index)
                    self.canvas.draw()
                    frame = np.asarray(self.canvas.buffer_rgba())[..., :3].copy()
                    pad_y, pad_x = frame.shape[0] % 2, frame.shape[1] % 2
                    if pad_y or pad_x:
                        frame = np.pad(frame, ((0, pad_y), (0, pad_x), (0, 0)), mode='edge')
                    writer.append_data(np.ascontiguousarray(frame))
                    if number == 1 or number == frame_count or number % 10 == 0:
                        self.status.set(f'Exportiere {number}/{frame_count} Bilder ...')
                        self.root.update_idletasks()
            duration = frame_count / video_fps
            self.status.set(f'AVI gespeichert: {path} | {frame_count} Bilder | {duration:.2f} s')
            messagebox.showinfo('Export abgeschlossen',
                                f'{frame_count} synchronisierte Bilder gespeichert.\n\n{path}')
        except Exception as exc:
            self.status.set(f'AVI-Export fehlgeschlagen: {exc}')
            messagebox.showerror('AVI-Export fehlgeschlagen',
                                 f'{exc}\n\nIst imageio-ffmpeg in der Notebook-Umgebung installiert?')
        finally:
            self.show(original_index)

    def click(self, event):
        if self.cine is not None and self.times is not None and event.inaxes == self.signal_ax and event.xdata is not None and not self.canvas.toolbar.mode:
            self.pause()
            right = min(len(self.times)-1, int(np.searchsorted(self.times, event.xdata)))
            left = max(0, right-1)
            self.show(left if abs(event.xdata-self.times[left]) <= abs(event.xdata-self.times[right]) else right)

    def show(self, index):
        if self.cine is None or self.times is None:
            return
        try:
            width = float(self.window.get()) / 1000
            if not np.isfinite(width) or width <= 0:
                raise ValueError('Signalfenster muss positiv sein.')
            self.index = int(np.clip(index, 0, len(self.cine) - 1))
            now = self.times[self.index]
            self.artist.set_data(np.array(self.cine[self.index], copy=True))
            left, right = now - width / 2, now + width / 2
            a, b = np.searchsorted(self.t, [left, right])
            # Keep extrema when reducing large windows; narrow peaks remain visible.
            x, y = self.t[max(0, a-1):min(len(self.t), b+1)], self.y[max(0, a-1):min(len(self.y), b+1)]
            if len(x) > 6000:
                groups = np.array_split(np.arange(len(x)), 1500)
                ix = np.sort(np.concatenate([[g[np.argmin(y[g])], g[np.argmax(y[g])]] for g in groups]))
                x, y = x[ix], y[ix]
            self.trace.set_data(x, y)
            self.cursor.set_xdata([now, now])
            self.signal_ax.relim(); self.signal_ax.autoscale_view(scalex=False)
            self.signal_ax.set_xlim(left, right)
            outside = ' | ausserhalb APD-Aufnahme' if now < self.t[0] or now > self.t[-1] else ''
            self.video_ax.set_title(f'Bild {self.index + 1}/{len(self.cine)} | NI-Zeit {now:.6f} s{outside}')
            self.slider.set(self.index)
            self.canvas.draw_idle()
        except Exception as exc:
            self.pause(); messagebox.showerror('Anzeige fehlgeschlagen', str(exc))

    def toggle(self):
        if self.cine is None or self.times is None:
            return
        if self.playing:
            self.pause(); return
        if self.index == len(self.cine) - 1:
            self.show(0)
        self.wall_start, self.data_start = time.perf_counter(), self.times[self.index]
        self.playing = True

    def tick(self):
        if self.playing:
            target = self.data_start + (time.perf_counter() - self.wall_start) * float(self.speed.get())
            index = min(len(self.times) - 1, int(np.searchsorted(self.times, target, side='right') - 1))
            if index != self.index:
                self.show(index)
            if target >= self.times[-1]:
                self.pause()
        self.root.after(30, self.tick)

    def close(self):
        self.pause()
        if self.cine is not None:
            self.cine.close()
        self.root.destroy()


if __name__ == '__main__':
    app = tk.Tk()
    Viewer(app)
    app.mainloop()
