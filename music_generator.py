#!/usr/bin/env python3
"""
ai_music_generator_with_visuals_fast.py

High-performance optimized AI Music Generator (LSTM) with UI and piano-roll visualization.
Optimizations included:
 - Smaller, efficient LSTM architecture (256 units)
 - tf.data pipeline, batching & prefetching
 - Parallel MIDI parsing + caching to disk
 - Mixed precision when GPU present
 - XLA JIT + threading tuning
 - EarlyStopping + ReduceLROnPlateau callbacks
 - Non-blocking UI updates via a queue (Tk-safe)
 - Reused model in memory and single pygame init

Drop this file alongside a `midi_songs/` folder and run.
"""

import os
import time
import threading
import random
import numpy as np
import music21 as m21
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import customtkinter as ctk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pygame
from datetime import datetime
import concurrent.futures
import queue

# ------------------------
# CONFIG
# ------------------------
MIDI_DIR = "midi_songs"
MODEL_PATH = "music_model_rnn.keras"
GENERATED_DIR = "generated_music"
CACHE_PATH = "notes_cache.npz"
SEQUENCE_LENGTH = 100
GENERATE_NOTES = 400
BATCH_SIZE = 64
EPOCHS_DEFAULT = 12
os.makedirs(MIDI_DIR, exist_ok=True)
os.makedirs(GENERATED_DIR, exist_ok=True)

# ------------------------
# TF PERFORMANCE TUNING
# ------------------------
USE_GPU = len(tf.config.list_physical_devices('GPU')) > 0
try:
    # prefer XLA where available
    tf.config.optimizer.set_jit(True)
except Exception:
    pass
# set threading to CPU core count to improve performance on CPU
try:
    cores = os.cpu_count() or 2
    tf.config.threading.set_intra_op_parallelism_threads(cores)
    tf.config.threading.set_inter_op_parallelism_threads(cores)
except Exception:
    pass
# mixed precision if GPU available
if USE_GPU:
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
    except Exception:
        pass

# ------------------------
# UI queue for thread-safe updates
# ------------------------
ui_queue = queue.Queue()

def ui_enqueue(fn, *args, **kwargs):
    ui_queue.put((fn, args, kwargs))

# ------------------------
# MIDI parsing (parallel + cache)
# ------------------------

def _parse_midi_file(path):
    tokens = []
    try:
        midi = m21.converter.parse(path)
        parts = m21.instrument.partitionByInstrument(midi)
        elements = parts.parts[0].recurse() if parts else midi.flat.notes
        for el in elements:
            if isinstance(el, m21.note.Note):
                tokens.append(str(el.pitch))
            elif isinstance(el, m21.chord.Chord):
                tokens.append('.'.join(str(n) for n in el.normalOrder))
    except Exception as e:
        print(f"Failed to parse {os.path.basename(path)}: {e}")
    return tokens


def get_notes(progress_callback=None, force_reload=False):
    files = [os.path.join(MIDI_DIR, f) for f in sorted(os.listdir(MIDI_DIR)) if f.lower().endswith((".mid", ".midi"))]
    if not force_reload and os.path.exists(CACHE_PATH):
        try:
            cache = np.load(CACHE_PATH, allow_pickle=True)
            if list(cache.get('files', [])) == files:
                tokens = cache['tokens'].tolist()
                if progress_callback:
                    progress_callback(1.0)
                return tokens
        except Exception:
            pass
    tokens = []
    total = max(1, len(files))
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, total or 1)) as ex:
        futures = {ex.submit(_parse_midi_file, f): f for f in files}
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            parsed = fut.result()
            if parsed:
                tokens.extend(parsed)
            if progress_callback:
                progress_callback(0.15 * ((i + 1) / total))
    try:
        np.savez_compressed(CACHE_PATH, files=files, tokens=np.array(tokens, dtype=object))
    except Exception:
        pass
    return tokens

# ------------------------
# Sequence preparation (returns integer sequences)
# ------------------------

def prepare_sequences(notes, seq_length=SEQUENCE_LENGTH):
    pitchnames = sorted(set(notes))
    note_to_int = {n: i for i, n in enumerate(pitchnames)}
    network_input = []
    network_output = []
    n = len(notes)
    for i in range(0, n - seq_length):
        seq_in = notes[i:i + seq_length]
        seq_out = notes[i + seq_length]
        network_input.append([note_to_int[nn] for nn in seq_in])
        network_output.append(note_to_int[seq_out])
    X = np.array(network_input, dtype=np.int32)
    y = np.array(network_output, dtype=np.int32)
    return X, y, note_to_int, pitchnames

# ------------------------
# Model: smaller, efficient LSTM
# ------------------------

def build_model(input_shape, n_vocab, units=256, use_gru=False):
    RNN = layers.GRU if use_gru else layers.LSTM
    model = models.Sequential()
    model.add(RNN(units, return_sequences=True, input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))
    model.add(RNN(units))
    model.add(layers.Dense(max(128, units // 2), activation='relu'))
    model.add(layers.Dropout(0.25))
    # ensure final layer returns float32 for stable loss with mixed precision
    model.add(layers.Dense(n_vocab, activation='softmax', dtype='float32'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-3))
    return model

# ------------------------
# Training / generation helpers
# ------------------------
_global_model = None
_model_lock = threading.Lock()


def train_model_thread(status_label, progress_bar, epochs=EPOCHS_DEFAULT, seq_length=SEQUENCE_LENGTH):
    try:
        ui_enqueue(status_label.configure, text="Loading MIDI files...")
        ui_enqueue(progress_bar.set, 0.0)
        notes = get_notes(progress_callback=lambda v: ui_enqueue(progress_bar.set, v * 0.05))
        if len(notes) < seq_length + 1:
            ui_enqueue(messagebox.showerror, "Not enough data", f"Need more notes. Found {len(notes)} notes.")
            ui_enqueue(status_label.configure, text="Idle")
            return
        ui_enqueue(status_label.configure, text="Preparing sequences...")
        X_raw, y_raw, note_to_int, pitchnames = prepare_sequences(notes, seq_length=seq_length)
        n_vocab = len(pitchnames)
        # dynamic seq_length fallback
        if X_raw.shape[0] < 1000 and seq_length > 60:
            seq_length = max(50, seq_length // 2)
            X_raw, y_raw, note_to_int, pitchnames = prepare_sequences(notes, seq_length=seq_length)
            n_vocab = len(pitchnames)
        ui_enqueue(status_label.configure, text="Building dataset...")
        ds = tf.data.Dataset.from_tensor_slices((X_raw, y_raw))
        ds = ds.shuffle(buffer_size=4096).map(lambda x, y: (tf.cast(x, tf.float32) / tf.cast(n_vocab, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(lambda x, y: (tf.expand_dims(x, -1), y), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

        global _global_model
        with _model_lock:
            if _global_model is not None:
                model = _global_model
                ui_enqueue(status_label.configure, text="Using cached model in memory...")
            elif os.path.exists(MODEL_PATH):
                ui_enqueue(status_label.configure, text="Loading existing model...")
                model = tf.keras.models.load_model(MODEL_PATH)
                _global_model = model
            else:
                ui_enqueue(status_label.configure, text="Building model...")
                model = build_model((seq_length, 1), n_vocab, units=256)
                _global_model = model

        checkpoint = ModelCheckpoint(os.path.join(GENERATED_DIR, "weights-best.keras"), monitor='loss', save_best_only=True, mode='min', verbose=0)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-6, verbose=0)
        early = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True, verbose=0)

        class UIProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                loss = logs.get('loss') if logs else 0.0
                frac = (epoch + 1) / float(epochs)
                ui_enqueue(status_label.configure, text=f"Epoch {epoch+1}/{epochs} - loss {loss:.4f}")
                ui_enqueue(progress_bar.set, 0.05 + 0.85 * frac)

        ui_enqueue(status_label.configure, text="Training...")
        model.fit(ds, epochs=epochs, callbacks=[checkpoint, reduce_lr, early, UIProgressCallback()], verbose=0)
        model.save(MODEL_PATH)
        ui_enqueue(progress_bar.set, 1.0)
        ui_enqueue(status_label.configure, text="Training complete ✅")
        ui_enqueue(messagebox.showinfo, "Training", "Model training completed.")
    except Exception as e:
        ui_enqueue(messagebox.showerror, "Training error", str(e))
        ui_enqueue(status_label.configure, text="Idle")

# ------------------------
# Generation
# ------------------------

def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    if temperature <= 0:
        return np.argmax(preds)
    preds = np.log(np.maximum(preds, 1e-20)) / temperature
    exp_preds = np.exp(preds - np.max(preds))
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)


def generate_music_thread(status_label, progress_bar, length=GENERATE_NOTES, tempo=120, seq_length=SEQUENCE_LENGTH, temperature=1.0):
    try:
        ui_enqueue(status_label.configure, text="Preparing data for generation...")
        notes = get_notes()
        if len(notes) < seq_length + 1:
            ui_enqueue(messagebox.showerror, "Not enough data", f"Need more notes. Found {len(notes)} notes.")
            ui_enqueue(status_label.configure, text="Idle")
            return
        _, _, note_to_int, pitchnames = prepare_sequences(notes, seq_length=seq_length)
        int_to_note = {i: n for i, n in enumerate(pitchnames)}
        n_vocab = len(pitchnames)
        global _global_model
        with _model_lock:
            if _global_model is None:
                if not os.path.exists(MODEL_PATH):
                    ui_enqueue(messagebox.showerror, "Model missing", "Model not found — train first.")
                    ui_enqueue(status_label.configure, text="Idle")
                    return
                ui_enqueue(status_label.configure, text="Loading model...")
                _global_model = tf.keras.models.load_model(MODEL_PATH)
            model = _global_model

        start_index = np.random.randint(0, len(notes) - seq_length - 1)
        pattern = [note_to_int[n] for n in notes[start_index:start_index + seq_length]]
        output_notes = []
        ui_enqueue(status_label.configure, text="Generating notes...")
        for i in range(length):
            if stop_requested.is_set():
                ui_enqueue(status_label.configure, text="Generation stopped.")
                return
            prediction_input = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
            prediction = model.predict(prediction_input, verbose=0)[0]
            index = sample_with_temperature(prediction, temperature=temperature)
            result = int_to_note[index]
            output_notes.append(result)
            pattern.append(index)
            pattern = pattern[1:]
            if i % 8 == 0:
                ui_enqueue(progress_bar.set, (i + 1) / length)
        ui_enqueue(progress_bar.set, 1.0)

        # convert to music21 and notes_for_visual
        offset = 0.0
        notes_for_visual = []
        midi_elements = []
        for token in output_notes:
            if '.' in token or token.isdigit():
                try:
                    nums = [int(n) for n in token.split('.')]
                    chord_notes = [m21.note.Note(n) for n in nums]
                except Exception:
                    chord_notes = [m21.note.Note(int(random.choice(range(60, 72))))]
                chord = m21.chord.Chord(chord_notes)
                chord.offset = offset
                chord.storedInstrument = m21.instrument.Piano()
                midi_elements.append(chord)
                for n in chord_notes:
                    notes_for_visual.append((n.pitch.midi, offset, 0.5))
            else:
                try:
                    n_obj = m21.note.Note(token)
                except Exception:
                    n_obj = m21.note.Note(random.choice(range(60, 72)))
                n_obj.offset = offset
                n_obj.storedInstrument = m21.instrument.Piano()
                midi_elements.append(n_obj)
                notes_for_visual.append((n_obj.pitch.midi, offset, 0.5))
            offset += 0.5

        midi_stream = m21.stream.Stream(midi_elements)
        midi_stream.insert(0, m21.tempo.MetronomeMark(number=tempo))
        filename = f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mid"
        filepath = os.path.join(GENERATED_DIR, filename)
        midi_stream.write('midi', fp=filepath)

        ui_enqueue(status_label.configure, text=f"Saved: {filename}")
        ui_enqueue(messagebox.showinfo, "Generated", f"Music saved: {filename}")
        app_state['current_midi_path'] = filepath
        app_state['notes_for_visual'] = notes_for_visual
        ui_enqueue(prepare_pianoroll_plot, notes_for_visual)
    except Exception as e:
        ui_enqueue(messagebox.showerror, "Generation error", str(e))
        ui_enqueue(status_label.configure, text="Idle")

# ------------------------
# Visualizer
# ------------------------

def prepare_pianoroll_plot(notes_for_visual):
    ax.clear()
    if not notes_for_visual:
        ax.text(0.5, 0.5, "No generated sequence yet", ha='center')
        canvas.draw()
        return
    starts = [s for (_, s, _) in notes_for_visual]
    ends = [s + d for (_, s, d) in notes_for_visual]
    min_time = min(starts) if starts else 0
    max_time = max(ends) if ends else 1
    pitches = [p for (p, _, _) in notes_for_visual]
    min_pitch, max_pitch = min(pitches) - 1, max(pitches) + 1
    for (pitch, start, dur) in notes_for_visual:
        ax.barh(pitch, dur, left=start, height=0.6, align='center')
    ax.set_xlabel("Time (beats)")
    ax.set_ylabel("MIDI Pitch")
    ax.set_xlim(min_time - 0.5, max_time + 0.5)
    ax.set_ylim(min_pitch, max_pitch)
    ax.invert_yaxis()
    ax.grid(True, linewidth=0.3, alpha=0.6)
    canvas.draw()

# ------------------------
# Playback / animation
# ------------------------
pygame_inited = False

def init_pygame_once():
    global pygame_inited
    if not pygame_inited:
        pygame.mixer.init()
        pygame_inited = True


def play_current_midi():
    path = app_state.get('current_midi_path')
    notes = app_state.get('notes_for_visual', [])
    if not path or not os.path.exists(path):
        messagebox.showerror("No MIDI", "No generated MIDI to play. Generate first.")
        return
    try:
        init_pygame_once()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        app_state['playing'] = True
        app_state['paused'] = False
        anim_thread = threading.Thread(target=animate_cursor, args=(notes,), daemon=True)
        anim_thread.start()
    except Exception as e:
        messagebox.showerror("Playback error", f"Playback failed: {e}")


def pause_playback():
    try:
        if app_state.get('playing') and not app_state.get('paused'):
            pygame.mixer.music.pause()
            app_state['paused'] = True
        elif app_state.get('playing') and app_state.get('paused'):
            pygame.mixer.music.unpause()
            app_state['paused'] = False
    except Exception as e:
        messagebox.showerror("Pause error", str(e))


def stop_playback():
    try:
        pygame.mixer.music.stop()
    except Exception:
        pass
    app_state['playing'] = False
    app_state['paused'] = False
    if 'cursor_line' in app_state:
        try:
            app_state['cursor_line'].remove()
            app_state.pop('cursor_line', None)
        except Exception:
            pass
    canvas.draw()


def animate_cursor(notes_for_visual):
    if not notes_for_visual:
        return
    tempo = tempo_var.get() if 'tempo_var' in globals() else 120
    seconds_per_beat = 60.0 / tempo
    total_duration_beats = max([s + d for (_, s, d) in notes_for_visual]) if notes_for_visual else 0
    cursor = ax.axvline(0, color='red', linewidth=1.5)
    app_state['cursor_line'] = cursor
    canvas.draw()
    last_draw = 0.0
    while app_state.get('playing'):
        if app_state.get('paused'):
            time.sleep(0.1)
            continue
        pos_ms = pygame.mixer.music.get_pos()
        elapsed = pos_ms / 1000.0 if pos_ms >= 0 else 0.0
        current_beats = elapsed / seconds_per_beat
        if current_beats > total_duration_beats:
            app_state['playing'] = False
            break
        now = time.time()
        if now - last_draw > 1.0 / 30.0:
            try:
                cursor.set_xdata([current_beats, current_beats])
                canvas.draw_idle()
            except Exception:
                pass
            last_draw = now
        time.sleep(0.01)
    try:
        cursor.remove()
        canvas.draw()
    except Exception:
        pass
    app_state['playing'] = False

# ------------------------
# App state, UI build
# ------------------------
app_state = {'current_midi_path': None, 'notes_for_visual': [], 'playing': False, 'paused': False}
stop_requested = threading.Event()

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
root = ctk.CTk()
root.title("AI Music Composer — Fast")
root.geometry("1100x720")

left_frame = ctk.CTkFrame(root, width=360)
left_frame.pack(side="left", fill="y", padx=12, pady=12)

title_label = ctk.CTkLabel(left_frame, text="AI Music Composer", font=("Roboto", 20, "bold"))
title_label.pack(pady=(8, 4))

desc = ctk.CTkLabel(left_frame, text="Optimized: cache, tf.data, mixed precision, XLA.\nTrain → Generate → Play", wraplength=320, justify="left")
desc.pack(pady=(0, 12))

status_label = ctk.CTkLabel(left_frame, text="Status: Idle")
status_label.pack(pady=6)

progress = ctk.CTkProgressBar(left_frame, width=320)
progress.set(0.0)
progress.pack(pady=6)

epoch_var = ctk.IntVar(value=EPOCHS_DEFAULT)
ctk.CTkLabel(left_frame, text="Epochs").pack(pady=(8, 0))
epoch_entry = ctk.CTkEntry(left_frame, textvariable=epoch_var, width=120)
epoch_entry.pack(pady=(2, 8))

seq_len_var = ctk.IntVar(value=SEQUENCE_LENGTH)
ctk.CTkLabel(left_frame, text="Sequence length").pack(pady=(2, 0))
seq_entry = ctk.CTkEntry(left_frame, textvariable=seq_len_var, width=120)
seq_entry.pack(pady=(2, 8))

tempo_var = ctk.IntVar(value=120)
ctk.CTkLabel(left_frame, text="Tempo (BPM)").pack(pady=(2, 0))
tempo_entry = ctk.CTkEntry(left_frame, textvariable=tempo_var, width=120)
tempo_entry.pack(pady=(2, 8))

temp_var = ctk.DoubleVar(value=1.0)
ctk.CTkLabel(left_frame, text="Sampling temperature").pack(pady=(2, 0))
temp_entry = ctk.CTkEntry(left_frame, textvariable=temp_var, width=120)
temp_entry.pack(pady=(2, 8))

train_btn = ctk.CTkButton(left_frame, text="🎼 Train / Resume", width=300)
train_btn.pack(pady=(12, 6))

gen_btn = ctk.CTkButton(left_frame, text="🎹 Generate Music", width=300)
gen_btn.pack(pady=(6, 12))

ctk.CTkLabel(left_frame, text="Playback Controls").pack(pady=(8, 4))
play_btn = ctk.CTkButton(left_frame, text="▶ Play", width=90, command=lambda: threading.Thread(target=play_current_midi, daemon=True).start())
pause_btn = ctk.CTkButton(left_frame, text="⏸ Pause/Resume", width=140, command=pause_playback)
stop_btn = ctk.CTkButton(left_frame, text="■ Stop", width=90, command=stop_playback)
play_btn.pack(pady=4)
pause_btn.pack(pady=4)
stop_btn.pack(pady=4)

stop_ops_btn = ctk.CTkButton(left_frame, text="🛑 Stop Training/Gen", fg_color="red", hover_color="#aa0000", width=300, command=lambda: stop_requested.set())
stop_ops_btn.pack(pady=(12, 6))

right_frame = ctk.CTkFrame(root)
right_frame.pack(side="right", expand=True, fill="both", padx=12, pady=12)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
fig.patch.set_facecolor('#0f0f1b')
ax.set_facecolor('#111122')
ax.tick_params(colors='white', which='both')
for spine in ax.spines.values():
    spine.set_color('white')

canvas = FigureCanvasTkAgg(fig, master=right_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill="both", expand=True, padx=8, pady=8)
ax.text(0.5, 0.5, "Generate music to see piano-roll", ha='center', va='center', color='white', fontsize=14)
canvas.draw()

log_box = ctk.CTkTextbox(right_frame, height=6)
log_box.insert("0.0", "Log:\n")
log_box.pack(fill="x", padx=8, pady=(0, 8))

def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_box.insert("end", f"[{timestamp}] {msg}\n")
    log_box.see("end")

# button wiring with logging

def train_model_thread_statused(status_label_widget, progress_widget, epochs, seq_len):
    log("Training started")
    train_model_thread(status_label_widget, progress_widget, epochs, seq_len)
    log("Training finished or stopped")

def generate_music_thread_statused(status_label_widget, progress_widget, length, tempo, seq_len, temp):
    log("Generation started")
    generate_music_thread(status_label_widget, progress_widget, length, tempo, seq_len, temp)
    log("Generation finished")

train_btn.configure(command=lambda: threading.Thread(target=train_model_thread_statused, args=(status_label, progress, epoch_var.get(), seq_len_var.get()), daemon=True).start())

gen_btn.configure(command=lambda: threading.Thread(target=generate_music_thread_statused, args=(status_label, progress, GENERATE_NOTES, tempo_var.get(), seq_len_var.get(), temp_var.get()), daemon=True).start())

# UI queue processing

def process_ui_queue():
    try:
        while True:
            fn, args, kwargs = ui_queue.get_nowait()
            try:
                fn(*args, **kwargs)
            except Exception as e:
                print("UI update failed:", e)
    except queue.Empty:
        pass
    root.after(80, process_ui_queue)

root.after(80, process_ui_queue)

# cleanup

def on_closing():
    stop_requested.set()
    try:
        pygame.mixer.music.stop()
    except Exception:
        pass
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
