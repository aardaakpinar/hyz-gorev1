from nicegui import ui, events
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from pathlib import Path
import os
import threading
import time
import asyncio
import inspect
import tempfile
from collections import deque

# --- Configuration ---
ROOT_DIR = Path(__file__).parent.resolve()
PARENT_DIR = ROOT_DIR.parent.resolve()

# --- State ---
class AppState:
    def __init__(self):
        self._model = None
        self.model_path = ""
        self.current_image_bytes = None
        self.annotated_image_b64 = None
        self.busy = False
        self.results = None
        self.detected_objects = []
        self.training = False
        self.training_log = deque(maxlen=200)
        self.last_train_run_dir = ""
        self.eval_busy = False
        self.eval_metrics = None
        self.eval_error = ""
        
        # Video-related state
        self.video_source = None  # 'file' or None
        self.video_path = ""
        self.is_processing_video = False
        self.cap = None
        self.video_thread = None
        self.frame_count = 0
        self.fps = 0
        self.detection_history = deque(maxlen=30)  # Last 30 frames
        self.conf_threshold = 0.3
        # Processed frames per second (target sampling rate)
        self.target_fps = 1.0  # 60 frames per minute

    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, value):
        self._model = value
        print(f"DEBUG: Model set to {getattr(value, 'model_name', 'Loaded')}")

state = AppState()

# --- Helpers ---
def extract_detections(result) -> list[dict[str, float | int | str]]:
    detections = []

    if not result or result.boxes is None:
        return detections

    names = getattr(result, 'names', {}) or {}

    for index, box in enumerate(result.boxes, 1):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = names.get(cls, f"class_{cls}") if isinstance(names, dict) else f"class_{cls}"
        detections.append({
            'index': index,
            'class_id': cls,
            'label': label,
            'confidence': conf,
        })

    return detections

def get_available_models():
    models = list(ROOT_DIR.glob("*.pt"))
    models += list(PARENT_DIR.glob("*.pt"))
    models += list(ROOT_DIR.glob("runs/detect/*/weights/*.pt"))
    
    standard_models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt', 'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt']
    
    all_models = sorted(list(set([str(m) for m in models] + standard_models)))
    print(f"DEBUG: Found models: {all_models}")
    return all_models

def load_model(path: str):
    if not path: return
    state.busy = True
    print(f"DEBUG: Loading model from {path}")
    try:
        state.model = YOLO(path)
        state.model_path = path
        ui.notify(f"Model yüklendi: {Path(path).name}", type='positive', position='top')
    except Exception as e:
        print(f"DEBUG: Model load error: {e}")
        ui.notify(f"Model yükleme hatası: {e}", type='negative', position='top')
    finally:
        state.busy = False

def append_training_log(message: str):
    timestamp = time.strftime("%H:%M:%S")
    state.training_log.appendleft(f"[{timestamp}] {message}")

def train_model_ui(weights_path: str, data_path: str, epochs: int, imgsz: int, device: str, batch: int, run_name: str):
    if state.training:
        return

    def _worker():
        state.training = True
        state.last_train_run_dir = ""
        state.training_log.clear()
        append_training_log("Training started.")
        append_training_log(f"Weights: {weights_path}")
        append_training_log(f"Data: {data_path}")
        append_training_log(f"Epochs: {epochs}, ImgSz: {imgsz}, Device: {device}, Batch: {batch}")
        try:
            model = YOLO(weights_path)
            results = model.train(
                data=data_path,
                epochs=epochs,
                imgsz=imgsz,
                device=device,
                batch=batch,
                patience=20,
                save=True,
                name=run_name,
                lr0=0.001,
                momentum=0.937,
                weight_decay=0.0005,
                augment=True,
                mosaic=1.0,
                flipud=0.5,
                fliplr=0.5,
                degrees=15,
            )
            save_dir = getattr(results, "save_dir", None)
            if save_dir:
                state.last_train_run_dir = str(save_dir)
                append_training_log(f"Training done. Outputs: {state.last_train_run_dir}")
            else:
                append_training_log("Training done.")
        except Exception as ex:
            append_training_log(f"Training error: {ex}")
        finally:
            state.training = False

    threading.Thread(target=_worker, daemon=True).start()

def evaluate_model_ui(model_path: str, data_path: str):
    if state.eval_busy:
        return

    def _worker():
        state.eval_busy = True
        state.eval_metrics = None
        state.eval_error = ""
        try:
            model = YOLO(model_path)
            metrics = model.val(data=data_path)
            state.eval_metrics = {
                "map50": float(metrics.box.map50),
                "map": float(metrics.box.map),
                "precision": float(metrics.box.p.mean()),
                "recall": float(metrics.box.r.mean()),
            }
        except Exception as ex:
            state.eval_error = str(ex)
        finally:
            state.eval_busy = False

    threading.Thread(target=_worker, daemon=True).start()

def run_detection():
    if not state.model:
        ui.notify("Lütfen önce bir model seçin!", type='warning', position='top')
        return
    if not state.current_image_bytes:
        ui.notify("Lütfen bir görüntü yükleyin!", type='warning', position='top')
        return

    state.busy = True
    
    try:
        nparr = np.frombuffer(state.current_image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        results = state.model.predict(img, conf=state.conf_threshold)
        state.results = results[0]
        state.detected_objects = extract_detections(state.results)
        
        annotated_frame = results[0].plot()
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        state.annotated_image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Timer görüntüyü güncelleyecek
        
        ui.notify(f"Tespit tamamlandı: {len(state.detected_objects)} nesne bulundu", type='positive', position='top')
    except Exception as e:
        state.detected_objects = []
        ui.notify(f"Tespit hatası: {e}", type='negative', position='top')
    finally:
        state.busy = False

def stop_video():
    """Video işlemeyi durdur"""
    state.is_processing_video = False
    if state.cap:
        state.cap.release()
        state.cap = None
    if state.video_thread:
        state.video_thread.join(timeout=2)
        state.video_thread = None
    state.video_source = None
    print("DEBUG: Video stopped")

def process_video_frame(frame):
    """Tek bir frame'i işle"""
    if not state.model or not frame is not None:
        return None
    
    try:
        results = state.model.predict(frame, conf=state.conf_threshold, verbose=False)
        annotated = results[0].plot()
        detections = extract_detections(results[0])
        state.detection_history.append({
            'frame': state.frame_count,
            'detections': detections,
            'count': len(detections)
        })
        return annotated, detections
    except Exception as e:
        print(f"DEBUG: Frame processing error: {e}")
        return frame, []

def video_thread_file(video_path):
    """Video dosyasını işle"""
    state.cap = cv2.VideoCapture(video_path)
    if not state.cap.isOpened():
        ui.notify(f"Video açılamadı: {video_path}", type='negative', position='top')
        state.is_processing_video = False
        return
    
    total_frames = int(state.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = 500
    original_fps = state.cap.get(cv2.CAP_PROP_FPS)
    
    frame_times = deque(maxlen=30)
    last_capture_time = 0.0

    # Build evenly-spaced sample indices across the video (start/middle/end)
    if total_frames > 0 and max_frames > 1:
        if total_frames <= max_frames:
            sample_indices = list(range(total_frames))
        else:
            step = (total_frames - 1) / (max_frames - 1)
            sample_indices = [int(round(i * step)) for i in range(max_frames)]
    else:
        sample_indices = []

    if sample_indices:
        effective_total = len(sample_indices)
        for target_index in sample_indices:
            if not state.is_processing_video or not state.cap.isOpened():
                break

            start_time = time.time()
            # Throttle processing to target FPS
            if state.target_fps > 0:
                min_interval = 1.0 / state.target_fps
                if (start_time - last_capture_time) < min_interval:
                    time.sleep(min_interval - (start_time - last_capture_time))
            last_capture_time = time.time()

            state.cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
            ret, frame = state.cap.read()
            if not ret:
                continue

            state.frame_count += 1

            annotated, detections = process_video_frame(frame)
            if annotated is None:
                continue

            progress = (state.frame_count / effective_total) * 100 if effective_total > 0 else 0

            cv2.putText(annotated, f"Frame: {state.frame_count}/{effective_total}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"FPS: {state.fps:.1f}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"Nesneler: {len(detections)}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"İlerleme: {progress:.1f}%", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            _, buffer = cv2.imencode('.jpg', annotated)
            state.annotated_image_b64 = base64.b64encode(buffer).decode('utf-8')
            state.detected_objects = detections

            frame_times.append(time.time() - start_time)
            if frame_times:
                state.fps = 1.0 / (sum(frame_times) / len(frame_times))

            # Orijinal FPS hızında oynat
            delay = (1.0 / original_fps) if original_fps > 0 else 0.033
            time.sleep(max(0, delay - (time.time() - start_time)))
    else:
        # Fallback: sequential processing when total frame count is unknown
        while state.is_processing_video and state.cap.isOpened():
            start_time = time.time()
            # Throttle processing to target FPS
            if state.target_fps > 0:
                min_interval = 1.0 / state.target_fps
                if (start_time - last_capture_time) < min_interval:
                    time.sleep(min_interval - (start_time - last_capture_time))
                    continue
            last_capture_time = start_time
            ret, frame = state.cap.read()
            
            if not ret:
                break
            if state.frame_count >= max_frames:
                break
            
            state.frame_count += 1
            
            annotated, detections = process_video_frame(frame)
            if annotated is None:
                continue
            
            progress = (state.frame_count / max_frames) * 100 if max_frames > 0 else 0
            
            cv2.putText(annotated, f"Frame: {state.frame_count}/{max_frames}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"FPS: {state.fps:.1f}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"Nesneler: {len(detections)}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"İlerleme: {progress:.1f}%", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            _, buffer = cv2.imencode('.jpg', annotated)
            state.annotated_image_b64 = base64.b64encode(buffer).decode('utf-8')
            state.detected_objects = detections
            
            frame_times.append(time.time() - start_time)
            if frame_times:
                state.fps = 1.0 / (sum(frame_times) / len(frame_times))
            
            # Orijinal FPS hızında oynat
            delay = (1.0 / original_fps) if original_fps > 0 else 0.033
            time.sleep(max(0, delay - (time.time() - start_time)))

async def handle_video_upload(e: events.UploadEventArguments):
    """Video dosyası yükleme"""
    if not state.model:
        ui.notify("Lütfen önce bir model seçin!", type='warning', position='top')
        return
    
    try:
        file_obj = getattr(e, 'file', None)
        if not file_obj:
            raise AttributeError("Video yükleme hatası")
        
        name = getattr(file_obj, 'name', 'video.mp4')
        
        # Geçici dosya olarak kaydet (Windows/Linux uyumlu)
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, name)
        read_attr = getattr(file_obj, 'read')
        
        if asyncio.iscoroutinefunction(read_attr) or inspect.iscoroutine(read_attr):
            video_bytes = await read_attr()
        else:
            video_bytes = read_attr()
        
        with open(temp_path, 'wb') as f:
            f.write(video_bytes)
        
        print(f"DEBUG: Video saved to {temp_path}")
        
        # Video işlemeyi başlat
        stop_video()
        state.is_processing_video = True
        state.video_source = 'file'
        state.video_path = temp_path
        state.frame_count = 0
        state.fps = 0
        state.detection_history.clear()
        
        state.video_thread = threading.Thread(target=video_thread_file, args=(temp_path,), daemon=True)
        state.video_thread.start()
        
        ui.notify(f"Video analizi başladı: {name}", type='positive', position='top')
    except Exception as ex:
        print(f"DEBUG: Video upload error: {ex}")
        ui.notify(f"Video yükleme hatası: {ex}", type='negative', position='top')

# --- UI Layout ---

# Global image display element
img_display = None

def render_nav():
    with ui.row().classes('w-full items-center gap-3 mb-4'):
        ui.button('GORUNTU', on_click=lambda: ui.navigate.to('/')).props('outline')
        ui.button('EGITIM', on_click=lambda: ui.navigate.to('/train')).props('outline')

@ui.page('/')
def index():
    global img_display
    
    ui.colors(primary='#3b82f6', secondary='#64748b', accent='#10b981', dark='#0f172a')
    ui.dark_mode().enable()

    card_style = 'bg-[#1b1b1b] border border-slate-700 shadow-xl rounded-xl'
    def page_timer(interval, callback):
        def _safe():
            try:
                callback()
            except RuntimeError as ex:
                if 'parent slot' in str(ex):
                    try:
                        timer.cancel()
                    except Exception:
                        pass
                else:
                    raise
        timer = ui.timer(interval, _safe)
        return timer

    with ui.column().classes('w-full h-full p-6 gap-6'):
        render_nav()

        with ui.row().classes('w-full flex-grow gap-6 no-wrap overflow-auto'):
            
            # --- LEFT SIDEBAR ---
            with ui.column().classes('w-96 flex-none gap-6'):
                
                # Model Card
                with ui.card().classes(card_style + ' w-full p-5'):
                    ui.label('MODEL YÖNETİMİ').classes('text-xs font-bold text-slate-500 mb-2 uppercase tracking-widest')
                    models = get_available_models()
                    ui.label('Aktif Model').classes('text-xs text-slate-400 mb-1')
                    m_select = ui.select(models, on_change=lambda e: load_model(e.value)).classes('w-full')
                    if models and not state.model_path:
                        m_select.value = models[0]
                        load_model(models[0])
                    
                    with ui.row().classes('mt-4 items-center gap-2 text-slate-400'):
                        ui.icon('info', size='16px')
                        ui.label('YOLOv8/v11 desteklenir.').classes('text-[11px] font-medium')

                # --- Input Source Selection ---
                with ui.card().classes(card_style + ' w-full p-5'):
                    ui.label('KAYNAK SEÇİMİ').classes('text-xs font-bold text-slate-500 mb-4 uppercase tracking-widest')
                    
                    with ui.tabs().classes('w-full') as tabs:
                        image_tab = ui.tab('Görüntü', icon='image')
                        video_tab = ui.tab('Video', icon='video_library')
                    
                    with ui.tab_panels(tabs, value=image_tab).classes('w-full'):
                        # IMAGE TAB PANEL
                        with ui.tab_panel(image_tab):
                            upload_label = ui.label('Henüz görüntü seçilmedi').classes('text-[10px] text-slate-500 mb-2 italic')
                            
                            async def handle_image_upload(e: events.UploadEventArguments):
                                try:
                                    stop_video()
                                    state.video_source = None
                                    
                                    file_obj = getattr(e, 'file', None)
                                    if file_obj:
                                        read_attr = getattr(file_obj, 'read')
                                        if asyncio.iscoroutinefunction(read_attr) or inspect.iscoroutine(read_attr):
                                            state.current_image_bytes = await read_attr()
                                        else:
                                            state.current_image_bytes = read_attr()
                                        name = getattr(file_obj, 'name', 'unknown')
                                    else:
                                        raise AttributeError('Upload event error')
                                    
                                    b64 = base64.b64encode(state.current_image_bytes).decode('utf-8')
                                    state.annotated_image_b64 = b64  # State'e kaydet, timer güncelleyecek
                                    upload_label.set_text(f"Yüklendi: {name}")
                                    state.results = None
                                    state.detected_objects = []
                                    ui.notify(f"Görüntü yüklendi: {name}", position='top', type='info')
                                except Exception as ex:
                                    ui.notify(f"Yükleme hatası: {ex}", type='negative')
                            
                            ui.upload(on_upload=handle_image_upload, auto_upload=True).classes('w-full border-dashed border-2 border-slate-700 bg-slate-800/20 rounded-lg overflow-hidden').props('flat color=primary icon=cloud_upload label=GÖRÜNTÜ_SEÇ')
                            ui.button('ANALİZ BAŞLAT', icon='psychology', on_click=run_detection).classes('w-full mt-4 py-3 rounded-xl shadow-lg shadow-primary/20').props('color=primary unelevated')
                        
                        # VIDEO TAB PANEL
                        with ui.tab_panel(video_tab):
                            ui.upload(on_upload=handle_video_upload, auto_upload=True).classes('w-full border-dashed border-2 border-slate-700 bg-slate-800/20 rounded-lg overflow-hidden mb-3').props('flat color=primary icon=video_library label=VİDEO_SEÇ')
                            
                            ui.button('VİDEOYU DURDUR', icon='stop', on_click=stop_video).classes('w-full py-3 rounded-xl').props('color=negative unelevated')
                            
                            ui.separator().classes('my-3 opacity-30')
                            
                            ui.label('AYARLAR').classes('text-xs font-bold text-slate-500 mb-2 uppercase tracking-widest')
                            ui.label('Güven Eşiği').classes('text-xs text-slate-400 mb-1')
                            conf_slider = ui.slider(min=0, max=1, step=0.05, value=0.3).classes('w-full')
                            conf_slider.bind_value(state, 'conf_threshold')
                
                # Statistics Card
                with ui.card().classes(card_style + ' w-full p-5'):
                    ui.label('İSTATİSTİKLER').classes('text-xs font-bold text-slate-500 mb-4 uppercase tracking-widest')
                    
                    with ui.row().classes('w-full justify-between items-center p-3 bg-slate-800/40 rounded-lg mb-2'):
                        ui.label('Tespit Edilen:').classes('text-slate-300 text-sm')
                        stats_label = ui.label('0').classes('text-primary font-black text-xl')
                    
                    with ui.row().classes('w-full justify-between items-center p-3 bg-slate-800/40 rounded-lg mb-2'):
                        ui.label('FPS:').classes('text-slate-300 text-sm')
                        fps_label = ui.label('0.0').classes('text-accent font-black text-xl')
                    
                    with ui.row().classes('w-full justify-between items-center p-3 bg-slate-800/40 rounded-lg'):
                        ui.label('Çerçeve:').classes('text-slate-300 text-sm')
                        frame_label = ui.label('0').classes('text-secondary font-black text-xl')
                    
                    def refresh_stats():
                        stats_label.set_text(str(len(state.detected_objects)))
                        fps_label.set_text(f"{state.fps:.1f}")
                        frame_label.set_text(str(state.frame_count))
                    
                    page_timer(0.5, refresh_stats)

            # --- MAIN DISPLAY AREA ---
            with ui.column().classes('flex-grow gap-6'):
                
                # Viewport
                with ui.card().classes(card_style + ' w-full flex-grow p-0 overflow-hidden relative flex items-center justify-center bg-black/40'):
                    ui.label('VİZYON ÇIKTISI').classes('absolute top-4 left-4 z-10 text-[10px] font-bold text-white bg-primary/80 px-3 py-1 rounded-full uppercase tracking-tighter shadow-lg')
                    img_display = ui.image().classes('max-w-full max-h-full object-contain')
                    
                    with ui.column().classes('absolute inset-0 backdrop-blur-sm items-center justify-center z-20').bind_visibility_from(state, 'busy'):
                        ui.spinner('gears', size='64px', color='primary')
                        ui.label('ANALİZ EDİLİYOR...').classes('mt-4 text-primary font-bold tracking-widest')
                    
                    # Timer ile görüntüyü güncelle
                    def update_display():
                        if img_display and state.annotated_image_b64:
                            img_display.set_source(f"data:image/jpeg;base64,{state.annotated_image_b64}")
                    
                    page_timer(0.1, update_display)

                # Details
                with ui.card().classes(card_style + ' w-full h-64 p-5 overflow-hidden'):
                    ui.label('NESNE DETAYLARI').classes('text-xs font-bold text-slate-500 mb-4 uppercase tracking-widest')
                    
                    with ui.scroll_area().classes('w-full h-full pr-4'):
                        log_container = ui.column().classes('w-full gap-3')
                        
                        def refresh_details():
                            log_container.clear()

                            if state.detected_objects:
                                with log_container:
                                    detected_names = ', '.join(item['label'] for item in state.detected_objects)
                                    with ui.card().classes('w-full p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-lg'):
                                        ui.label('Tespit Edilen Nesneler').classes('text-[11px] font-bold uppercase tracking-widest text-emerald-300')
                                        ui.label(detected_names).classes('text-sm text-slate-100 font-medium break-words')

                                    for item in state.detected_objects:
                                        label = item['label']
                                        conf = item['confidence']
                                        index = item['index']
                                        class_id = item['class_id']
                                        with ui.row().classes('w-full items-center gap-3 p-3 bg-slate-800/50 rounded-lg border border-slate-700/50 hover:bg-slate-800 transition-all'):
                                            ui.label(f"{index}.").classes('text-slate-500 font-bold min-w-[1.5rem]')
                                            with ui.column().classes('gap-0'):
                                                ui.label(label).classes('font-bold text-slate-200 text-sm')
                                                ui.label(f"Sınıf ID: {class_id}").classes('text-[11px] text-slate-500')
                                            ui.label(f"%{conf*100:.1f}").classes('ml-auto bg-primary/20 text-primary px-3 py-1 rounded-full text-xs font-bold')
                            elif state.results:
                                with log_container:
                                    ui.label('Nesne bulunamadı...').classes('text-slate-500 italic text-sm text-center w-full mt-4')
                        
                        page_timer(0.5, refresh_details)

# --- Training Page ---
@ui.page('/train')
def train_page():
    ui.colors(primary='#3b82f6', secondary='#64748b', accent='#10b981', dark='#0f172a')
    ui.dark_mode().enable()

    card_style = 'bg-[#1b1b1b] border border-slate-700 shadow-xl rounded-xl'
    def page_timer(interval, callback):
        def _safe():
            try:
                callback()
            except RuntimeError as ex:
                if 'parent slot' in str(ex):
                    try:
                        timer.cancel()
                    except Exception:
                        pass
                else:
                    raise
        timer = ui.timer(interval, _safe)
        return timer

    with ui.column().classes('w-full p-6 gap-6'):
        render_nav()

        with ui.card().classes(card_style + ' w-full p-5'):
            ui.label('MODEL EGITIMI').classes('text-xs font-bold text-slate-500 mb-4 uppercase tracking-widest')

            models = get_available_models()
            default_data = str((ROOT_DIR / 'data.yaml').resolve())

            weights_select = ui.select(models, value=models[0] if models else '', label='Weights').classes('w-full')
            data_input = ui.input(label='data.yaml yolu', value=default_data).classes('w-full')
            epochs_input = ui.number(label='Epochs', value=100, min=1, step=1).classes('w-full')
            imgsz_input = ui.number(label='Image Size', value=640, min=64, step=32).classes('w-full')
            device_input = ui.input(label='Device', value='0').classes('w-full')
            batch_input = ui.number(label='Batch', value=16, min=1, step=1).classes('w-full')
            name_input = ui.input(label='Run Name', value='custom_drone_v1').classes('w-full')

            def start_training():
                if not weights_select.value:
                    ui.notify('Weights secin', type='warning', position='top')
                    return
                train_model_ui(
                    weights_select.value,
                    data_input.value.strip(),
                    int(epochs_input.value),
                    int(imgsz_input.value),
                    device_input.value.strip(),
                    int(batch_input.value),
                    name_input.value.strip() or 'custom_drone_v1',
                )
                ui.notify('Egitim basladi', type='positive', position='top')

            ui.button('EGITIMI BASLAT', on_click=start_training).classes('w-full mt-3 py-3 rounded-xl shadow-lg shadow-primary/20').props('color=primary unelevated')

        with ui.row().classes('w-full gap-6'):
            with ui.card().classes(card_style + ' w-full p-5'):
                ui.label('DURUM').classes('text-xs font-bold text-slate-500 mb-4 uppercase tracking-widest')
                status_label = ui.label('Hazir').classes('text-slate-200 text-sm')
                run_dir_label = ui.label('').classes('text-slate-400 text-xs mt-2')

                def refresh_training_status():
                    status_label.set_text('Egitim suruyor' if state.training else 'Hazir')
                    if state.last_train_run_dir:
                        run_dir_label.set_text(f"Son cikti: {state.last_train_run_dir}")
                page_timer(0.5, refresh_training_status)

            with ui.card().classes(card_style + ' w-full p-5'):
                ui.label('LOG').classes('text-xs font-bold text-slate-500 mb-4 uppercase tracking-widest')
                with ui.scroll_area().classes('w-full h-64 pr-2'):
                    log_container = ui.column().classes('w-full gap-1')

                    def refresh_training_log():
                        log_container.clear()
                        for line in list(state.training_log):
                            ui.label(line).classes('text-[11px] text-slate-400')
                    page_timer(0.5, refresh_training_log)

# --- Entry Point ---
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title='TKNFST ModernUI',
        port=8080,
        dark=True,
        reload=True,
        favicon='🚀'
    )