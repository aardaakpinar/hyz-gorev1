"""
TEKNOFEST 2026 - Model TEST GUI
pip install nicegui ultralytics opencv-python torch torchvision
"""

from nicegui import ui, events
from ultralytics import YOLO

import cv2
import numpy as np
import base64
import tempfile
import threading
import torch
import asyncio

from pathlib import Path


# =========================================================
# DEVICE
# =========================================================

DEVICE = "0" if torch.cuda.is_available() else "cpu"
HALF = torch.cuda.is_available()

print(f"DEVICE: {DEVICE}")
print(f"FP16  : {HALF}")


# =========================================================
# STATE
# =========================================================

class AppState:

    def __init__(self):

        self.model = None
        self.model_path = "yolov8n.pt"

        self.busy = False

        self.current_image = None
        self.current_frame_b64 = None

        self.detected_objects = []

        self.conf = 0.30

        self.video_running = False
        self.video_thread = None


state = AppState()


# =========================================================
# MODELS
# =========================================================

def get_models():

    models = [
        "yolov8n.pt",
        "yolov8s.pt",
        "yolov8m.pt",
        "yolo11n.pt",
        "yolo11s.pt",
    ]

    custom_models = list(Path(".").rglob("*.pt"))

    for model in custom_models:
        models.append(str(model))

    return sorted(list(set(models)))


# =========================================================
# LOAD MODEL
# =========================================================

def load_model(model_path: str):

    try:

        state.busy = True

        model = YOLO(model_path)

        if HALF:
            model.model.half()

        state.model = model
        state.model_path = model_path

        ui.notify(
            f"Model yüklendi: {Path(model_path).name}",
            type="positive"
        )

    except Exception as e:

        ui.notify(
            f"Model yükleme hatası: {e}",
            type="negative"
        )

    finally:

        state.busy = False


# =========================================================
# DETECTIONS
# =========================================================

def extract_detections(result):

    detections = []

    names = result.names if hasattr(result, "names") else {}

    for box in result.boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        detections.append({
            "label": names.get(cls, str(cls)),
            "conf": conf,
        })

    return detections


# =========================================================
# IMAGE DETECTION
# =========================================================

def run_image_detection():

    if state.model is None:
        ui.notify("Önce model seç", type="warning")
        return

    if state.current_image is None:
        ui.notify("Önce resim yükle", type="warning")
        return

    try:

        state.busy = True

        results = state.model.predict(
            state.current_image,
            conf=state.conf,
            device=DEVICE,
            half=HALF,
            verbose=False
        )

        result = results[0]

        state.detected_objects = extract_detections(result)

        annotated = result.plot()

        _, buffer = cv2.imencode(".jpg", annotated)

        state.current_frame_b64 = base64.b64encode(
            buffer
        ).decode()

        ui.notify(
            f"{len(state.detected_objects)} nesne bulundu",
            type="positive"
        )

    except Exception as e:

        ui.notify(
            f"Tespit hatası: {e}",
            type="negative"
        )

    finally:

        state.busy = False


# =========================================================
# VIDEO
# =========================================================

def stop_video():

    state.video_running = False

    if state.video_thread:
        state.video_thread.join(timeout=1)

    state.video_thread = None


def process_video(video_path: str):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        ui.notify(
            "Video açılamadı",
            type="negative"
        )
        return

    try:

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"TOTAL FRAME: {total_frames}")
        print(f"FPS: {fps}")

        # ==========================================
        # MAX SAMPLE
        # ==========================================

        SAMPLE_COUNT = 200

        # videodaki örnek frame noktaları
        frame_indices = np.linspace(
            0,
            total_frames - 1,
            SAMPLE_COUNT,
            dtype=int
        )

        found_objects = {}

        for idx in frame_indices:

            if not state.video_running:
                break

            # direkt frame'e git
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

            success, frame = cap.read()

            if not success:
                continue

            # ======================================
            # YOLO
            # ======================================

            results = state.model.predict(
                frame,
                conf=state.conf,
                device=DEVICE,
                half=HALF,
                verbose=False
            )

            result = results[0]

            detections = extract_detections(result)

            # ======================================
            # UNIQUE OBJECTS
            # ======================================

            for item in detections:

                label = item["label"]

                if label not in found_objects:
                    found_objects[label] = item

            # ======================================
            # PREVIEW
            # ======================================

            annotated = result.plot()

            _, buffer = cv2.imencode(
                ".jpg",
                annotated
            )

            state.current_frame_b64 = base64.b64encode(
                buffer
            ).decode()

            state.detected_objects = list(
                found_objects.values()
            )

        ui.notify(
            f"""
Video tarama tamamlandı

Bulunan nesne:
{len(found_objects)}
            """,
            type="positive"
        )

    except Exception as e:

        ui.notify(
            f"Video analiz hatası: {e}",
            type="negative"
        )

    finally:

        cap.release()
        state.video_running = False


# =========================================================
# UPLOAD VIDEO
# =========================================================

async def upload_video(e: events.UploadEventArguments):

    if state.model is None:

        ui.notify("Önce model seç", type="warning")
        return

    try:

        file = e.file

        if asyncio.iscoroutinefunction(file.read):
            content = await file.read()
        else:
            content = file.read()

        temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp.write(content)
        temp.close()

        stop_video()

        state.video_running = True

        state.video_thread = threading.Thread(
            target=process_video,
            args=(temp.name,),
            daemon=True
        )

        state.video_thread.start()

        ui.notify(
            "Video analizi başladı",
            type="positive"
        )

    except Exception as e:

        ui.notify(
            f"Video yükleme hatası: {e}",
            type="negative"
        )


# =========================================================
# UPLOAD IMAGE
# =========================================================

async def upload_image(e: events.UploadEventArguments):

    try:

        stop_video()

        file = e.file

        if asyncio.iscoroutinefunction(file.read):
            content = await file.read()
        else:
            content = file.read()

        nparr = np.frombuffer(content, np.uint8)

        img = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR
        )

        state.current_image = img

        state.current_frame_b64 = base64.b64encode(
            content
        ).decode()

        state.detected_objects.clear()

        ui.notify(
            "Görüntü yüklendi",
            type="positive"
        )

    except Exception as e:

        ui.notify(
            f"Görüntü yükleme hatası: {e}",
            type="negative"
        )


# =========================================================
# UI
# =========================================================

ui.dark_mode().enable()

ui.colors(
    primary="#2563eb",
    secondary="#64748b",
    accent="#10b981",
)

# =========================================================
# AUTO LOAD
# =========================================================

load_model("yolov8n.pt")


# =========================================================
# LAYOUT
# =========================================================

with ui.row().classes("w-full h-screen"):

    # =====================================================
    # SIDEBAR
    # =====================================================

    with ui.column().classes(
        "w-96 h-full bg-[#111111] p-4 gap-4"
    ):

        ui.label(
            "MODEL TEST GUI"
        ).classes(
            "text-2xl font-bold text-white"
        )

        ui.separator()

        # MODEL

        ui.label(
            "MODEL"
        ).classes(
            "text-sm text-slate-400"
        )

        ui.select(
            get_models(),
            value="yolov8n.pt",
            on_change=lambda e: load_model(e.value)
        ).classes("w-full")

        # CONF

        ui.label(
            "CONFIDENCE"
        ).classes(
            "text-sm text-slate-400"
        )

        ui.slider(
            min=0.1,
            max=1.0,
            step=0.05,
            value=0.3
        ).bind_value(
            state,
            "conf"
        ).classes(
            "w-full"
        )

        ui.separator()

        # =================================================
        # SIDEBAR TABS
        # =================================================

        with ui.tabs().classes(
            "w-full"
        ) as tabs:

            image_tab = ui.tab(
                "RESİM",
                icon="image"
            )

            video_tab = ui.tab(
                "VİDEO",
                icon="video_library"
            )

        with ui.tab_panels(
            tabs,
            value=image_tab
        ).classes(
            "w-full bg-transparent"
        ):

            # =============================================
            # IMAGE TAB
            # =============================================

            with ui.tab_panel(image_tab):

                with ui.column().classes(
                    "w-full gap-4"
                ):

                    ui.upload(
                        on_upload=upload_image,
                        auto_upload=True
                    ).props(
                        "label=RESİM_SEÇ"
                    ).classes(
                        "w-full"
                    )

                    ui.button(
                        "ANALİZ ET",
                        icon="image_search",
                        on_click=run_image_detection
                    ).classes(
                        "w-full"
                    )

            # =============================================
            # VIDEO TAB
            # =============================================

            with ui.tab_panel(video_tab):

                with ui.column().classes(
                    "w-full gap-4"
                ):

                    ui.upload(
                        on_upload=upload_video,
                        auto_upload=True
                    ).props(
                        "label=VİDEO_SEÇ"
                    ).classes(
                        "w-full"
                    )

                    ui.button(
                        "VİDEOYU DURDUR",
                        icon="stop",
                        on_click=stop_video
                    ).props(
                        "color=negative"
                    ).classes(
                        "w-full"
                    )

        ui.separator()

        # =================================================
        # STATS
        # =================================================

        stats = ui.label().classes(
            "text-green-400 text-sm"
        )

        def update_stats():

            stats.set_text(
                f"""
MODEL:
{Path(state.model_path).name}

DEVICE:
{DEVICE}

NESNE:
{len(state.detected_objects)}
                """
            )

        ui.timer(0.5, update_stats)

    # =====================================================
    # RIGHT PANEL
    # =====================================================

    with ui.column().classes(
        "flex-1 h-full bg-black relative"
    ):

        # =================================================
        # PREVIEW
        # =================================================

        preview = ui.image().classes(
            "w-full h-full object-contain"
        )

        # =================================================
        # UPDATE FRAME
        # =================================================

        def update_frame():

            if not state.current_frame_b64:
                return

            preview.set_source(
                f"data:image/jpeg;base64,"
                f"{state.current_frame_b64}"
            )

        ui.timer(0.03, update_frame)

        # =================================================
        # LOADING
        # =================================================

        ui.spinner(
            size="lg"
        ).classes(
            "absolute bottom-4 right-4"
        ).bind_visibility_from(
            state,
            "busy"
        )


# =========================================================
# RUN
# =========================================================

ui.run(
    title="Model Test GUI",
    port=8080,
    dark=True,
    reload=False
)