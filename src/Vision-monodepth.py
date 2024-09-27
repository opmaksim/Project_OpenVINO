# 필요한 라이브러리 import
import requests
import time
from pathlib import Path
import cv2
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov
from notebook_utils import device_widget, download_file, load_image
import tkinter as tk
import threading
from typing import Optional
import openvino.properties as props

# 모델 다운로드 경로 및 설정
model_folder = Path("model")
ir_model_url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/depth-estimation-midas/FP32/"
ir_model_name_xml = "MiDaS_small.xml"
ir_model_name_bin = "MiDaS_small.bin"

# 모델 다운로드
download_file(ir_model_url + ir_model_name_xml, filename=ir_model_name_xml, directory=model_folder)
download_file(ir_model_url + ir_model_name_bin, filename=ir_model_name_bin, directory=model_folder)

# 모델 경로 설정
model_xml_path = model_folder / ir_model_name_xml

# 유틸리티 함수들 정의
def normalize_minmax(data):
    """Normalizes the values in `data` between 0 and 1"""
    return (data - data.min()) / (data.max() - data.min())

def convert_result_to_image(result, colormap="viridis"):
    """Convert network result of floating point numbers to an RGB image."""
    cmap = matplotlib.cm.get_cmap(colormap)
    result = result.squeeze(0)
    result = normalize_minmax(result)
    result = cmap(result)[:, :, :3] * 255
    result = result.astype(np.uint8)
    return result

def to_rgb(image_data) -> np.ndarray:
    """Convert image_data from BGR to RGB"""
    return cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

def detect_closer_object(depth_map: np.ndarray, threshold: float = 0.2) -> bool:
    """Detect if any object in the scene is closer than the threshold."""
    if np.min(depth_map) < threshold:
        return True
    return False

def popup_warning(message: str, duration: int = 5) -> None:
    """Show a popup window for a warning."""
    root = tk.Tk()
    root.title("Warning")
    label = tk.Label(root, text=message, padx=20, pady=20)
    label.pack()
    root.after(duration * 1000, root.destroy)
    root.mainloop()

def show_warning() -> None:
    """Show a warning in a separate thread."""
    warning_thread = threading.Thread(target=popup_warning, args=("Object too close!", 3))
    warning_thread.start()

# 장치 설정 및 모델 컴파일
device = device_widget()
cache_folder = Path("cache")
cache_folder.mkdir(exist_ok=True)

core = ov.Core()
core.set_property({props.cache_dir(): cache_folder})
model = core.read_model(model_xml_path)
compiled_model = core.compile_model(model=model, device_name=device.value)

input_key = compiled_model.input(0)
output_key = compiled_model.output(0)

network_input_shape = list(input_key.shape)
network_image_height, network_image_width = network_input_shape[2:]

# 스트림 처리 함수
def process_stream_with_warning(
    use_webcam: bool = False,
    video_file: Optional[str] = None,
    threshold: float = 0.2,
    scale_output: float = 0.5,
    advance_frames: int = 2,
    debounce_time: float = 2,
    smoothing_window: int = 5,
) -> None:
    """Process video stream and show warning if object is too close."""
    if use_webcam:
        cap = cv2.VideoCapture(0)  # Open the default webcam
        if not cap.isOpened():
            raise ValueError("Could not open webcam.")
    else:
        if video_file is None:
            raise ValueError("No video file provided.")
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise ValueError(f"The video at {video_file} cannot be read.")

    input_fps = cap.get(cv2.CAP_PROP_FPS) if not use_webcam else 30
    input_video_frame_height, input_video_frame_width = (
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )

    target_fps = input_fps / advance_frames
    target_frame_height = int(input_video_frame_height * scale_output)
    target_frame_width = int(input_video_frame_width * scale_output)

    cv2.namedWindow("Monodepth Estimation", cv2.WINDOW_NORMAL)

    last_warning_time = time.time()
    distances = []

    try:
        input_video_frame_nr = 0
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                cap.release()
                break

            if not use_webcam and input_video_frame_nr >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
                break

            resized_image = cv2.resize(src=image, dsize=(network_image_height, network_image_width))
            input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)

            result = compiled_model([input_image])[output_key]

            # Normalize the depth result between 0 and 1
            normalized_result = normalize_minmax(result)

            # Detect if object is too close
            min_distance = np.min(normalized_result)
            print(f"Minimum distance (normalized) in frame: {min_distance}")

            distances.append(min_distance)

            if len(distances) > smoothing_window:
                distances.pop(0)

            smoothed_distance = np.mean(distances)
            print(f"Smoothed distance (normalized): {smoothed_distance}")

            current_time = time.time()

            if smoothed_distance < threshold:
                print("Object detected too close!")
                if current_time - last_warning_time > debounce_time:
                    print(f"Showing warning, debounce time: {debounce_time}")
                    show_warning()
                    last_warning_time = current_time

            result_frame = to_rgb(convert_result_to_image(normalized_result))

            result_frame = cv2.resize(result_frame, (target_frame_width, target_frame_height))
            image = cv2.resize(image, (target_frame_width, target_frame_height))

            stacked_frame = np.hstack((image, result_frame))

            cv2.imshow("Monodepth Estimation", stacked_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            input_video_frame_nr += advance_frames
            cap.set(1, input_video_frame_nr)

    except KeyboardInterrupt:
        print("Processing interrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

# 웹캠 스트림을 사용하여 처리 실행
process_stream_with_warning(use_webcam=True, threshold=0.8)
