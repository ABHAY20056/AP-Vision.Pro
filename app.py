import cv2
import streamlit as st
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import os
import urllib.request

# --- 1. BRANDING & ASSETS ---
BRAND_NAME = " AP-Vision.Pro"
st.set_page_config(page_title=BRAND_NAME, layout="centered")

MODEL_PATH = "pose_landmarker_lite.task"
if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task", MODEL_PATH)

# --- 2. AI ENGINE SETUP ---
@st.cache_resource
def load_ai():
    yolo = YOLO('yolo11n.pt')
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE)
    pose_detector = vision.PoseLandmarker.create_from_options(options)
    return yolo, pose_detector

yolo_model, pose_detector = load_ai()

# --- 3. THE WEBRTC CALLBACK (The "Cloud Fix") ---
# This class runs on the server and processes frames sent from the user's browser
class VideoProcessor:
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # YOLO Tracking
        results = yolo_model.track(img, persist=True, verbose=False, conf=0.3)

        if results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0]) if box.id is not None else 0
                
                # Draw Box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"ID: {track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. UI INTERFACE ---
st.title(f" {BRAND_NAME}")
st.write("Real-time AI surveillance deployed via WebRTC.")

# This configuration ensures the connection works through firewalls (Google's STUN server)
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_streamer(
    key="sentinel-ai",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
) 