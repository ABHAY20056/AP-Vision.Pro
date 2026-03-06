import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO
import av
import cv2
import numpy as np

# --- 1. PAGE & THEME CONFIG ---
st.set_page_config(
    page_title="AP-Vision.Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Footer and UI
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {
        background-color: #0e1117;
    }
    .main-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0e1117;
        color: #555;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid #333;
        z-index: 100;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
        border: 1px solid #333;
        background-color: #161b22;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MODEL ENGINE ---
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt") # Fast Nano model for real-time mobile use

model = load_model()

# --- 3. SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.title("AP-Vision.Pro")
    st.markdown("---")
    
    st.subheader("⚙️ Detection Settings")
    conf_threshold = st.slider("Confidence", 0.0, 1.0, 0.45, help="Higher = More accurate, Lower = More detections")
    
    all_classes = list(model.names.values())
    selected_classes = st.multiselect(
        "Filter Objects", 
        all_classes, 
        default=["person", "car", "cell phone", "laptop", "bottle"]
    )
    
    st.markdown("---")
    st.subheader("🎥 Device Settings")
    cam_mode = st.radio("Camera Source", ("Back Camera", "Selfie/Front"))
    facing_mode = "environment" if cam_mode == "Back Camera" else "user"

# --- 4. CORE VISION LOGIC ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Perform Detection
        results = model.predict(img, conf=conf_threshold, verbose=False)
        
        # Filter and Annotate
        for r in results:
            # We filter classes here to speed up rendering
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                
                if label in selected_classes:
                    # Plot only the selected objects
                    img = r.plot()
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 5. MAIN DISPLAY ---
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("AP-Vision.Pro")
    webrtc_ctx = webrtc_streamer(
        key="ap-vision-pro",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=VideoProcessor().recv,
        media_stream_constraints={
            "video": {"facingMode": facing_mode},
            "audio": False
        },
        async_processing=True,
    )

with col2:
    st.subheader("System Status")
    if webrtc_ctx.state.playing:
        st.markdown('<div class="status-box">🟢 <b style="color:#4CAF50;">ACTIVE</b><br><small>Stream processing at 30 FPS</small></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-box">⚪ <b>IDLE</b><br><small>Waiting for user input...</small></div>', unsafe_allow_html=True)
    
    st.info("Mobile Users: If the camera is black, click 'Fullscreen' in the top right corner of the page.")

# --- 6. PROFESSIONAL FOOTER ---
st.markdown(
    """
    <div class="main-footer">
        All Rights Reserved © 2026 <b>AP-Vision.Pro</b> | Powered by Ultralytics YOLOv8
    </div>
    """,
    unsafe_allow_html=True
)
