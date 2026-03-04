# 🛡️ AP-Vision Pro
### Developed by: Abhay Pandey

![Status](https://img.shields.io/badge/Status-Production--Ready-success?style=for-the-badge)
![Tech](https://img.shields.io/badge/Tech-YOLOv11%20|%20WebRTC-blue?style=for-the-badge)

**AP-Vision Pro** is a cutting-edge, privacy-centric AI surveillance system designed for real-time personnel tracking and safety auditing. It combines the speed of YOLOv11 with the precision of MediaPipe to provide actionable insights at the edge.



---

## 🚀 Key Engineering Features
* **Real-Time Tracking:** Utilizes **YOLOv11** with **ByteTrack** for persistent identity management of multiple subjects.
* **Behavioral Intelligence:** Integrates **MediaPipe Tasks API** to analyze 33 skeletal landmarks for action recognition.
* **Cloud-Native Architecture:** Implements **WebRTC** to enable secure, low-latency video streaming from the user's browser to the inference engine.
* **Data Sovereignty:** Processes video frames locally/in-memory to ensure zero data retention and total privacy.
* **Audit Reporting:** Built-in system to generate and download CSV safety reports for industrial compliance.

---

## 🏗️ System Architecture
The system follows a modular, asynchronous design to prevent UI freezing during heavy AI computation:

1.  **Frontend:** Streamlit-based mobile-responsive UI.
2.  **Streaming:** WebRTC peer-to-peer connection for live frame transfer.
3.  **Inference Engine:** YOLOv11 (Detection) + MediaPipe (Pose) + Custom Logic.
4.  **Data Layer:** Session-state logging for exportable reports.



---

## 🛠️ Quick Start

### 1. Clone & Install
```bash
git clone [https://github.com/YOUR_USERNAME/AP-Vision-Pro.git](https://github.com/YOUR_USERNAME/AP-Vision-Pro.git)
cd AP-Vision-Pro
pip install -r requirements.txt