# 🎥 Video Event Detection (YOLO)

A Python-based tool for detecting human activity in surveillance videos using YOLOv8.

The project processes video recordings, detects events (e.g. person appearance), extracts relevant clips, saves key frames, and packs results into an archive.

---

## 🚀 Features

- 🔍 Human detection using YOLOv8
- 🎯 ROI-based filtering (per camera logic)
- ⏱ Event time detection with cooldown
- 🖼 Saving event frames with timestamps
- ✂️ Automatic video clipping around events
- 📦 Export results as ZIP archive
- ⚡ Optimized frame processing

---

## ⚙️ Installation

pip install -r requirements.txt

Make sure you have Python 3.9+ installed

---

## ▶️ Usage

Run any camera processor:

python main.py 2

---

## 🧠 How It Works
1. Parses video filename to extract timestamp
2. Loads YOLOv8 model
3. Iterates through video frames
4. Detects persons inside region of interest (ROI)
5. Triggers events with cooldown logic
6. Saves:
   event frames,
   video segments
7. Merges intervals and exports final video
8. Packs everything into a ZIP archive

---

## 📸 Output

After processing, you get:

🎞 Processed video with only event segments  
🖼 Frames where events occurred  
📦 ZIP archive containing all results

---

## 📸 Configuration

Basic settings are defined in:  

video_utils.py → DEFAULT_SETTINGS

Includes:
- detection threshold
- cooldown time
- frame skipping
- pre/post event duration

The current AxxonNet web-client route is resolved automatically for each
camera. `PROXY_KEY` is not required in `.env`.

Access to `BASE_API` may require the corporate VPN. If the VPN is disabled or
the internal API is unreachable, the downloader stops with an explicit VPN
connectivity hint.

---

## 🧩 Notes
- YOLO model (yolov8x.pt) is loaded automatically  
- GPU (CUDA) is used if available  
- Each camera has custom ROI logic
