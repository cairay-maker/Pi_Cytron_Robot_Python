# Pi_Cytron_Robot_Python

This repository contains the OpenCV Vision Brain for the Team Leli RoboCup Rescue Line robot. It runs on a Raspberry Pi 4, processing camera frames to detect the black line and green turn indicators, and sends steering corrections to the Cytron RP2350 via Hardware UART.

## 📸 Vision Processing Strategy

### 1. Region of Interest (ROI) "Blinding"
To prevent the robot from reacting to upcoming intersections too early, the camera frame (320x180) is cropped. The algorithm only evaluates the bottom section of the screen (e.g., `Y=120` to `Y=180`).
*   **Visual Debugging:** The ROI is drawn on the output display as a **Bright Blue Rectangle**.

### 2. Line Detection (Error Calculation)
The script calculates the centroid (center of mass) of the largest vertical black contour inside the ROI.
*   **Visual Debugging:** The detected black line is shaded with a **semi-transparent Red overlay**.
*   **Targeting:** A **Bright Green Vertical Line** shows exactly where the algorithm thinks the center of the line is.
*   `Error = Detected_Center_X - Frame_Center_X`

### 3. Green Square Detection
The script uses an HSV color range to identify Robocup green squares.
*   `0` = No Green
*   `1` = Left Green (To be implemented)
*   `2` = Right Green
*   `3` = U-Turn (To be implemented)

---

## 🔌 Hardware UART Connection to Cytron

The Pi sends the vision data to the Cytron exactly 30 times a second using this serial format:
`V:<error_pixels>,<green_code>\n`

**Wiring Map:**
*   **RPi TX (Pin 8 / GPIO 14)** → Connects to Cytron RX (GP5)
*   **RPi RX (Pin 10 / GPIO 15)** → Connects to Cytron TX (GP4)
*   **GND (Pin 6)** → Connects to Cytron GND
*   **Baud Rate:** `115200`

---

## 🚀 Running the Script
*Ensure you are running this in a desktop environment (VNC or HDMI out) if you want to see the `cv2.imshow` debug window.*

```bash
python3 vision_sender.py
```