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

6/25/2026 Claude AI - Green Square

Cytron Side — How to Handle the Serial Messages
Here's what the Cytron receives and what to do with it. This is pseudocode for your MicroPython/C++ on the Cytron:
Serial message format:
  V:{error},{green_code},{blob_count},{distance_factor}

Examples:
  V:12,0,0,0.00    → line slightly right, no green, follow normally
  V:3,2,1,0.85     → line near center, green RIGHT seen far ahead → start slowing
  V:1,2,1,0.20     → green RIGHT now very close → ready to turn
  V:0,3,2,0.45     → at intersection, U-turn indicated by 2 blobs
State machine on the Cytron:
STATE: LINE_FOLLOW  (default)
  - Use error for PID steering as normal
  - If green_code > 0 AND pending_turn == 0:
      → latch: pending_turn = green_code
      → latch: pending_blobs = blob_count
  - If distance_factor < 0.3 AND pending_turn > 0:
      → transition to APPROACHING

STATE: APPROACHING
  - Slow down to intersection speed
  - Keep line following with error
  - When error ≈ 0 AND line lost (no contour) → at T-junction
      → transition to EXECUTE_TURN

STATE: EXECUTE_TURN
  - pending_turn == 1 → spin left  until line reacquired
  - pending_turn == 2 → spin right until line reacquired
  - pending_turn == 3 → spin 180°  (use IMU for angle) then find line
  - When line reacquired:
      → clear pending_turn = 0
      → transition to LINE_FOLLOW

KEY RULES:
  1. Latch on FIRST detection — ignore repeated green_code until turn is done
  2. distance_factor drives speed: 0.8+ = full speed, 0.5 = medium, 0.3 = slow
  3. For U-turn (code 3): use your BNO085 IMU to rotate exactly 180°
  4. blob_count is diagnostic — log it but only green_code drives decisions

GREEN_MIN_AREA Tuning Guide
Camera setup                        Suggested GREEN_MIN_AREA
─────────────────────────────────────────────────────────────
Narrow FOV, camera close to ground  1200 – 2000
Standard FOV, mid height (default)   600 – 1000  ← start here
Wide FOV, camera high up             300 –  600
320×180 resolution (current)         800  ← your starting point
640×480 resolution                  3000 – 5000  (scales with px²)


Here's a clean summary to guide your Cytron implementation:

Cytron Turn Handling — Full Strategy
Serial Message Received from Pi
V:{error},{green_code},{blob_count},{distance_px}

green_code:  0=none  1=LEFT  2=RIGHT  3=U-TURN
distance_px: 0=no Dn blob yet,  >0=Dn blob active (pixels from centroid to bottom)

State Machine
STATE 0 — LINE_FOLLOW  (default)
────────────────────────────────
- Run PID using error from Pi
- Watch for: distance_px > 0 (first Dn blob appears)
- When distance_px > 0:
    → latch pending_turn = green_code
    → transition to STATE 1

STATE 1 — APPROACHING
──────────────────────
- Keep PID running but reduce speed
- IGNORE all further green_code updates  ← critical
  (camera may flip between codes as more squares enter frame)
- When distance_px < 60:
    → transition to STATE 2

STATE 2 — EXECUTE_TURN
───────────────────────
- Stop PID immediately
- Disable ALL vision input from Pi  ← camera will see confusing squares during spin
- Execute turn based on latched pending_turn:
    pending_turn == 1  → spin LEFT
    pending_turn == 2  → spin RIGHT
    pending_turn == 3  → spin 180° (U-turn)
- Use BNO085 IMU as the turn controller:
    LEFT/RIGHT turns:  rotate until IMU heading change >= 80°
                       then switch to line-seek mode (STATE 3)
    U-TURN:            rotate until IMU heading change >= 170°
                       then switch to line-seek mode (STATE 3)

STATE 3 — LINE_SEEK
────────────────────
- Robot crept past intersection, line is somewhere nearby
- Slow forward creep while reading Pi error
- When abs(error) < threshold AND error is stable for N frames:
    → clear pending_turn = 0
    → clear distance_px latch
    → transition to STATE 0

Why IMU + 80° is the Right Call
Problem without IMU floor:
  Robot starts turning → camera sees new squares → distance_px triggers again
  → robot tries to turn AGAIN mid-turn → disaster

With IMU 80° floor:
  Robot commits to the turn regardless of what camera sees
  80° ensures the robot has physically passed the intersection
  Remaining ~10° is recovered naturally by PID in STATE 3
  This forgives slight overshoot or undershoot cleanly

Key Rules Summary
1. LATCH on first Dn detection — green_code is set once, never updated mid-turn
2. IGNORE Pi vision entirely during STATE 2 — IMU is in charge
3. IMU floor angles:
     Left/Right turn  →  80°  minimum
     U-turn           →  170° minimum
4. Exit turn via LINE_SEEK not directly to LINE_FOLLOW
     — gives the robot time to find the line before full PID resumes
5. Only resume full speed once error is stable in STATE 0

Timing Overview
Robot approaching:     STATE 0 → STATE 1  (Dn blob appears)
Robot slowing:         STATE 1            (distance_px counting down)
distance_px < 60:      STATE 1 → STATE 2  (commit to turn)
IMU reaches 80°:       STATE 2 → STATE 3  (turn complete)
Line reacquired:       STATE 3 → STATE 0  (resume full PID)

One Extra Suggestion — Speed Profile
Rather than a hard slow-down at STATE 1, use distance_px directly to scale speed smoothly:
// Pseudocode for Cytron
base_speed = 80
if state == APPROACHING:
    speed = base_speed * (distance_px / 120.0)  // ramps down as dist shrinks
    speed = max(speed, 25)                       // never slower than 25 (keep moving)
This gives a smooth deceleration curve into the intersection rather than a sudden speed step, which makes the IMU turn start from a more controlled position.