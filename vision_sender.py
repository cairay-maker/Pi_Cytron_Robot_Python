import cv2
from picamera2 import Picamera2
from libcamera import Transform
import numpy as np
import serial
import time
import os

print("=== RoboCup Vision Sender - Prefer Vertical Line ===")

# ================== CONFIG ==================
USE_FULL_WIDE_MODE = False
PROCESS_WIDTH = 320
PROCESS_HEIGHT = 180
DISPLAY_SCALE = 2.5

ROI_TOP = int(PROCESS_HEIGHT * 0.33)
ROI_BOTTOM = PROCESS_HEIGHT

LOWER_GREEN = np.array([35, 70, 70])
UPPER_GREEN = np.array([85, 255, 255])

SERIAL_PORT = "/dev/serial0"
BAUDRATE = 115200

DEBUG_SHOW_MASKS = True        # ← Toggle this on/off
DEBUG_REFRESH_FRAMES = 30      # ← Refresh every ~1 sec (assumes ~30fps)

BLUR_SIZE = 0                # ← 0 = off, or try 3, 5, 7, 9 (must be odd numbers)

# ================== CAMERA ==================
picam2 = Picamera2()
if USE_FULL_WIDE_MODE:
    config = picam2.create_preview_configuration(main={"size": (PROCESS_WIDTH, PROCESS_HEIGHT)},
                                                 raw={"size": (2304, 1296)},
                                                 transform=Transform(hflip=1, vflip=1))
else:
    config = picam2.create_preview_configuration(main={"size": (PROCESS_WIDTH, PROCESS_HEIGHT)},
                                                 transform=Transform(hflip=1, vflip=1))

picam2.configure(config)

# Autofocus
picam2.set_controls({"AfMode": 2, "AfSpeed": 1})   # 2 = continuous AF, 1 = fast
picam2.start()
time.sleep(2)                                        # give AF time to lock before processing starts

try:
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
except:
    ser = None

frame_count = 0
last_time = time.time()
last_debug_frame = -1   # tracks when we last refreshed debug windows

while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    display = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
    
    frame_count += 1
    
    # Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0) if BLUR_SIZE >= 3 else gray
    roi_gray = blurred[ROI_TOP:ROI_BOTTOM, :]
    binary = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11,8)

    #Block size (currently 11) — the neighbourhood size it uses to calculate the local threshold for each pixel. Must be odd.
    #Small (7, 11)  → reacts to fine local detail → picks up noise and double edges
    #Large (31, 51) → averages over bigger area   → smoother, ignores small bright spots
    #Constant (currently 2) — subtracted from the local average. Higher = more aggressive at ignoring bright areas.
    #Low  (2, 4)  → sensitive, picks up faint edges → more noise
    #High (8, 16) → only strong dark areas survive  → less noise, cleaner single edg
        
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find Contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    error = 0
    cx = PROCESS_WIDTH // 2
    main_contour = None

    if contours:
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        best_score = -1
        for cnt in sorted_contours[:5]:
            if cv2.contourArea(cnt) < 120:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = h / w if w > 0 else 0
            score = aspect * cv2.contourArea(cnt)
            if score > best_score:
                best_score = score
                main_contour = cnt
        
        if main_contour is not None:
            M = cv2.moments(main_contour)
            if M["m00"] > 100:
                cx = int(M["m10"] / M["m00"])
                error = cx - (PROCESS_WIDTH // 2)
            
            overlay = display.copy()
            scaled = [[int(p[0][0]*DISPLAY_SCALE), int((p[0][1] + ROI_TOP)*DISPLAY_SCALE)] for p in main_contour]
            cv2.drawContours(overlay, [np.array(scaled)], -1, (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
            cv2.drawContours(display, [np.array(scaled)], -1, (0, 255, 0), 4)
            
            for cnt in sorted_contours[:3]:
                if cnt is not main_contour and cv2.contourArea(cnt) > 100:
                    scaled_other = [[int(p[0][0]*DISPLAY_SCALE), int((p[0][1] + ROI_TOP)*DISPLAY_SCALE)] for p in cnt]
                    cv2.drawContours(display, [np.array(scaled_other)], -1, (0, 100, 0), 2)

    # Green Square
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in green_contours:
        if cv2.contourArea(cnt) > 800:
            scaled_green = [[int(p[0][0]*DISPLAY_SCALE), int(p[0][1]*DISPLAY_SCALE)] for p in cnt]
            cv2.drawContours(display, [np.array(scaled_green)], -1, (0, 165, 255), 6)

    green_code = 2 if len(green_contours) > 0 else 0

    if ser:
        ser.write(f"V:{error},{green_code}\n".encode())

# ================== DEBUG WINDOWS ==================
    if DEBUG_SHOW_MASKS:
        if frame_count - last_debug_frame >= DEBUG_REFRESH_FRAMES:
            last_debug_frame = frame_count

            # binary is only ROI height, pad it back to full height for clarity
            binary_full = np.zeros((PROCESS_HEIGHT, PROCESS_WIDTH), dtype=np.uint8)
            binary_full[ROI_TOP:ROI_BOTTOM, :] = binary

            # Scale all up so they're easier to see
            gray_display        = cv2.resize(gray,        None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            blurred_display     = cv2.resize(blurred,     None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            roi_gray_display    = cv2.resize(roi_gray,    None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            binary_display      = cv2.resize(binary_full, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            green_display       = cv2.resize(green_mask,  None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)

            # Label them
            cv2.putText(gray_display,     "GRAY",               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
            cv2.putText(blurred_display,  "BLURRED",            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
            cv2.putText(roi_gray_display, "ROI GRAY (cropped)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
            cv2.putText(binary_display,   "BINARY (line mask)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
            cv2.putText(green_display,    "GREEN MASK",         (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

            cv2.imshow("Debug - Gray",        gray_display)
            cv2.imshow("Debug - Blurred",     blurred_display)
            cv2.imshow("Debug - ROI Gray",    roi_gray_display)
            cv2.imshow("Debug - Binary",      binary_display)
            cv2.imshow("Debug - Green Mask",  green_display)
    else:
        cv2.destroyWindow("Debug - Gray")
        cv2.destroyWindow("Debug - Blurred")
        cv2.destroyWindow("Debug - ROI Gray")
        cv2.destroyWindow("Debug - Binary")
        cv2.destroyWindow("Debug - Green Mask")
    # ====================================================

    # Center + Error
    scaled_w = int(PROCESS_WIDTH * DISPLAY_SCALE)
    center_x = scaled_w // 2
    detected_x = int(cx * DISPLAY_SCALE)
    detected_y = int((ROI_TOP + (ROI_BOTTOM - ROI_TOP)//2) * DISPLAY_SCALE)

    cv2.rectangle(display, (0, int(ROI_TOP*DISPLAY_SCALE)), 
                  (scaled_w, int(ROI_BOTTOM*DISPLAY_SCALE)), (255, 0, 0), 3)

    cv2.line(display, (center_x, int(ROI_TOP*DISPLAY_SCALE)), 
             (center_x, int(ROI_BOTTOM*DISPLAY_SCALE)), (255, 255, 100), 2)
    cv2.circle(display, (detected_x, detected_y), 12, (0, 0, 255), -1)
    cv2.line(display, (center_x, detected_y), (detected_x, detected_y), (0, 255, 255), 3)

    cv2.putText(display, f"Error: {error}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Vision Debug - Prefer Vertical", display)
    
    if frame_count % 20 == 0:
        print(f"FPS: {20 / (time.time() - last_time):.1f} | Error: {error:4d} | Green: {green_code} | Contours: {len(contours)}")
        last_time = time.time()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
if ser: ser.close()
cv2.destroyAllWindows()