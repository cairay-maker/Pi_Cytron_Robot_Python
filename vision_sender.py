import cv2
from picamera2 import Picamera2
from libcamera import Transform
import numpy as np
import serial
import time

print("=== RoboCup Vision Sender - Green Square and Black Line ===")

# ================================================================
# BLOCK 1: CONFIGURATION
# ================================================================

USE_FULL_WIDE_MODE = False
PROCESS_WIDTH = 320
PROCESS_HEIGHT = 180
DISPLAY_SCALE = 2.5  # Display Window Scaler

# ROI: Region of Interest — visual reference only, blue rectangle on display
# Future use: re-enable ROI crop here when needed
ROI_TOP = int(PROCESS_HEIGHT * 0.20)
ROI_BOTTOM = PROCESS_HEIGHT

# Green square detection range in HSV (Hue, Saturation, Value):
LOWER_GREEN = np.array([35, 80, 40])
UPPER_GREEN = np.array([85, 255, 255])

# Color code for display in BGR (BLUE, GREEN, RED)
COLOR_BLUE          = (255, 0, 0)      # Blue
COLOR_LIGHT_BLUE    = (255, 200, 100)  # Light Sky Blue
COLOR_GREEN         = (0, 255, 0)      # Green
COLOR_DARK_GREEN    = (0, 200, 0)      # Dark Green
COLOR_RED           = (0, 0, 255)      # Red
COLOR_CYAN          = (255, 255, 0)    # Cyan
COLOR_YELLOW        = (0, 255, 255)    # Yellow
COLOR_ORANGE        = (0, 165, 255)    # Orange
COLOR_MAGENTA       = (255, 0, 255)    # Magenta
COLOR_PURPLE        = (128, 0, 128)    # Purple
COLOR_WHITE         = (255, 255, 255)  # White
COLOR_GRAY          = (100, 100, 100)  # Gray

# Serial port and baudrate for communication with the robot controller
SERIAL_PORT = "/dev/serial0"
BAUDRATE = 115200

# Show debug windows for binary and green mask. Refresh every N frames (to reduce CPU load)
DEBUG_SHOW_MASKS = True        # ← Toggle on/off
DEBUG_REFRESH_FRAMES = 30      # ← ~1 sec at 30fps
BLUR_SIZE = 0                  # ← 0=off, or 3,5,7,9 (must be odd)

# Performance timing: set True to print per-section ms breakdown every 20 frames
DEBUG_TIMING = True            # ← Toggle timing on/off

# ================== GREEN SQUARE CONFIG ==================
# Tune GREEN_MIN_AREA based on camera FOV and robot height:
# Measured under 320x180, 40x40=1600, a little smaller = 1200
GREEN_MIN_AREA = 1200

# Vertical split: blobs above this y = "Up" (far ahead), below = "Dn" (close/arriving)
# Default: middle of frame. Raise it if you want "Dn" zone to be larger.
GREEN_SPLIT_Y = PROCESS_HEIGHT // 2      # ← tune this (0=top, 180=bottom)

# ================================================================
# BLOCK 2: CAMERA SETUP
# Future: move to vision/camera.py
# ================================================================

picam2 = Picamera2()
if USE_FULL_WIDE_MODE:
    config = picam2.create_preview_configuration(main={"size": (PROCESS_WIDTH, PROCESS_HEIGHT)},
                                                 raw={"size": (2304, 1296)},
                                                 transform=Transform(hflip=1, vflip=1))
else:
    config = picam2.create_preview_configuration(main={"size": (PROCESS_WIDTH, PROCESS_HEIGHT)},
                                                 transform=Transform(hflip=1, vflip=1))

picam2.configure(config)
picam2.set_controls({"AfMode": 2, "AfSpeed": 1})   # continuous AF, fast
picam2.start()
time.sleep(2)                                        # wait for AF to lock

# ================================================================
# BLOCK 3: SERIAL SETUP
# Future: move to comms/serial_link.py
# ================================================================

# Serial port and baudrate for communication with the robot controller
# Protocol: V:{error},{green_code},{blob_count},{distance_px}
#   error        : int, negative=line left, positive=line right of center
#   green_code   : 0=none, 1=left, 2=right, 3=U-turn
#   blob_count   : raw green blob count (1-4), diagnostic
#   distance_px  : avg pixels from active Dn blob centroid(s) to bottom of frame
#                  0=no Dn blob yet, >0=actionable (suggest trigger at <60)
try:
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
except:
    ser = None

# ================================================================
# MAIN LOOP
# ================================================================

frame_count = 0
last_time = time.time()
last_debug_frame = -1

while True:

    t_start = time.perf_counter()

    # ================================================================
    # BLOCK 4: FRAME CAPTURE
    # Future: move to vision/camera.py → camera.capture()
    # ================================================================

    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Scale up for display — all drawing happens on this larger image
    display = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)

    frame_count += 1

    t_capture = time.perf_counter()

    # ================================================================
    # BLOCK 5: GREEN SQUARE DETECTION
    # Done FIRST so green mask can be subtracted from line binary below
    # Future: move to vision/green_detector.py → green.detect(frame)
    # ================================================================

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    CENTER_X = PROCESS_WIDTH // 2

    # Collect all valid green blobs with position info
    green_blobs = []
    for cnt in green_contours:
        if cv2.contourArea(cnt) > GREEN_MIN_AREA:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                gcx = int(M["m10"] / M["m00"])
                gcy = int(M["m01"] / M["m00"])
                h_side = "L" if gcx < CENTER_X else "R"
                v_side = "Up" if gcy < GREEN_SPLIT_Y else "Dn"
                zone   = f"{h_side}{v_side}"     # "LUp", "LDn", "RUp", "RDn"
                green_blobs.append({
                    "cx": gcx, "cy": gcy,
                    "side": h_side, "vside": v_side,
                    "zone": zone, "contour": cnt
                })

    # Sort blobs by y descending — closest to robot first
    green_blobs.sort(key=lambda b: b["cy"], reverse=True)

    t_green = time.perf_counter()

    # ================================================================
    # BLOCK 6: GREEN CLASSIFICATION
    # Future: part of vision/green_detector.py → green.classify()
    # ================================================================

    # green_code:
    #   0 = no green
    #   1 = turn LEFT
    #   2 = turn RIGHT
    #   3 = U-TURN (dead end)
    green_code = 0
    decision_blobs = green_blobs   # default: all blobs used

    if len(green_blobs) == 1:
        decision_blobs = green_blobs
        green_code = 1 if green_blobs[0]["side"] == "L" else 2

    elif len(green_blobs) == 2:
        dn_only = [b for b in green_blobs if b["vside"] == "Dn"]
        up_only = [b for b in green_blobs if b["vside"] == "Up"]

        if len(dn_only) == 1 and len(up_only) == 1:
            # One Dn + one Up → the Up blob is a background marker, ignore it
            # Decision is made ONLY from the Dn blob
            decision_blobs = dn_only
            green_code = 1 if dn_only[0]["side"] == "L" else 2
        else:
            # Both Dn or both Up → use all two for decision
            decision_blobs = green_blobs
            sides = {b["side"] for b in green_blobs}
            if sides == {"L", "R"}:
                green_code = 3                      # one each side = U-turn
            elif sides == {"L"}:
                green_code = 1
            else:
                green_code = 2

    elif len(green_blobs) >= 3:
        decision_blobs = green_blobs[:2]            # only closest two decide
        sides = {b["side"] for b in decision_blobs}
        if sides == {"L", "R"}:
            green_code = 3
        elif sides == {"L"}:
            green_code = 1
        else:
            green_code = 2

    # Distance: only from active Dn blobs in decision_blobs
    # For U-turn (2 orange Dn blobs), use the average of both
    # 0 = no Dn blob yet (not actionable), >0 = Dn blob active
    distance_px = 0
    active_dn_blobs = [b for b in decision_blobs if b["vside"] == "Dn"]
    if active_dn_blobs:
        avg_y = sum(b["cy"] for b in active_dn_blobs) / len(active_dn_blobs)
        distance_px = int(PROCESS_HEIGHT - avg_y)

    t_green_class = time.perf_counter()

    # ================================================================
    # BLOCK 7: LINE PREPROCESSING
    # Future: move to vision/line_detector.py → line.preprocess(frame)
    # ================================================================

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0) if BLUR_SIZE >= 3 else gray
    blurred = gray  # Skip blurring for performance; can be re-enabled if needed.

    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 8)
    # Threshold tuning reference:
    #   Block size (arg 5): must be odd. Small(7-11)=sensitive/noisy. Large(31-51)=smooth/robust.
    #   Constant  (arg 6):  Low(2-4)=picks up faint edges. High(8-16)=only strong dark areas survive.

    # Subtract green mask — prevents dark green squares confusing the line detector
    binary = cv2.bitwise_and(binary, cv2.bitwise_not(green_mask))

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    t_line_pre = time.perf_counter()

    # ================================================================
    # BLOCK 8: LINE CONTOUR DETECTION
    # Full frame used — ROI_TOP rectangle is visual reference only
    # Future: move to vision/line_detector.py → line.detect(binary)
    # ================================================================

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Pre-filter by area immediately — avoids sorting/scoring hundreds of tiny contours
    # MIN_CONTOUR_AREA tuning:
    #   Too low  → slow, noise contours still processed
    #   Too high → may miss thin line sections
    MIN_CONTOUR_AREA = 300
    contours = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]

    error = 0
    cx = PROCESS_WIDTH // 2
    main_contour = None

    if contours:
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        best_score = -1
        for cnt in sorted_contours[:10]:
            x, y, w, h = cv2.boundingRect(cnt)

            # Only accept contours whose bounding box reaches into the action zone
            if (y + h) < ROI_TOP:
                continue

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
            scaled = [[int(p[0][0] * DISPLAY_SCALE), int(p[0][1] * DISPLAY_SCALE)] for p in main_contour]
            cv2.drawContours(overlay, [np.array(scaled)], -1, (COLOR_RED), -1)
            cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
            cv2.drawContours(display, [np.array(scaled)], -1, (COLOR_GREEN), 4)

            for cnt in sorted_contours[:5]:
                if cnt is not main_contour and cv2.contourArea(cnt) > 100:
                    scaled_other = [[int(p[0][0] * DISPLAY_SCALE), int(p[0][1] * DISPLAY_SCALE)] for p in cnt]
                    _x, _y, _w, _h = cv2.boundingRect(cnt)
                    if (_y + _h) < ROI_TOP:
                        # Approaching lines above action zone — light blue
                        cv2.drawContours(display, [np.array(scaled_other)], -1, (COLOR_LIGHT_BLUE), 2)
                    else:
                        # Ignored lines inside action zone — dark green
                        cv2.drawContours(display, [np.array(scaled_other)], -1, (COLOR_DARK_GREEN), 2)

    t_line_contour = time.perf_counter()

    # ================================================================
    # BLOCK 9: SERIAL OUTPUT
    # Future: move to comms/serial_link.py → serial.send(...)
    # ================================================================

    # Protocol: V:{error},{green_code},{blob_count},{distance_px}
    #   error        : int, negative=line left, positive=line right of center
    #   green_code   : 0=none, 1=left, 2=right, 3=U-turn
    #   blob_count   : raw green blob count (1-4), diagnostic
    #   distance_px  : avg pixels from active Dn blob centroid(s) to bottom of frame
    #                  0=no Dn blob yet, >0=actionable (suggest trigger at <60)
    if ser:
        ser.write(f"V:{error},{green_code},{len(green_blobs)},{distance_px}\n".encode())

    # ================================================================
    # BLOCK 10: DISPLAY — GREEN BLOBS
    # Future: move to display/debug_view.py → view.draw_green(...)
    # ================================================================

    for blob in green_blobs:
        cnt = blob["contour"]
        scaled_green = [[int(p[0][0] * DISPLAY_SCALE), int(p[0][1] * DISPLAY_SCALE)] for p in cnt]

        # Only orange + distance line if: used for decision AND in Dn zone
        is_active = (blob in decision_blobs) and (blob["vside"] == "Dn")

        if is_active:
            cv2.drawContours(display, [np.array(scaled_green)], -1, COLOR_ORANGE, 6)
            color_dot = COLOR_ORANGE
        else:
            cv2.drawContours(display, [np.array(scaled_green)], -1, COLOR_PURPLE, 3)
            color_dot = COLOR_PURPLE

        dx = int(blob["cx"] * DISPLAY_SCALE)
        dy = int(blob["cy"] * DISPLAY_SCALE)
        cv2.circle(display, (dx, dy), 8, color_dot, -1)
        cv2.putText(display, blob["zone"], (dx + 10, dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_dot, 2)

        # Distance line only for active (Dn) blobs
        if is_active:
            bottom_y = int(PROCESS_HEIGHT * DISPLAY_SCALE)
            cv2.line(display, (dx, dy), (dx, bottom_y), COLOR_ORANGE, 2)
            cv2.line(display, (dx - 8, bottom_y), (dx + 8, bottom_y), COLOR_ORANGE, 2)
            label_y = dy + (bottom_y - dy) // 2
            blob_dist = PROCESS_HEIGHT - blob["cy"]
            dist_label = f"{blob_dist}px"
            if len(active_dn_blobs) == 2:
                dist_label = f"{blob_dist}px*"    # * = averaged with partner blob
            cv2.putText(display, dist_label, (dx + 6, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_ORANGE, 2)

    # ================================================================
    # BLOCK 11: DISPLAY — OVERLAYS (rulers, ROI, zones, status, packet)
    # Future: move to display/debug_view.py → view.draw_overlays(...)
    # ================================================================

    scaled_w = int(PROCESS_WIDTH * DISPLAY_SCALE)
    scaled_h = int(PROCESS_HEIGHT * DISPLAY_SCALE)
    center_x   = scaled_w // 2
    detected_x = int(cx * DISPLAY_SCALE)
    detected_y = int((ROI_TOP + (ROI_BOTTOM - ROI_TOP) // 2) * DISPLAY_SCALE)

    # Vertical Ruler - Left Edge (every 20 pixels in original resolution)
    for i in range(0, PROCESS_HEIGHT + 1, 20):
        y = int(i * DISPLAY_SCALE)
        tick_len = 25 if i % 100 == 0 else 12
        cv2.line(display, (0, y), (tick_len, y), COLOR_BLUE, 2)
        if i % 100 == 0:
            cv2.putText(display, str(i), (tick_len + 8, y + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_BLUE, 2)

    # Horizontal Ruler - Bottom Edge
    for i in range(0, PROCESS_WIDTH + 1, 20):
        x = int(i * DISPLAY_SCALE)
        tick_len = 25 if i % 100 == 0 else 12
        cv2.line(display, (x, scaled_h), (x, scaled_h - tick_len), COLOR_BLUE, 2)
        if i % 100 == 0:
            cv2.putText(display, str(i), (x - 20, scaled_h - tick_len - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_BLUE, 2)

    # ROI action zone box — visual reference only (blue rectangle)
    cv2.rectangle(display, (0, int(ROI_TOP * DISPLAY_SCALE)),
                  (scaled_w, int(ROI_BOTTOM * DISPLAY_SCALE)), COLOR_BLUE, 3)

    # Green Up/Dn split line (cyan)
    split_y_scaled = int(GREEN_SPLIT_Y * DISPLAY_SCALE)
    cv2.line(display, (0, split_y_scaled), (scaled_w, split_y_scaled), (COLOR_CYAN), 2)

    # Zone corner labels
    cv2.putText(display, "LUp", (5,             split_y_scaled - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CYAN, 2)
    cv2.putText(display, "RUp", (scaled_w - 50, split_y_scaled - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CYAN, 2)
    cv2.putText(display, "LDn", (5,             split_y_scaled + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CYAN, 2)
    cv2.putText(display, "RDn", (scaled_w - 50, split_y_scaled + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CYAN, 2)

    # Center vertical line + error dot + error line
    cv2.line(display, (center_x, int(ROI_TOP * DISPLAY_SCALE)),
             (center_x, int(ROI_BOTTOM * DISPLAY_SCALE)), COLOR_YELLOW, 2)
    cv2.circle(display, (detected_x, detected_y), 12, COLOR_RED, -1)
    cv2.line(display, (center_x, detected_y), (detected_x, detected_y), COLOR_YELLOW, 3)

    # Status text labels and colors per green_code
    green_labels = {0: "NONE",
                    1: "LEFT",
                    2: "RIGHT",
                    3: "U-TURN"}
    green_label_colors = {0: COLOR_GRAY,
                          1: COLOR_DARK_GREEN,
                          2: COLOR_DARK_GREEN,
                          3: COLOR_ORANGE}
    zones_seen = [b["zone"] for b in green_blobs]

    cv2.putText(display, f"Error: {error}",
                (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
    cv2.putText(display, f"Green: {green_code} {len(green_blobs)} {green_labels[green_code]} {zones_seen}",
                (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, green_label_colors[green_code], 2)
    if green_blobs:
        cv2.putText(display, f"Dist: {distance_px}px",
                    (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ORANGE, 2)

    # Raw packet display — top right corner (BEFORE imshow)
    packet_str = f"V:{error},{green_code},{len(green_blobs)},{distance_px}"
    (pw, ph), _ = cv2.getTextSize(packet_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    packet_x = scaled_w - pw - 40
    cv2.putText(display, packet_str, (packet_x, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)

    t_draw = time.perf_counter()

    # ================================================================
    # BLOCK 12: DISPLAY — SHOW MAIN WINDOW
    # imshow must always be LAST — anything drawn after this is lost
    # ================================================================

    cv2.imshow("Vision Debug - Green Aware", display)

    t_imshow = time.perf_counter()

    # ================================================================
    # BLOCK 13: DEBUG WINDOWS — BINARY AND GREEN MASK
    # Refreshed every DEBUG_REFRESH_FRAMES to reduce CPU load
    # Future: move to display/debug_view.py → view.draw_debug_windows(...)
    # ================================================================

    if DEBUG_SHOW_MASKS:
        if frame_count - last_debug_frame >= DEBUG_REFRESH_FRAMES:
            last_debug_frame = frame_count

            # gray_display    = cv2.resize(gray,       None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            # blurred_display = cv2.resize(blurred,    None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            binary_display  = cv2.resize(binary,     None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            green_display   = cv2.resize(green_mask, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)

            # cv2.putText(gray_display,    "GRAY",    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
            # cv2.putText(blurred_display, "BLURRED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
            cv2.putText(binary_display,  "BINARY (full frame)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
            cv2.putText(green_display,   "GREEN MASK",          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

            # cv2.imshow("Debug - Gray",       gray_display)
            # cv2.imshow("Debug - Blurred",    blurred_display)
            cv2.imshow("Debug - Binary",     binary_display)
            cv2.imshow("Debug - Green Mask", green_display)
    else:
        # cv2.destroyWindow("Debug - Gray")
        # cv2.destroyWindow("Debug - Blurred")
        cv2.destroyWindow("Debug - Binary")
        cv2.destroyWindow("Debug - Green Mask")

    t_debug = time.perf_counter()

    # ================================================================
    # BLOCK 14: TIMING + FPS PRINT + QUIT
    # ================================================================

    # Fix 2: single waitKey call — handles both display refresh and quit detection
    # Previously called twice per loop which doubled X11 flush cost (~18ms wasted)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    t_end = time.perf_counter()

    if frame_count % 20 == 0:
        print(f"FPS: {20 / (time.time() - last_time):.1f} | Error: {error:4d} | "
              f"Green: {green_labels[green_code]} {zones_seen} | "
              f"Blobs: {len(green_blobs)} | Dist: {distance_px}px")
        if DEBUG_TIMING:
            print(f"  capture:{(t_capture-t_start)*1000:.1f}ms "
                  f"resize:{(t_capture-t_start)*1000:.1f}ms "
                  f"green:{(t_green-t_capture)*1000:.1f}ms "
                  f"classify:{(t_green_class-t_green)*1000:.1f}ms "
                  f"line_pre:{(t_line_pre-t_green_class)*1000:.1f}ms "
                  f"line_cnt:{(t_line_contour-t_line_pre)*1000:.1f}ms "
                  f"draw:{(t_draw-t_line_contour)*1000:.1f}ms "
                  f"imshow:{(t_imshow-t_draw)*1000:.1f}ms "
                  f"debug:{(t_debug-t_imshow)*1000:.1f}ms "
                  f"waitkey:{(t_end-t_debug)*1000:.1f}ms "
                  f"TOTAL:{(t_end-t_start)*1000:.1f}ms")
        last_time = time.time()

# ================================================================
# BLOCK 15: CLEANUP
# ================================================================

picam2.stop()
if ser: ser.close()
cv2.destroyAllWindows()