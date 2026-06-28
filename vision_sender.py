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

# Padding around green bounding box used in Plan B centroid filtering
# Contour points within GREEN_EXCLUSION_PAD pixels of a green square edge
# are excluded from centroid calculation
# 3px = ~1.5mm at your camera scale (1cm = 20px)
# Too small → green edge pixels still influence centroid
# Too large → excludes too much tape near the green square
GREEN_EXCLUSION_PAD = 3        # ← tune: try 3, 5, 8

# ================== LINE DETECTION CONFIG ==================
# Camera sees 16cm wide x 9cm tall → 1cm = 20px in both axes (square mapping)
# All pixel values below can be converted to real world: px ÷ 20 = cm

# Adaptive threshold parameters:
#   THRESH_BLOCK_SIZE: neighbourhood size for local threshold calculation (must be odd)
#     Small (7–11) → sensitive, picks up shadows and noise
#     Large (21–51) → smoother, ignores gradual brightness changes like shadows
THRESH_BLOCK_SIZE = 21     # ✅ tuned — shadows suppressed

#   THRESH_CONSTANT: subtracted from local average — higher = only darker pixels survive
#     Low  (4–8)  → picks up faint edges and shadows
#     High (12–18) → only strong black areas like tape survive
THRESH_CONSTANT = 16       # ✅ tuned — shadows suppressed

# CLOSE_KERNEL_SIZE: bridges the two detected edges of the tape into one solid blob
# Tape ~1cm wide = ~20px. Kernel must be >= tape width to fill the gap between edges.
# Too small → double white line in binary, centroid falls on edge not center
# Too large → merges tape with nearby shadows or objects
# 1cm tape = 20px, kernel 41 = 2cm bridges both edges cleanly
CLOSE_KERNEL_SIZE = 41     # ✅ tuned — solid single blob on tape

# MIN_CONTOUR_AREA: minimum blob size to be considered a line segment (in 320x180 pixels²)
# Must be set ABOVE GREEN_MIN_AREA so green squares are never mistaken for line
# Short line section ~1cm × 2cm = 20px × 40px = 800px²
# Full line across frame ~1cm × 9cm = 20px × 180px = 3600px²
# Noise specks << 400px²
MIN_CONTOUR_AREA = 1500    # ✅ above GREEN_MIN_AREA (1200px²)

# ================== LOOKAHEAD CONFIG ==================
# The robot steers toward a BLEND of two error signals:
#   error_centroid = center of mass of the whole contour blob
#                    → stable, but LAGS on tight curves (sees where line WAS)
#   error_top      = x position of the topmost point of the contour
#                    → where the line is HEADING (lookahead)
#                    → reacts earlier on curves, less stable on noise
#
# LOOKAHEAD_WEIGHT controls the blend:
#   0.0 = pure centroid (original behaviour, no lookahead)
#   0.5 = equal blend
#   0.7 = lean toward lookahead (recommended starting point for curves)
#   1.0 = pure topmost point (aggressive, may be noisy)
#
# RoboCup field has ~60% curves — lookahead helps significantly
# Tune this on actual tiles: increase if robot overshoots curves,
# decrease if robot oscillates on straight sections
LOOKAHEAD_WEIGHT = 0.7     # ← tune this (0.0=centroid only, 1.0=lookahead only)

# ================== GAP BRIDGING CONFIG ==================
# When the line disappears (gap in tape), hold the last known error
# rather than sending error=0 which causes the robot to go straight
# and potentially drive off the course.
#
# GAP_MAX_FRAMES: how many consecutive frames without a line before giving up
# At 20fps: 10 frames = 0.5 sec. At 20cm/s robot speed: 0.5s × 20cm/s = 10cm travel
# RoboCup gaps are max 20cm. At 20cm/s robot speed, gap takes ~1 sec = ~20 frames.
# Set GAP_MAX_FRAMES slightly above expected gap crossing time.
GAP_MAX_FRAMES = 20        # ← tune based on robot speed and gap length

# gap_flag values sent to Cytron in serial packet:
#   0 = line detected normally → full PID steering
#   1 = gap bridging → Cytron holds speed, trusts held error from Pi
#   2 = line truly lost → Cytron slows down and searches
GAP_FLAG_NORMAL   = 0
GAP_FLAG_BRIDGING = 1
GAP_FLAG_LOST     = 2

# ================== SPEED-DEPENDENT TRIGGER CONFIG ==================
# Camera sees 9cm ahead. At 1cm = 20px:
#   distance_px ÷ 20 = cm ahead of robot
#
# GREEN_SLOW_DIST_PX: distance at which robot starts slowing for intersection
# GREEN_EXECUTE_DIST_PX: distance at which Cytron executes the turn
#
# Set ROBOT_SPEED_CMS to match current Cytron speed setting.
# Trigger distances auto-scale with speed:
#   faster robot → needs to see green earlier → larger trigger distances
#
# Formula rationale:
#   slow:    need ~(speed/10 × 2)cm + 3cm safety = px conversion
#   execute: need ~(speed/10 × 1)cm + 1cm safety = px conversion
ROBOT_SPEED_CMS = 20       # ← set this to match actual robot speed (cm/s)

GREEN_SLOW_DIST_PX    = int((ROBOT_SPEED_CMS / 10) * 40 + 60)
GREEN_EXECUTE_DIST_PX = int((ROBOT_SPEED_CMS / 10) * 20 + 20)
# At 20cm/s: SLOW=140px (7cm ahead), EXECUTE=60px (3cm ahead)
# At 30cm/s: SLOW=180px (9cm ahead), EXECUTE=80px (4cm ahead)
# At 10cm/s: SLOW=100px (5cm ahead), EXECUTE=40px (2cm ahead)

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

# Serial protocol: V:{error},{green_code},{blob_count},{distance_px},{gap_flag}
#   error         : int, negative=line left, positive=line right of center
#                   range: -160 to +160 (half of 320px frame width)
#   green_code    : 0=none, 1=turn left, 2=turn right, 3=U-turn
#   blob_count    : raw green blob count (1-4), diagnostic only
#   distance_px   : avg pixels from active Dn blob centroid(s) to bottom of frame
#                   0=no Dn blob yet, >0=actionable
#                   ÷20 = cm ahead of robot
#                   Cytron: slow at GREEN_SLOW_DIST_PX, execute at GREEN_EXECUTE_DIST_PX
#   gap_flag      : 0=line normal, 1=gap bridging (hold speed), 2=line lost (slow+search)
try:
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
except:
    ser = None

# ================================================================
# MAIN LOOP — persistent state variables
# These survive across frames (unlike local variables inside the loop)
# ================================================================

frame_count      = 0
last_time        = time.time()
last_debug_frame = -1

# Gap bridging state — persists across frames
last_good_error = 0        # last error when line was visible
gap_frame_count = 0        # how many consecutive frames without a line

while True:

    t_start = time.perf_counter()

    # ================================================================
    # BLOCK 4: FRAME CAPTURE
    # Future: move to vision/camera.py → camera.capture()
    # ================================================================

    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Scale up for display — all drawing happens on this larger image
    # Detection always uses the original 320×180 frame
    display = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)

    frame_count += 1

    t_capture = time.perf_counter()

    # ================================================================
    # BLOCK 5: GREEN SQUARE DETECTION
    # Done FIRST so green blob positions are available for Plan B
    # centroid filtering in Block 8
    # Future: move to vision/green_detector.py → green.detect(frame)
    # ================================================================

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)

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

    # Build green exclusion boxes for Plan B centroid filtering (Block 8)
    # Each box is (x1, y1, x2, y2) with GREEN_EXCLUSION_PAD border added
    # These boxes define regions where contour points are NOT used for centroid
    green_boxes = []
    for blob in green_blobs:
        gx, gy, gw, gh = cv2.boundingRect(blob["contour"])
        green_boxes.append((
            gx - GREEN_EXCLUSION_PAD,
            gy - GREEN_EXCLUSION_PAD,
            gx + gw + GREEN_EXCLUSION_PAD,
            gy + gh + GREEN_EXCLUSION_PAD
        ))

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
            # e.g. RDn + LUp → RIGHT turn (not U-turn)
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
    # Divide by 20 to convert to cm ahead
    distance_px = 0
    active_dn_blobs = [b for b in decision_blobs if b["vside"] == "Dn"]
    if active_dn_blobs:
        avg_y = sum(b["cy"] for b in active_dn_blobs) / len(active_dn_blobs)
        distance_px = int(PROCESS_HEIGHT - avg_y)

    t_green_class = time.perf_counter()

    # ================================================================
    # BLOCK 7: LINE PREPROCESSING
    # Future: move to vision/line_detector.py → line.preprocess(frame)
    #
    # PIPELINE EXPLANATION — why this order matters:
    #
    #   Step 1: threshold → finds ALL dark areas including tape
    #   Step 2: NO green subtraction (Plan B — see Block 8)
    #           Previous approach subtracted green_mask from binary here,
    #           which created holes in tape where green overlapped tape edge.
    #           Plan B instead filters green zones from centroid math only,
    #           leaving binary intact so CLOSE can work on the full tape shape.
    #   Step 3: MORPH_CLOSE → fills gap between tape edges into one solid blob
    #
    #   ┌──────────────────────────────────────────────────────────┐
    #   │  threshold  →  (no subtraction)  →  CLOSE  →  contours  │
    #   │  (find dark)                       (merge edges)         │
    #   │                                                          │
    #   │  Green zones excluded LATER at centroid math (Block 8)  │
    #   └──────────────────────────────────────────────────────────┘
    # ================================================================

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0) if BLUR_SIZE >= 3 else gray
    blurred = gray  # Skip blurring for performance; can be re-enabled if needed.

    # Step 1: Adaptive threshold — converts grayscale to black/white
    # Detects the tape edges as two white lines in binary
    # block=21 and constant=16 tuned to suppress floor shadows
    #   Block size (arg 5): must be odd. Small(7-11)=sensitive/noisy. Large(21-51)=smoother.
    #   Constant  (arg 6):  Low(4-8)=picks up faint edges. High(12-18)=only strong dark areas.
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,
                                   THRESH_BLOCK_SIZE, THRESH_CONSTANT)

    # Save binary BEFORE close for debug window — shows raw threshold output
    # Compare with AFTER to verify CLOSE kernel is merging tape edges correctly
    binary_before_close = binary.copy()

    # Step 2: NO green subtraction from binary
    # Green squares are excluded later at centroid calculation (Plan B in Block 8)
    # This avoids holes in tape when green overlaps the black line edge
    # Reason for change: binary subtraction created artifact black squares
    # mirrored from green square position, especially when green was partially
    # detected or on HSV boundary causing irregular mask shapes

    # Step 3: MORPH_CLOSE fills the gap between the two tape edge lines → one solid blob
    # Also repairs any minor gaps in the tape from noise
    # Kernel must be >= tape width (20px) to bridge both edges
    # At kernel 41: bridges up to ~2cm gap, cleanly merges both edges
    # MUST come after threshold — no green subtraction to interfere
    close_kernel = np.ones((CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    t_line_pre = time.perf_counter()

    # ================================================================
    # BLOCK 8: LINE CONTOUR DETECTION + PLAN B GREEN FILTERING
    #          + LOOKAHEAD + GAP BRIDGING
    #
    # PLAN B — GREEN EXCLUSION AT CENTROID LEVEL:
    #   Instead of subtracting green from binary (which damaged the tape shape),
    #   we find the contour normally from the full binary, then filter out
    #   contour points that fall inside green square bounding boxes BEFORE
    #   computing moments/centroid.
    #
    #   Benefits over binary subtraction:
    #   ✓ Binary stays intact → CLOSE kernel always gets full tape shape
    #   ✓ No holes punched in tape → no artifacts near green square edges
    #   ✓ Centroid reflects tape center only, green square ignored
    #   ✓ Works correctly even when green partially overlaps tape
    #   ✓ No flicker — each frame processed independently, no temporal state
    #
    #   How it works:
    #   1. Find contour normally from full binary (includes green area if any)
    #   2. Get all contour points as numpy array
    #   3. Build boolean mask: True = point outside ALL green bounding boxes
    #   4. Compute moments only from the surviving (outside) points
    #   5. Centroid and topmost point both use filtered points
    #
    # Full frame used — ROI_TOP rectangle is visual reference only
    # Future: move to vision/line_detector.py → line.detect(binary)
    # ================================================================

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Pre-filter by area immediately — avoids sorting/scoring hundreds of tiny contours
    # MIN_CONTOUR_AREA tuning:
    #   Too low  → slow, noise contours still processed
    #   Too high → may miss thin line sections
    contours = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]

    # Default values — overwritten below if contour found
    error        = 0
    cx           = PROCESS_WIDTH // 2
    main_contour = None

    # Lookahead display points — set when contour found, used in drawing block
    cx_centroid  = PROCESS_WIDTH // 2   # center of mass x (where line IS)
    cx_top       = PROCESS_WIDTH // 2   # topmost point x (where line is HEADING)
    topmost_y    = 0                    # y coordinate of topmost point (for display)

    # Plan B debug — filtered points count (shown on display)
    filtered_pts       = None           # contour points after green exclusion
    pts_total          = 0              # total contour points before filtering
    pts_after_filter   = 0             # points remaining after green exclusion

    # gap_flag default — updated below based on detection result
    gap_flag = GAP_FLAG_NORMAL

    if contours:
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        best_score = -1
        for cnt in sorted_contours[:10]:
            x, y, w, h = cv2.boundingRect(cnt)

            # Only accept contours whose bounding box reaches into the action zone
            # Contours entirely above ROI_TOP are approaching lines — visible but
            # not yet used for steering (drawn in light blue for reference)
            if (y + h) < ROI_TOP:
                continue

            aspect = h / w if w > 0 else 0
            score = aspect * cv2.contourArea(cnt)
            if score > best_score:
                best_score = score
                main_contour = cnt

        if main_contour is not None:

            # ── PLAN B: FILTER CONTOUR POINTS — exclude green zones ──────
            # Extract all contour points as flat (N, 2) numpy array
            # main_contour shape is (N, 1, 2) → squeeze to (N, 2)
            pts = main_contour[:, 0, :]        # shape (N, 2): each row = [x, y]
            pts_total = len(pts)

            if green_boxes:
                # Build boolean array: True = point is OUTSIDE all green boxes
                # Start with all True (all outside), then mark inside points False
                outside = np.ones(len(pts), dtype=bool)

                for (x1, y1, x2, y2) in green_boxes:
                    # Points inside this green box: x in [x1,x2] AND y in [y1,y2]
                    inside_box = ((pts[:, 0] >= x1) & (pts[:, 0] <= x2) &
                                  (pts[:, 1] >= y1) & (pts[:, 1] <= y2))
                    # Exclude these points (AND with NOT inside)
                    outside &= ~inside_box

                filtered_pts     = pts[outside]     # points outside all green boxes
                pts_after_filter = len(filtered_pts)

                # Need at least 5 points for moments to be meaningful
                # If too few survive, green square covers most of tape —
                # fall back to full contour (better than wrong centroid)
                if pts_after_filter >= 5:
                    contour_for_centroid = filtered_pts.reshape(-1, 1, 2)
                else:
                    # Fallback: green covers most of tape, use full contour
                    # This is rare but prevents crash on edge case
                    contour_for_centroid = main_contour
                    filtered_pts = pts   # show all points in debug
            else:
                # No green blobs — use full contour unchanged
                contour_for_centroid = main_contour
                filtered_pts         = pts
                pts_after_filter     = pts_total

            # ── CENTROID ERROR (where the line IS) ──────────────────────
            # Moments give the center of mass of the contour blob
            # Uses green-filtered contour → green square pixels excluded
            # On a curve this LAGS behind the exit direction
            M = cv2.moments(contour_for_centroid)
            if M["m00"] > 100:
                cx_centroid    = int(M["m10"] / M["m00"])
                error_centroid = cx_centroid - CENTER_X
            else:
                cx_centroid    = CENTER_X
                error_centroid = 0

            # ── TOPMOST POINT ERROR (where the line is HEADING) ─────────
            # The topmost point of the filtered contour = where the line
            # exits the top of the camera view = furthest ahead point visible
            # Also filtered to exclude green zones so lookahead target
            # isn't pulled toward the top edge of a green square
            #
            # Example on a right curve:
            #   centroid says: error = +30  (line slightly right)
            #   topmost  says: error = +90  (line heading far right)
            #   blend 0.7:     error = +72  → robot steers harder right earlier
            if len(contour_for_centroid) > 0:
                topmost_idx = contour_for_centroid[:, :, 1].argmin()
                topmost_pt  = tuple(contour_for_centroid[topmost_idx][0])
            else:
                topmost_pt  = (CENTER_X, ROI_TOP)
            cx_top    = topmost_pt[0]
            topmost_y = topmost_pt[1]
            error_top = cx_top - CENTER_X

            # ── BLENDED ERROR (sent to Cytron) ───────────────────────────
            # LOOKAHEAD_WEIGHT = 0.0 → pure centroid (original behaviour)
            # LOOKAHEAD_WEIGHT = 0.7 → 30% centroid + 70% lookahead (recommended)
            # LOOKAHEAD_WEIGHT = 1.0 → pure topmost point
            error = int((1.0 - LOOKAHEAD_WEIGHT) * error_centroid +
                              LOOKAHEAD_WEIGHT    * error_top)

            # Update cx for the display dot (always show centroid position)
            cx = cx_centroid

            # Line is visible — reset gap state
            last_good_error = error
            gap_frame_count = 0
            gap_flag        = GAP_FLAG_NORMAL

            # ── DRAW MAIN CONTOUR (semi-transparent red fill + green border) ─
            # Full contour drawn (not filtered) so you can see the complete
            # tape shape including any green square overlap area
            # Lighter fill (0.25 alpha) so underlying frame is still readable
            overlay = display.copy()
            scaled = [[int(p[0][0] * DISPLAY_SCALE), int(p[0][1] * DISPLAY_SCALE)]
                      for p in main_contour]
            cv2.drawContours(overlay, [np.array(scaled)], -1, COLOR_RED, -1)
            cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)  # 25% red fill
            cv2.drawContours(display, [np.array(scaled)], -1, COLOR_GREEN, 4)

            # ── DRAW FILTERED POINTS (Plan B debug visualization) ────────
            # CYAN polyline = contour points that actually voted for centroid
            # Disappears where tape runs through/near green square bounding box
            # If cyan line looks correct (follows tape away from green),
            # Plan B is working as intended
            if filtered_pts is not None and len(filtered_pts) >= 2:
                scaled_filtered = np.array(
                    [[int(p[0] * DISPLAY_SCALE), int(p[1] * DISPLAY_SCALE)]
                     for p in filtered_pts], dtype=np.int32)
                cv2.polylines(display, [scaled_filtered], False, COLOR_CYAN, 2)
                # CYAN = points used for centroid
                # Gap in cyan = green square exclusion zone working

            # Draw secondary contours for debugging
            for cnt in sorted_contours[:5]:
                if cnt is not main_contour and cv2.contourArea(cnt) > 100:
                    scaled_other = [[int(p[0][0] * DISPLAY_SCALE),
                                     int(p[0][1] * DISPLAY_SCALE)] for p in cnt]
                    _x, _y, _w, _h = cv2.boundingRect(cnt)
                    if (_y + _h) < ROI_TOP:
                        # Approaching lines above action zone — light blue
                        cv2.drawContours(display, [np.array(scaled_other)],
                                         -1, COLOR_LIGHT_BLUE, 2)
                    else:
                        # Ignored secondary lines inside action zone — dark green
                        cv2.drawContours(display, [np.array(scaled_other)],
                                         -1, COLOR_DARK_GREEN, 2)

    else:
        # ── NO CONTOUR FOUND — GAP BRIDGING ─────────────────────────────
        # Line disappeared: could be a tape gap (max 20cm) or truly lost
        gap_frame_count += 1

        if gap_frame_count < GAP_MAX_FRAMES:
            # Still within expected gap length → hold last known error
            # Robot keeps driving with last steering, trusting it will
            # reacquire the line after the gap
            error    = last_good_error
            gap_flag = GAP_FLAG_BRIDGING
        else:
            # Exceeded max gap frames → line is truly lost
            # Send error=0 (go straight) and tell Cytron to slow down
            error    = 0
            gap_flag = GAP_FLAG_LOST

    t_line_contour = time.perf_counter()

    # ================================================================
    # BLOCK 9: SERIAL OUTPUT
    # Future: move to comms/serial_link.py → serial.send(...)
    # ================================================================

    # Protocol: V:{error},{green_code},{blob_count},{distance_px},{gap_flag}
    #   error         : blended lookahead+centroid, negative=left, positive=right
    #   green_code    : 0=none, 1=left, 2=right, 3=U-turn
    #   blob_count    : raw green blob count (1-4), diagnostic
    #   distance_px   : avg pixels from active Dn blob to bottom (÷20 = cm ahead)
    #                   Cytron slow threshold : GREEN_SLOW_DIST_PX
    #                   Cytron execute threshold: GREEN_EXECUTE_DIST_PX
    #   gap_flag      : 0=normal, 1=gap bridging (hold speed), 2=lost (slow+search)
    if ser:
        ser.write(f"V:{error},{green_code},{len(green_blobs)},{distance_px},{gap_flag}\n".encode())

    # ================================================================
    # BLOCK 10: DISPLAY — GREEN BLOBS
    # Future: move to display/debug_view.py → view.draw_green(...)
    # ================================================================

    for blob in green_blobs:
        cnt = blob["contour"]
        scaled_green = [[int(p[0][0] * DISPLAY_SCALE), int(p[0][1] * DISPLAY_SCALE)]
                        for p in cnt]

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

        # Draw green exclusion box on display for Plan B debug
        # Shows exactly which area is excluded from centroid calculation
        # Drawn in magenta dashed-style (thin) so it doesn't obscure other info
        for (x1, y1, x2, y2) in green_boxes:
            ex1 = int(x1 * DISPLAY_SCALE)
            ey1 = int(y1 * DISPLAY_SCALE)
            ex2 = int(x2 * DISPLAY_SCALE)
            ey2 = int(y2 * DISPLAY_SCALE)
            cv2.rectangle(display, (ex1, ey1), (ex2, ey2), COLOR_MAGENTA, 1)
            # MAGENTA rectangle = exclusion zone boundary
            # Points inside this box are NOT used for centroid

    # ================================================================
    # BLOCK 11: DISPLAY — OVERLAYS (rulers, ROI, zones, status, packet)
    # Future: move to display/debug_view.py → view.draw_overlays(...)
    # ================================================================

    scaled_w   = int(PROCESS_WIDTH * DISPLAY_SCALE)
    scaled_h   = int(PROCESS_HEIGHT * DISPLAY_SCALE)
    center_x   = scaled_w // 2
    detected_x = int(cx * DISPLAY_SCALE)
    detected_y = int((ROI_TOP + (ROI_BOTTOM - ROI_TOP) // 2) * DISPLAY_SCALE)

    # ── Lookahead visualization ──────────────────────────────────────
    # Only draw when contour was found (main_contour is not None)
    if main_contour is not None:
        # Centroid dot — where the line center of mass IS (magenta hollow circle)
        # Computed from green-filtered contour points
        centroid_disp_x = int(cx_centroid * DISPLAY_SCALE)
        centroid_disp_y = detected_y
        cv2.circle(display, (centroid_disp_x, centroid_disp_y),
                   10, COLOR_MAGENTA, 2)   # hollow = centroid position

        # Topmost point dot — where the line is HEADING (white filled circle)
        # This is the lookahead point — furthest visible point on the line ahead
        # Also computed from filtered points — excludes green square top edges
        top_disp_x = int(cx_top    * DISPLAY_SCALE)
        top_disp_y = int(topmost_y * DISPLAY_SCALE)
        cv2.circle(display, (top_disp_x, top_disp_y),
                   10, COLOR_WHITE, -1)    # filled = lookahead target

        # Line connecting centroid to topmost point — visualises lookahead direction
        # If this line leans left/right the robot will steer that way early
        cv2.line(display,
                 (centroid_disp_x, centroid_disp_y),
                 (top_disp_x, top_disp_y),
                 COLOR_WHITE, 1)

        # Labels: C = centroid, T = topmost (lookahead target)
        cv2.putText(display, "C", (centroid_disp_x + 12, centroid_disp_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_MAGENTA, 1)
        cv2.putText(display, "T", (top_disp_x + 12, top_disp_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)

    # ── Vertical Ruler - Left Edge (every 20px = 1cm in original resolution) ──
    for i in range(0, PROCESS_HEIGHT + 1, 20):
        y = int(i * DISPLAY_SCALE)
        tick_len = 25 if i % 100 == 0 else 12
        cv2.line(display, (0, y), (tick_len, y), COLOR_BLUE, 2)
        if i % 100 == 0:
            cv2.putText(display, str(i), (tick_len + 8, y + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_BLUE, 2)

    # ── Horizontal Ruler - Bottom Edge ──────────────────────────────
    for i in range(0, PROCESS_WIDTH + 1, 20):
        x = int(i * DISPLAY_SCALE)
        tick_len = 25 if i % 100 == 0 else 12
        cv2.line(display, (x, scaled_h), (x, scaled_h - tick_len), COLOR_BLUE, 2)
        if i % 100 == 0:
            cv2.putText(display, str(i), (x - 20, scaled_h - tick_len - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_BLUE, 2)

    # ── ROI action zone box — visual reference only (blue rectangle) ─
    cv2.rectangle(display, (0, int(ROI_TOP * DISPLAY_SCALE)),
                  (scaled_w, int(ROI_BOTTOM * DISPLAY_SCALE)), COLOR_BLUE, 3)

    # ── Green Up/Dn split line (cyan) ───────────────────────────────
    split_y_scaled = int(GREEN_SPLIT_Y * DISPLAY_SCALE)
    cv2.line(display, (0, split_y_scaled), (scaled_w, split_y_scaled), COLOR_CYAN, 2)

    # Zone corner labels
    cv2.putText(display, "LUp", (5,             split_y_scaled - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CYAN, 2)
    cv2.putText(display, "RUp", (scaled_w - 50, split_y_scaled - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CYAN, 2)
    cv2.putText(display, "LDn", (5,             split_y_scaled + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CYAN, 2)
    cv2.putText(display, "RDn", (scaled_w - 50, split_y_scaled + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CYAN, 2)

    # ── Green trigger distance lines ─────────────────────────────────
    # SLOW line (yellow): start decelerating when green blob reaches this Y
    # EXECUTE line (red): execute the turn when green blob reaches this Y
    # Both auto-calculated from ROBOT_SPEED_CMS in config
    slow_y    = int((PROCESS_HEIGHT - GREEN_SLOW_DIST_PX)    * DISPLAY_SCALE)
    execute_y = int((PROCESS_HEIGHT - GREEN_EXECUTE_DIST_PX) * DISPLAY_SCALE)

    # Clamp to frame bounds in case speed config pushes values off screen
    slow_y    = max(0, min(scaled_h, slow_y))
    execute_y = max(0, min(scaled_h, execute_y))

    cv2.line(display, (0, slow_y), (scaled_w, slow_y), COLOR_YELLOW, 1)
    cv2.putText(display, f"SLOW {GREEN_SLOW_DIST_PX}px ({GREEN_SLOW_DIST_PX//20}cm)",
                (scaled_w - 200, slow_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_YELLOW, 1)

    cv2.line(display, (0, execute_y), (scaled_w, execute_y), COLOR_RED, 1)
    cv2.putText(display, f"GO {GREEN_EXECUTE_DIST_PX}px ({GREEN_EXECUTE_DIST_PX//20}cm)",
                (scaled_w - 180, execute_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_RED, 1)

    # ── Center vertical line + blended error dot + error line ────────
    cv2.line(display, (center_x, int(ROI_TOP * DISPLAY_SCALE)),
             (center_x, int(ROI_BOTTOM * DISPLAY_SCALE)), COLOR_YELLOW, 2)
    cv2.circle(display, (detected_x, detected_y), 12, COLOR_RED, -1)
    cv2.line(display, (center_x, detected_y), (detected_x, detected_y), COLOR_YELLOW, 3)

    # ── Status text ──────────────────────────────────────────────────
    green_labels = {0: "NONE",
                    1: "LEFT",
                    2: "RIGHT",
                    3: "U-TURN"}
    green_label_colors = {0: COLOR_GRAY,
                          1: COLOR_DARK_GREEN,
                          2: COLOR_DARK_GREEN,
                          3: COLOR_ORANGE}
    gap_labels = {GAP_FLAG_NORMAL:   "",
                  GAP_FLAG_BRIDGING: " GAP-BRIDGE",
                  GAP_FLAG_LOST:     " LINE-LOST"}
    zones_seen = [b["zone"] for b in green_blobs]

    # Error color changes to warn about gap state:
    #   RED     = normal line following
    #   MAGENTA = gap bridging (holding last error, line temporarily missing)
    #   PURPLE  = line truly lost (sending error=0, Cytron should slow+search)
    error_color = COLOR_RED
    if gap_flag == GAP_FLAG_BRIDGING:
        error_color = COLOR_MAGENTA
    elif gap_flag == GAP_FLAG_LOST:
        error_color = COLOR_PURPLE

    cv2.putText(display, f"Error: {error}{gap_labels[gap_flag]}",
                (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, error_color, 2)
    cv2.putText(display,
                f"Green: {green_code} {len(green_blobs)} {green_labels[green_code]} {zones_seen}",
                (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                green_label_colors[green_code], 2)
    if green_blobs:
        cv2.putText(display, f"Dist: {distance_px}px ({distance_px//20}cm)",
                    (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ORANGE, 2)

    # Lookahead blend info — shows centroid error, topmost error, weight, result
    # Use this to tune LOOKAHEAD_WEIGHT: watch C and T diverge on curves
    if main_contour is not None:
        cv2.putText(display,
                    f"C:{cx_centroid - CENTER_X:+d} T:{cx_top - CENTER_X:+d} "
                    f"W:{LOOKAHEAD_WEIGHT:.1f} →{error:+d}",
                    (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

    # Plan B filter stats — shows how many contour points survived green exclusion
    # pts_total=N pts_after_filter=M → M points used for centroid out of N total
    # If M is much smaller than N near green square → exclusion is working
    if main_contour is not None and green_boxes:
        cv2.putText(display,
                    f"PlanB: {pts_after_filter}/{pts_total}pts used",
                    (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_CYAN, 1)

    # ── Raw packet display — top right corner (BEFORE imshow) ────────
    packet_str = f"V:{error},{green_code},{len(green_blobs)},{distance_px},{gap_flag}"
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
    #
    # TWO binary debug windows to diagnose threshold and CLOSE kernel:
    #   "Debug - Binary BEFORE" : raw threshold output before CLOSE
    #                             Shows two white edge lines for the tape
    #                             Use this to check threshold parameters
    #   "Debug - Binary AFTER"  : after CLOSE kernel applied
    #                             Should show ONE solid white blob per tape section
    #                             Green square may appear here but is handled
    #                             by Plan B at centroid level, not here
    #   "Debug - Green Mask"    : shows which pixels are classified as green
    #                             Use this to check green detection range
    #
    # Note: no green subtraction in binary anymore (Plan B)
    # So Binary AFTER may show the green square area as white — this is normal
    # The green area is excluded from centroid math in Block 8, not here
    # Future: move to display/debug_view.py → view.draw_debug_windows(...)
    # ================================================================

    if DEBUG_SHOW_MASKS:
        if frame_count - last_debug_frame >= DEBUG_REFRESH_FRAMES:
            last_debug_frame = frame_count

            # Binary BEFORE CLOSE — shows raw threshold result
            # Should show two thin white lines (tape edges)
            before_display = cv2.resize(binary_before_close, None,
                                        fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            cv2.putText(before_display,
                        f"BINARY BEFORE CLOSE (block={THRESH_BLOCK_SIZE} c={THRESH_CONSTANT})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
            cv2.putText(before_display, "expect: two thin white edge lines",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            # Binary AFTER CLOSE — what findContours processes
            # Should show one solid white blob per tape section
            # Green area may be white here — handled by Plan B not here
            after_display = cv2.resize(binary, None,
                                       fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            cv2.putText(after_display,
                        f"BINARY AFTER CLOSE (kernel={CLOSE_KERNEL_SIZE}px)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
            cv2.putText(after_display,
                        "expect: solid white blob on tape (green ok here - Plan B)",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            # Green mask — shows which pixels are detected as green
            green_display = cv2.resize(green_mask, None,
                                       fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            cv2.putText(green_display, "GREEN MASK",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
            cv2.putText(green_display,
                        f"H:{LOWER_GREEN[0]}-{UPPER_GREEN[0]} "
                        f"S:{LOWER_GREEN[1]}-{UPPER_GREEN[1]} "
                        f"V:{LOWER_GREEN[2]}-{UPPER_GREEN[2]}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            cv2.imshow("Debug - Binary BEFORE", before_display)
            cv2.imshow("Debug - Binary AFTER",  after_display)
            cv2.imshow("Debug - Green Mask",    green_display)
    else:
        cv2.destroyWindow("Debug - Binary BEFORE")
        cv2.destroyWindow("Debug - Binary AFTER")
        cv2.destroyWindow("Debug - Green Mask")

    t_debug = time.perf_counter()

    # ================================================================
    # BLOCK 14: TIMING + FPS PRINT + QUIT
    # ================================================================

    # Fix 2: single waitKey call — handles both display refresh and quit detection
    # Previously called twice per loop which doubled X11 flush cost (~18ms wasted)
    # Press 'f' to freeze frame for inspection, any key to unfreeze
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        # Freeze on current frame — useful to catch bad frames mid-run
        # Prints Plan B stats so you can inspect what happened
        print(f"FROZEN | pts_total={pts_total} pts_after_filter={pts_after_filter} "
              f"green_boxes={len(green_boxes)} gap_flag={gap_flag} "
              f"error={error} cx_centroid={cx_centroid} cx_top={cx_top}")
        cv2.waitKey(0)   # wait until any key pressed to unfreeze

    t_end = time.perf_counter()

    if frame_count % 20 == 0:
        print(f"FPS: {20 / (time.time() - last_time):.1f} | Error: {error:+4d} | "
              f"Gap: {gap_labels[gap_flag].strip() or 'OK'} ({gap_frame_count}f) | "
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