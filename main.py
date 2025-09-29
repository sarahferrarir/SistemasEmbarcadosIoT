import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import pygetwindow as gw  # For detecting UI windows

# 🛠️ Adjustable parameters
FINGER_DISTANCE_THRESHOLD = 0.05  # Minimum distance for fingers to be considered "together"
VECTOR_MULTIPLIER = 4.0  # Controls how far point C extends
SENSITIVITY = 0.5  # Adjusts how quickly point C moves (0.1 - 1.0 recommended)
DEADZONE = 0.04  # Ignores small movements (prevents jitter)
CLICK_DISTANCE_THRESHOLD = 0.2  # Adjust for click sensitivity (Increase if click never triggers)
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# State tracking variables
previous_click_time = 0
dragging = False
previous_point_c = (0, 0)  # Store previous Point C position to smooth movement


def detect_click_hold(hand_landmarks):
    """Detect click (tap) when Thumb Tip (#4) touches Index PIP (#6) and hold if they remain together."""
    global previous_click_time, dragging

    try:
        thumb_tip = hand_landmarks.landmark[4]  # ✅ Thumb Tip (#4)
        index_pip = hand_landmarks.landmark[6]  # ✅ Index PIP (#6)

        # Calculate Euclidean distance
        distance = np.hypot(thumb_tip.x - index_pip.x, thumb_tip.y - index_pip.y)
        print(f"DEBUG: Distance #4-#6 = {distance:.4f}")  # ✅ Print for debugging

        # ✅ Adjust sensitivity if needed
        CLICK_DISTANCE_THRESHOLD = 0.05  # Increase this if click never triggers

        # Click if the distance is below threshold
        if distance < CLICK_DISTANCE_THRESHOLD:
            current_time = time.time()

            if not dragging:
                pyautogui.mouseDown()  # Click down
                dragging = True
                previous_click_time = current_time
                print("🖱️ Mouse Click & Hold (Drag Started)")

        else:
            if dragging:  # Release click if fingers separate
                pyautogui.mouseUp()
                dragging = False
                print("🖱️ Mouse Released")

    except Exception as e:
        print(f"⚠️ Error in detect_click_hold: {e}")


last_sticky_time = 0  # Track last sticky cursor effect

def is_over_button(cursor_x, cursor_y):
    """Check if the cursor is over a button-like UI element."""
    try:
        active_window = gw.getActiveWindow()
        if not active_window:
            return False  # No active window detected
        
        win_x, win_y, win_width, win_height = active_window.left, active_window.top, active_window.width, active_window.height

        if (win_x + win_width * 0.3 < cursor_x < win_x + win_width * 0.7) and (
            win_y + win_height * 0.3 < cursor_y < win_y + win_height * 0.7):
            return True
        return False
    except Exception as e:
        print(f"⚠️ Error in is_over_button: {e}")
        return False



def move_cursor(smoothed_point_c):
    """Move the mouse cursor based on normalized hand position with 'stickiness' effect for buttons."""
    global last_sticky_time

    try:
        if smoothed_point_c:
            cursor_x = int(smoothed_point_c[0] * SCREEN_WIDTH)
            cursor_y = int(smoothed_point_c[1] * SCREEN_HEIGHT)

            # ✅ Prevent cursor from going out of bounds
            cursor_x = max(0, min(SCREEN_WIDTH - 1, cursor_x))
            cursor_y = max(0, min(SCREEN_HEIGHT - 1, cursor_y))

            # ✅ If hovering over a button, slow down movement slightly
            if is_over_button(cursor_x, cursor_y):
                if time.time() - last_sticky_time > 0.1:  # Stick for a short time
                    pyautogui.moveTo(cursor_x, cursor_y, duration=0.15)  # Slow movement
                    last_sticky_time = time.time()
                    print("📌 Cursor Sticking to Button")
                return  # Stop further movement to keep cursor there briefly

            # ✅ Normal cursor movement
            pyautogui.moveTo(cursor_x, cursor_y, duration=0.05)
            print(f"🖱️ Moving cursor to: {cursor_x}, {cursor_y}")
    except Exception as e:
        print(f"⚠️ Error in move_cursor: {e}")

def calculate_vector(hand_landmarks):
    """Calculate vector from landmark 5 → 12 → C, applying deadzone & sensitivity."""
    global previous_point_c  # Keep track of last position for smoothing

    try:
        lm5 = hand_landmarks.landmark[5]
        lm12 = hand_landmarks.landmark[12]

        # Compute direction and extend for point C
        direction_x = lm12.x - lm5.x
        direction_y = lm12.y - lm5.y
        point_c_x = lm5.x + VECTOR_MULTIPLIER * direction_x
        point_c_y = lm5.y + VECTOR_MULTIPLIER * direction_y

        # Calculate movement difference from last frame
        delta_x = abs(point_c_x - previous_point_c[0])
        delta_y = abs(point_c_y - previous_point_c[1])

        # Apply deadzone (ignore small movements)
        if delta_x < DEADZONE and delta_y < DEADZONE:
            point_c_x, point_c_y = previous_point_c  # Keep the previous position

        # Apply sensitivity (scales movement)
        smoothed_x = previous_point_c[0] + (point_c_x - previous_point_c[0]) * SENSITIVITY
        smoothed_y = previous_point_c[1] + (point_c_y - previous_point_c[1]) * SENSITIVITY

        # Store new position for the next frame
        previous_point_c = (smoothed_x, smoothed_y)

        # Convert to pixel coordinates
        lm5_px = (int(lm5.x * 640), int(lm5.y * 480))
        lm12_px = (int(lm12.x * 640), int(lm12.y * 480))
        point_c_px = (int(smoothed_x * 640), int(smoothed_y * 480))

        print(f"🖱️ Point C (Pixels): {point_c_px} | Smoothed: ({smoothed_x:.2f}, {smoothed_y:.2f})")

        return lm5_px, lm12_px, point_c_px, (smoothed_x, smoothed_y)
    except Exception as e:
        print(f"⚠️ Error in calculate_vector: {e}")
        return None, None, None, None


def fingers_together(hand_landmarks):
    """Check if index and middle fingers are together."""
    try:
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        distance = np.hypot(index_tip.x - middle_tip.x, index_tip.y - middle_tip.y)

        return distance < FINGER_DISTANCE_THRESHOLD  # Uses the adjustable threshold
    except Exception as e:
        print(f"⚠️ Error in fingers_together: {e}")
        return False


def draw_hand_and_vector(frame, hand_landmarks):
    """Draw hand landmarks and move cursor ONLY if fingers are together."""
    try:
        if hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if fingers_together(hand_landmarks):  # ✅ Only move cursor if fingers are together
                lm5, lm12, point_c, smoothed_point_c = calculate_vector(hand_landmarks)
                if lm5 and lm12 and point_c:
                    cv2.line(frame, lm5, lm12, (0, 255, 0), 2)  # Green line (5 → 12)
                    cv2.line(frame, lm12, point_c, (0, 0, 255), 2)  # Red line (12 → C)
                    cv2.circle(frame, point_c, 5, (255, 0, 0), -1)  # Blue dot (Point C)

                    # ✅ Move the cursor only if fingers are together
                    move_cursor(smoothed_point_c)

                # ✅ Detect click & hold only if fingers are together
                detect_click_hold(hand_landmarks)
            else:
                print("✋ Cursor control disabled - Fingers apart")  # Debug message

    except Exception as e:
        print(f"⚠️ Error in draw_hand_and_vector: {e}")


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Detect only one hand
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

def detect_right_hand(frame):
    """Detect the right hand and return its landmarks."""
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0]  # Return first detected hand
        return None
    except Exception as e:
        print(f"⚠️ Error in detect_right_hand: {e}")
        return None


# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Camera not detected!")
        break

    frame = cv2.flip(frame, 1)  # Mirror frame
    hand_landmarks = detect_right_hand(frame)

    if hand_landmarks:
        draw_hand_and_vector(frame, hand_landmarks)

    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()

