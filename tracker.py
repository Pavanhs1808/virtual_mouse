import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Constants
SMOOTHING_FACTOR = 0.7
SENSITIVITY = 2.5
SCREEN_MARGIN = 10
BLINK_THRESHOLD = 0.23
DOUBLE_BLINK_TIME = 0.35
LONG_BLINK_THRESHOLD = 1.0
BLINK_COOLDOWN = 0.3
BLINK_STABILIZE_FRAMES = 3
BLINK_POSITION_THRESHOLD = 10

# Eye landmarks
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Initialize variables
last_blink_time = 0
blink_count = 0
last_click_time = 0
is_blinking = False
blink_start_time = 0
last_stable_x = 0
last_stable_y = 0
prev_x, prev_y = 0, 0
blink_positions = []

def calculate_ear(landmarks, eye_points):
    points = np.array([[landmarks[point].x, landmarks[point].y] for point in eye_points])
    vertical_dist1 = np.linalg.norm(points[1] - points[5])
    vertical_dist2 = np.linalg.norm(points[2] - points[4])
    horizontal_dist = np.linalg.norm(points[0] - points[3])
    ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
    return ear

def smooth_movement(current, previous, smoothing_factor):
    return previous * smoothing_factor + current * (1 - smoothing_factor)

# Initialize camera and screen
cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = False

print("Starting eye tracking. Press 'q' to quit.")

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        
        # Calculate EAR
        left_ear = calculate_ear(face_landmarks.landmark, LEFT_EYE)
        right_ear = calculate_ear(face_landmarks.landmark, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Get iris positions
        left_iris = face_landmarks.landmark[468]
        right_iris = face_landmarks.landmark[473]
        iris_x = (left_iris.x + right_iris.x) / 2
        iris_y = (left_iris.y + right_iris.y) / 2
        
        # Calculate cursor position
        x_ratio = iris_x
        y_ratio = iris_y
        
        # Apply non-linear mapping
        x_offset = (x_ratio - 0.5) * SENSITIVITY
        y_offset = (y_ratio - 0.5) * SENSITIVITY
        
        # Calculate screen position
        screen_x = (0.5 + x_offset) * screen_w
        screen_y = (0.5 + y_offset) * screen_h
        
        # Apply bounds
        screen_x = max(SCREEN_MARGIN, min(screen_w - SCREEN_MARGIN, screen_x))
        screen_y = max(SCREEN_MARGIN, min(screen_h - SCREEN_MARGIN, screen_y))
        
        # Blink detection with position stabilization
        current_time = time.time()
        if avg_ear < BLINK_THRESHOLD:
            if not is_blinking:
                is_blinking = True
                blink_start_time = current_time
                last_stable_x = screen_x
                last_stable_y = screen_y
            
            # Use last stable position during blink
            screen_x = last_stable_x
            screen_y = last_stable_y
            
            # Long blink for double click
            blink_duration = current_time - blink_start_time
            if blink_duration > LONG_BLINK_THRESHOLD:
                if not last_click_time or current_time - last_click_time > BLINK_COOLDOWN:
                    pyautogui.moveTo(int(last_stable_x), int(last_stable_y), duration=0.01)
                    pyautogui.doubleClick()
                    last_click_time = current_time
                    is_blinking = False
                    blink_count = 0
            elif blink_count == 0:
                last_blink_time = current_time
                blink_count = 1
            elif current_time - last_blink_time < DOUBLE_BLINK_TIME:
                if not last_click_time or current_time - last_click_time > BLINK_COOLDOWN:
                    pyautogui.moveTo(int(last_stable_x), int(last_stable_y), duration=0.01)
                    pyautogui.click()
                    last_click_time = current_time
                blink_count = 0
        else:
            if not is_blinking:
                # Apply smoothing only when not blinking
                screen_x = smooth_movement(screen_x, prev_x, SMOOTHING_FACTOR)
                screen_y = smooth_movement(screen_y, prev_y, SMOOTHING_FACTOR)
                pyautogui.moveTo(int(screen_x), int(screen_y), duration=0.01)
                prev_x, prev_y = screen_x, screen_y
                last_stable_x = screen_x
                last_stable_y = screen_y
            is_blinking = False
            if current_time - last_blink_time > DOUBLE_BLINK_TIME:
                blink_count = 0
        
        # Visualization
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Eye Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()