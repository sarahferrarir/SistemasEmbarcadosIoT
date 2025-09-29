import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Screen resolution (adjust accordingly)
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# Webcam reference resolution
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# OpenCV video capture
cap = cv2.VideoCapture(0)
cap.set(3, CAMERA_WIDTH)
cap.set(4, CAMERA_HEIGHT)

# Define eye landmark indices for iris tracking
LEFT_IRIS = [474, 475, 476, 477]  # Iris landmarks for left eye
RIGHT_IRIS = [469, 470, 471, 472]  # Iris landmarks for right eye
LEFT_EYE = [362, 385, 387, 263]  # Outer and inner corners of left eye
RIGHT_EYE = [33, 160, 158, 133]  # Outer and inner corners of right eye


def get_iris_position(landmarks, iris_points, eye_points):
    """Calculate the relative position of the iris inside the eye."""
    iris_x = np.mean([landmarks[i].x for i in iris_points])
    iris_y = np.mean([landmarks[i].y for i in iris_points])
    eye_x = np.mean([landmarks[i].x for i in eye_points])
    eye_y = np.mean([landmarks[i].y for i in eye_points])
    return np.array([(iris_x - eye_x) * 5, (iris_y - eye_y) * 5])  # Amplify sensitivity

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)  # Flip horizontally
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            
            left_iris_pos = get_iris_position(landmarks, LEFT_IRIS, LEFT_EYE)
            right_iris_pos = get_iris_position(landmarks, RIGHT_IRIS, RIGHT_EYE)
            gaze_vector = (left_iris_pos + right_iris_pos) / 2  # Average both eyes
            
            # Map gaze direction to screen coordinates
            ball_x = int((gaze_vector[0] + 0.5) * SCREEN_WIDTH)
            ball_y = int((gaze_vector[1] + 0.5) * SCREEN_HEIGHT)
            
            # Move the cursor based on highly sensitive eye movement
            pyautogui.moveTo(ball_x, ball_y, duration=0.01)  # Faster and more sensitive movement
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()