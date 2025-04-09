import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.set_page_config(page_title="AI Fitness Trainer", layout="wide")

st.title("🏋️ AI Fitness Trainer - Real-Time Squat Posture Feedback")
st.sidebar.header("How to Use:")
st.sidebar.markdown("""
1. Allow webcam access when prompted.
2. Stand in front of the camera.
3. Perform squats.
4. Get instant feedback on your form!
""")

frame_placeholder = st.empty()
feedback_placeholder = st.empty()

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))

    return angle

run = st.checkbox('Start Camera')

if run:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        feedback = "Waiting for Pose..."
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]

            knee_angle = calculate_angle(hip, knee, ankle)

            if knee_angle > 160:
                feedback = "Start Squat"
            elif knee_angle < 90:
                feedback = "Too Low, Come Up!"
            elif knee_angle < 120:
                feedback = "Perfect Squat!"
            else:
                feedback = "Good Posture"

            cv2.putText(frame, f'Knee Angle: {int(knee_angle)}°', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")
        feedback_placeholder.subheader(f"🧠 Feedback: **{feedback}**")

        if not run:
            break

    cap.release()
