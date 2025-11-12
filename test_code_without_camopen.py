import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import time
import os
from gtts import gTTS
import base64
from io import BytesIO

# Load pre-trained model
model_file = "yoga_pose_model.pkl"
with open(model_file, "rb") as f:
    model = pickle.load(f)

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to calculate angles
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Function to compare angles and determine incorrect parts
def compare_angles(detected_angles, expected_angles, threshold=10):
    incorrect_parts = []
    for part, detected_angle in detected_angles.items():
        expected_angle = expected_angles.get(part, None)
        if expected_angle is not None and abs(detected_angle - expected_angle) > threshold:
            incorrect_parts.append(part)
    return incorrect_parts

# Expected angles for each pose (you need to define these based on your dataset)
expected_angles_dict = {
    "adho mukha svanasana": {"L-elbow": 160, "R-elbow": 160, "L-knee": 175, "R-knee": 175},
    "balasana": {"L-elbow": 90, "R-elbow": 90, "L-knee": 90, "R-knee": 90},
    "garudasana": {"L-elbow": 45, "R-elbow": 45, "L-knee": 60, "R-knee": 60},
    "marjaryasana": {"L-elbow": 170, "R-elbow": 170, "L-knee": 90, "R-knee": 90},
    "parsva bakasana": {"L-elbow": 90, "R-elbow": 90, "L-knee": 45, "R-knee": 45},
    "salabhasana": {"L-elbow": 170, "R-elbow": 170, "L-knee": 180, "R-knee": 180},
    "setu bandha sarvangasana": {"L-elbow": 160, "R-elbow": 160, "L-knee": 90, "R-knee": 90},
    "utthita trikonasana": {"L-elbow": 180, "R-elbow": 180, "L-knee": 160, "R-knee": 160},
    "virabhadrasana ii": {"L-elbow": 180, "R-elbow": 180, "L-knee": 90, "R-knee": 90}
}

# Function to process image and extract angles
def process_image(image, selected_pose):
    """Detect pose and extract joint angles using Mediapipe."""
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            return None, None, "No human detected.", []

        landmarks = results.pose_landmarks.landmark
        angles = {
    "L-wrist": calculate_angle(
        [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_INDEX].x, landmarks[mp_pose.PoseLandmark.LEFT_INDEX].y]
    ),
    "R-wrist": calculate_angle(
        [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].x, landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].y]
    ),
    "L-elbow": calculate_angle(
        [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
    ),
    "R-elbow": calculate_angle(
        [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
    ),
    "L-shoulder": calculate_angle(
        [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
    ),
    "R-shoulder": calculate_angle(
        [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
    ),
    "L-knee": calculate_angle(
        [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
    ),
    "R-knee": calculate_angle(
        [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
    ),
    "L-ankle": calculate_angle(
        [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_HEEL].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y]
    ),
    "R-ankle": calculate_angle(
        [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].x, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y]
    ),
    "L-hip": calculate_angle(
        [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
    ),
    "R-hip": calculate_angle(
        [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
    )
}

        angle_df = pd.DataFrame([angles])
        predicted_pose = model.predict(angle_df)[0]

        # Compare with selected pose
        correct_pose = selected_pose == predicted_pose

        # Get expected angles for the selected pose
        expected_angles = expected_angles_dict.get(selected_pose, {})

        # Compare detected angles with expected angles
        incorrect_parts = compare_angles(angles, expected_angles)

        # Draw skeleton
        skeleton_image = image.copy()
        for landmark in mp_pose.POSE_CONNECTIONS:
            start = results.pose_landmarks.landmark[landmark[0]]
            end = results.pose_landmarks.landmark[landmark[1]]

            start_point = (int(start.x * image.shape[1]), int(start.y * image.shape[0]))
            end_point = (int(end.x * image.shape[1]), int(end.y * image.shape[0]))

            # Determine if this part is incorrect
            part_name = f"{landmark[0]}-{landmark[1]}"
            if part_name in incorrect_parts:
                color = (0, 0, 255)  # Red for incorrect parts
            else:
                color = (0, 255, 0)  # Green for correct parts

            cv2.line(skeleton_image, start_point, end_point, color, 5)
            # Draw the line on the image


            # Display the image
            #cv2.imshow("Skeleton Image", skeleton_image)
            #cv2.waitKey(0)  # Wait for a key press to close the window
            #cv2.destroyAllWindows()  # Close the window

        return angle_df, skeleton_image, predicted_pose, incorrect_parts

# Function to play voice feedback
def play_audio(text):
    tts = gTTS(text=text, lang="en")
    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    audio_b64 = base64.b64encode(audio_buffer.read()).decode()
    audio_html = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>'
    st.markdown(audio_html, unsafe_allow_html=True)

# Streamlit UI
st.title("🧘 Yoga Pose Classification")
st.write("Select a pose, click the webcam button, and the system will classify if your pose is correct.")

# Dropdown for selecting a yoga pose
pose_options = ['adho mukha svanasana', 'balasana', 'garudasana', 'marjaryasana', 
    'parsva bakasana', 'salabhasana', 'setu bandha sarvangasana', 
    'utthita trikonasana', 'virabhadrasana ii']
selected_pose = st.selectbox("Select Your Yoga Pose:", pose_options)

# Webcam button
if st.button("Capture Pose via Webcam"):
    st.write("Get ready! Capturing in 5 seconds...")
    pose = mp_pose.Pose()

    # Start video capture
    cap = cv2.VideoCapture(0)

    # Get the start time
    start_time = time.time()
    last_frame = None  # Variable to store the last frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB (MediaPipe requires RGB format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for pose estimation
        results = pose.process(rgb_frame)

        # Draw the skeleton on the frame if landmarks are detected
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

        # Display the frame
        cv2.imshow('Live Skeleton Detection', frame)

        # Store the last frame
        last_frame = frame.copy()

        # Stop after 10 seconds
        if time.time() - start_time > 10:
            break

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save the last frame as an image
    if last_frame is not None:
        cv2.imwrite('last_frame.jpg', last_frame)
        print("Last frame saved as 'last_frame.jpg'")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    frame1 = cv2.imread('last_frame.jpg')
    frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

    if ret:
        angles_df, skeleton_img, predicted_pose, incorrect_parts = process_image(frame, selected_pose)

        if angles_df is None:
            st.error("No human pose detected. Please try again.")
        else:
            # Display the classified pose
            st.subheader(f"🧘 Classified Pose: **{predicted_pose}**")

            # Check if the pose matches
            if predicted_pose == selected_pose:
                feedback = "Perfect pose!"
                st.success(feedback)
                play_audio(feedback)
            else:
                feedback = "Pose is not proper!"
                st.error(feedback)
                play_audio(feedback)

            # Display images
            col1, col2 = st.columns(2)
            with col1:
                st.image(frame, caption="Original Image", use_column_width=True)
            with col2:
                st.image(skeleton_img, caption="Pose Skeleton", use_column_width=True)

            # Show extracted angles
            st.subheader("📏 Extracted Angles:")
            st.write(angles_df)

            # Show incorrect parts
            if incorrect_parts:
                st.subheader("❌ Incorrect Parts:")
                st.write(incorrect_parts)
    else:
        st.error("Failed to capture image.")