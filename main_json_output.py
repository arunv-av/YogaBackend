import os
import cv2
import mediapipe as mp
import numpy as np
import json

# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, smooth_landmarks=True)

# Define angle calculation function
def calculate_angle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1.x, landmark1.y, landmark1.z
    x2, y2, _ = landmark2.x, landmark2.y, landmark2.z
    x3, y3, _ = landmark3.x, landmark3.y, landmark3.z

    radians = np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2)
    angle = np.abs(np.degrees(radians))
    return angle if angle >= 0 else angle + 360

# Define the key joint triplets for angle calculation
angle_name_list = ["L-wrist", "R-wrist", "L-elbow", "R-elbow", "L-shoulder", "R-shoulder",
                   "L-knee", "R-knee", "L-ankle", "R-ankle", "L-hip", "R-hip"]
angle_coordinates = [
    [13, 15, 19], [14, 16, 18], [11, 13, 15], [12, 14, 16], [13, 11, 23], [14, 12, 24],
    [23, 25, 27], [24, 26, 28], [23, 27, 31], [24, 28, 32], [24, 23, 25], [23, 24, 26]
]

# Folder containing pose subfolders
input_folder = "dataset"  # Change to your dataset folder path
output_json = "yoga_angles.json"

# Store all results in a dictionary
results_dict = {}

# Loop through all pose folders
for pose_name in os.listdir(input_folder):
    pose_folder = os.path.join(input_folder, pose_name)

    # Ensure it's a folder
    if not os.path.isdir(pose_folder):
        continue

    # List to store angles for this pose
    pose_data = []

    # Process all images in the folder
    for image_file in os.listdir(pose_folder):
        image_path = os.path.join(pose_folder, image_file)

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Skipping {image_file}: Unable to read image.")
            continue

        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # Extract angles if landmarks are detected
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            angles = {name: calculate_angle(landmarks[a], landmarks[b], landmarks[c]) 
                      for name, (a, b, c) in zip(angle_name_list, angle_coordinates)}

            # Store the angles along with image filename
            pose_data.append({"image": image_file, "angles": angles})

    # Store the pose data in the dictionary
    results_dict[pose_name] = pose_data

# Save results to JSON file
with open(output_json, 'w') as json_file:
    json.dump(results_dict, json_file, indent=4)

print(f"JSON file '{output_json}' successfully created with angles and pose labels.")
