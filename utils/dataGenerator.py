import os
import cv2
import mediapipe as mp
import pandas as pd

# Load MediaPipe pose estimation model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Define the body parts in the order provided by MediaPipe
body_parts = ['nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer',
              'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow',
              'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index', 'right_index',
              'left_thumb', 'right_thumb', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
              'right_ankle', 'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index']

# Function to extract keypoints from an image
def extract_keypoints(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        keypoints = [None] * len(body_parts)  # Initialize list for all body parts
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            keypoints[i] = (landmark.x, landmark.y, landmark.z)  # Store (x, y, z) coordinates
        return keypoints
    else:
        return None


# Function to process images in a dataset folder
def process_images(dataset_folder):
    data = []
    for folder_name in os.listdir(dataset_folder):
        folder_path = os.path.join(dataset_folder, folder_name)
        if os.path.isdir(folder_path):
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                if image_path.endswith(('.jpg', '.jpeg', '.png')):
                    keypoints = extract_keypoints(image_path)
                    if keypoints:
                        row = []
                        for kp in keypoints:
                            row.extend(kp)  # Unpack (x, y, z) and extend the row
                        row.append(folder_name)  # Add label
                        data.append(row)
    return data

# Function to save keypoints data to a CSV file
def save_to_csv(data, csv_file):
    columns = []
    for i in range(len(body_parts)):
        columns.extend([f"{body_parts[i]}_x", f"{body_parts[i]}_y", f"{body_parts[i]}_z"])  # Separate columns for x, y, z
    columns.append("Label")
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(csv_file, index=False)

# Specify the dataset folder
dataset_folder = r"C:\Users\rrsan\Documents\My Docs\College\Projects\StaircaseFall-detection\data\Images"

# Extract keypoints from images in the dataset folder
keypoints_data = process_images(dataset_folder)

# Save keypoints data to a CSV file
save_to_csv(keypoints_data, "../data/keypoints_data.csv")
