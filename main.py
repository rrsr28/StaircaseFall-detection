import cv2
import pickle
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import StandardScaler
import cvzone
import math
import torch
from ultralytics import YOLO
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# Load MediaPipe pose estimation model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Load OwlViT model for object detection
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model_vit = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# Define the body parts in the order provided by MediaPipe
body_parts = ['nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer',
              'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow',
              'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index', 'right_index',
              'left_thumb', 'right_thumb', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
              'right_ankle', 'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index']

# Function to extract keypoints using MediaPipe
def extract_keypoints(image):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        keypoints = [None] * len(body_parts)  # Initialize list for all body parts
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            keypoints[i] = (landmark.x, landmark.y, landmark.z)  # Store (x, y, z) coordinates
        return keypoints
    else:
        return None

# Load the model for keypoints extraction
with open('models/model2.pkl', 'rb') as file:
    model = pickle.load(file)

# Load YOLO object detection model
model_yolo = YOLO('yolov8s.pt')

classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# Video capture
cap = cv2.VideoCapture('data/Ben Fall.mp4')

# Create a resizable window
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection with OwlViT
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    texts = [["falling down stairs", "falling", "human who fell", "human falling down", "stairs", "ladder", "escalator", "steps"]]
    inputs = processor(text=texts, images=image_pil, return_tensors="pt")
    outputs = model_vit(**inputs)
    target_sizes = torch.Tensor([image_pil.size[::-1]])
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
    text = texts[0]
    boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]

    # Draw rectangles for OwlViT object detection
    for box, score, label in zip(boxes, scores, labels):
        box = [int(i) for i in box.tolist()]
        print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"{text[label]}: {score:.2f}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

    # YOLO object detection
    results_yolo = model_yolo(frame)

    for info in results_yolo[0]:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = classnames[class_detect]
            conf = math.ceil(confidence * 100)

            # Perform fall detection using model2 predictions
            keypoints = extract_keypoints(frame)
            if keypoints:
                row = []
                for kp in keypoints:
                    row.extend(kp)
                row = np.array(row).reshape(1, -1)
                row_scaled = StandardScaler().fit_transform(row)
                pred = model.predict(row_scaled)

            # implement fall detection using the coordinates x1,y1,x2
            height = y2 - y1
            width = x2 - x1
            threshold = height - width

            if conf > 80 and class_detect == 'person':
                cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=2)

            if threshold < 10 and class_detect == 'person':
                cvzone.putTextRect(frame, 'Fall Detected', [height, width], thickness=2, scale=2)

    # Show frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
