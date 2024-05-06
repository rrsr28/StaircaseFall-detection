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

# Video capture
cap = cv2.VideoCapture('Laddr Fall_keypoint_s.mp4')

# Get the original frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a window with the size of the original video
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', frame_width, frame_height)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection with OwlViT
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    texts = [["shoes", "person", "stairs", "ladder", "escalator", "steps"]]
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
        cv2.putText(frame, 'Fall Detected', (frame_width - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
