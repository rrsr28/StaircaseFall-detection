import cv2
import csv
import torch
import pandas as pd
from PIL import Image
import mediapipe as mp
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# Load MediaPipe pose estimation model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Load OwlViT model for object detection
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model_vit = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")


def read_boxes_from_csv(csv_file):
    boxes = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            xmin, ymin, xmax, ymax = map(float, row)
            boxes.append((xmin, ymin, xmax, ymax))
    return boxes


def check_intersection_with_csv(csv_file, box2):
    box1 = read_boxes_from_csv(csv_file)
    xmin2, ymin2, xmax2, ymax2 = box2

    for box in box1:
        xmin1, ymin1, xmax1, ymax1 = box
        # Check if boxes intersect along the x-axis
        x_intersect = (xmin1 <= xmax2) and (xmax1 >= xmin2)
        # Check if boxes intersect along the y-axis
        y_intersect = (ymin1 <= ymax2) and (ymax1 >= ymin2)
        # If both x and y intersections occur, the boxes intersect or touch
        if x_intersect and y_intersect:
            return True

    return False


# Video capture
video_file = 'fall_keypoint_s.mp4'
cap = cv2.VideoCapture(video_file)
video_name = video_file.split('_')[0]


# csv_file = "falls_boxs.csv"
csv_file = f"falls_boxs_{video_name}.csv"

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
    texts = [["stairs", "ladder", "escalator", "steps"]]
    inputs = processor(text=texts, images=image_pil, return_tensors="pt")
    outputs = model_vit(**inputs)
    target_sizes = torch.Tensor([image_pil.size[::-1]])
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
    text = texts[0]
    boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]

    # Draw rectangles for OwlViT object detection
    for box, score, label in zip(boxes, scores, labels):
        box = [int(i) for i in box.tolist()]
        print(box)
        intersection = check_intersection_with_csv(csv_file, box)
        print(intersection)
        print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"{text[label]}: {score:.2f}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

        # Check if intersection is detected and print the message
        if intersection:
            print(f"Fall detected from {text[label]}")

    # Show frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
