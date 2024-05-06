import cv2
import csv
import math
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from utils.poser import get_pose, get_pose_model
from utils.video_out import prepare_vid_out
from transformers import OwlViTProcessor, OwlViTForObjectDetection

import warnings
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")


def save_to_csv(falls_boxs, file_path):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['xmin', 'ymin', 'xmax', 'ymax'])  # Write header
        writer.writerows(falls_boxs)


def detect_objects(image):

    # Convert PIL image to OpenCV format (BGR)
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Get image size
    height, width, _ = image_cv2.shape

    # Detect objects in the image
    texts = [["stairs", "ladder", "escalator", "steps"]]
    inputs = processor(text=texts, images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([[height, width]])

    # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    # Draw bounding boxes on the image
    for box, score, label in zip(boxes, scores, labels):
        box = [int(i) for i in box.tolist()]
        print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
        cv2.rectangle(image_cv2, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(image_cv2, f"{text[label]}: {score:.2f}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image_cv2


def fall_detection(poses, image):
    global falls_boxs
    is_fall = False
    bbox = None

    for pose in poses:
        xmin, ymin = (pose[2] - pose[4] / 2), (pose[3] - pose[5] / 2)
        xmax, ymax = (pose[2] + pose[4] / 2), (pose[3] + pose[5] / 2)

        left_shoulder_y = pose[23]
        left_shoulder_x = pose[22]
        right_shoulder_y = pose[26]
        left_body_y = pose[41]
        left_body_x = pose[40]
        right_body_y = pose[44]

        len_factor = math.sqrt(((left_shoulder_y - left_body_y) ** 2 + (left_shoulder_x - left_body_x) ** 2))
        left_foot_y = pose[53]
        right_foot_y = pose[56]
        dx = int(xmax) - int(xmin)
        dy = int(ymax) - int(ymin)
        difference = dy - dx

        if left_shoulder_y > left_foot_y - len_factor and left_body_y > left_foot_y - (len_factor / 2) and left_shoulder_y > left_body_y - (len_factor / 2) or (right_shoulder_y > right_foot_y - len_factor and right_body_y > right_foot_y - (len_factor / 2) and right_shoulder_y > right_body_y - (len_factor / 2)) or difference < 0:
            is_fall = True
            bbox = (xmin, ymin, xmax, ymax)
            falls_boxs.append((xmin, ymin, xmax, ymax))
            break

    return is_fall, bbox

def process_video(video_path):
    vid_cap = cv2.VideoCapture(video_path)

    if not vid_cap.isOpened():
        print('Error while trying to read video. Please check path again')
        return

    model, device = get_pose_model()
    vid_out = prepare_vid_out(video_path, vid_cap)

    success, frame = vid_cap.read()
    _frames = []
    while success:
        _frames.append(frame)
        success, frame = vid_cap.read()

    for image in tqdm(_frames):
        image, output = get_pose(image, model, device)
        # detected_objects_image = detect_objects(image)

        _image = image[0].permute(1, 2, 0) * 255
        _image = _image.cpu().numpy().astype(np.uint8)
        _image = cv2.cvtColor(_image, cv2.COLOR_RGB2BGR)

        is_fall, bbox = fall_detection(output, image)

        if is_fall:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=(0, 0, 255),
                          thickness=5, lineType=cv2.LINE_AA)
            cv2.putText(_image, 'Person Fell down', (11, 100), 0, 1, [0, 0, 2550], thickness=3, lineType=cv2.LINE_AA)

        vid_out.write(_image)

    vid_out.release()
    vid_cap.release()

falls_boxs = []
videos_path = 'data/fall.mp4'
video_name = videos_path.split('/')[-1].split('.')[0]  # Extracting the video name without extension
process_video(videos_path)
save_to_csv(falls_boxs, f'falls_boxs_{video_name}.csv')  # Using f-string to insert the video name into the CSV file name

