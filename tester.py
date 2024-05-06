import cv2
import math
import numpy as np
from tqdm import tqdm
from utils.poser import get_pose, get_pose_model
from utils.video_out import prepare_vid_out

import warnings
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")


def fall_detection(poses):
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

        if left_shoulder_y > left_foot_y - len_factor and left_body_y > left_foot_y - (
                len_factor / 2) and left_shoulder_y > left_body_y - (len_factor / 2) or (
                right_shoulder_y > right_foot_y - len_factor and right_body_y > right_foot_y - (
                len_factor / 2) and right_shoulder_y > right_body_y - (len_factor / 2)) \
                or difference < 0:
            return True, (xmin, ymin, xmax, ymax)

    return False, None


def process_video(video_path):
    vid_cap = cv2.VideoCapture(video_path)

    if not vid_cap.isOpened():
        print('Error while trying to read video.')
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

        _image = image[0].permute(1, 2, 0) * 255
        _image = _image.cpu().numpy().astype(np.uint8)
        _image = cv2.cvtColor(_image, cv2.COLOR_RGB2BGR)

        is_fall, bbox = fall_detection(output)

        if is_fall:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=(0, 0, 255),
                          thickness=5, lineType=cv2.LINE_AA)
            cv2.putText(_image, 'Person Fell down', (11, 100), 0, 1, [0, 0, 2550], thickness=3, lineType=cv2.LINE_AA)

        vid_out.write(_image)

    vid_out.release()
    vid_cap.release()


videos_path = 'data/Laddr Fall.mp4'
process_video(videos_path)
