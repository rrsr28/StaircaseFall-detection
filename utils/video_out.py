import cv2
from utils.datasets import letterbox

def prepare_vid_out(video_path, vid_cap):
    vid_write_image = letterbox(vid_cap.read()[1], 960, stride=64, auto=True)[0]
    resize_height, resize_width = vid_write_image.shape[:2]
    out_video_name = f"results/{video_path.split('/')[-1].split('.')[0]}_keypoint_s.mp4"
    print(out_video_name)
    out = cv2.VideoWriter(out_video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (resize_width, resize_height))
    return out