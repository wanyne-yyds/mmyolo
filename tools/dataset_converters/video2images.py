import cv2
import os
import os.path as osp
from pathlib import Path

def converters_vides(input_dir, output_dir):
    save_fps = 1

    videopath = Path(input_dir).rglob('*.*')
    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    for videoname in videopath:
        cap = cv2.VideoCapture(str(videoname))
        name = str(videoname).replace(input_dir, output_dir).replace('.mp4', '') \
                                                          .replace('.avi', '') \
                                                          .replace('.h264', '')
        pathname, filename = osp.split(name)
        video_fps    = cap.get(cv2.CAP_PROP_FPS)
        frame_count  = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_chn    = 3
        if video_fps < 1 or video_fps > 100:
            video_fps = 15

        frame_inter = int((video_fps - save_fps) / save_fps)
        frame_index = 0

        while (True):
            if frame_index >= frame_count:
                break

            res, frame   = cap.read()

            frame_index += 1

            if frame_index % frame_inter != 0:
                continue

            if not res or frame is None:
                continue

            imagefilename = '%s_%s_%d.jpg'%(pathname, filename, frame_index)
            cv2.imwrite(imagefilename, frame)

if __name__ == "__main__":
    input_dir = '/mmyolo/data/HODTestDataVideo'
    output_dir = '/mmyolo/data/HODTestDataVideo/images'
    converters_vides(input_dir, output_dir)