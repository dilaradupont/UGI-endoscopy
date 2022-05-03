import cv2
from moviepy.editor import *

STARTING_FRAME = 0
VIDEO_NAME = '654acacf-82cb-4b1a-aca0-f045ccfe30a0.avi'
RELATIVE_PATH = 'Quality of Mucosal Views/Optimal Views'
IMAGE_PREFIX = 'optimal'
FRAME_MULT = 10

def main():
    ## Splitting the video into frames - source: https://techtutorialsx.com/2021/04/29/python-opencv-splitting-video-frames/
    count = STARTING_FRAME
    video = cv2.VideoCapture(f'/Volumes/USB-C/Dilara/{RELATIVE_PATH}/{VIDEO_NAME}')
    while(True):
        success, frame = video.read()
        if success:
            pass
        else:
            break
        if count % FRAME_MULT == 0:
            cv2.imwrite(f'/Volumes/USB-C/Dilara/{RELATIVE_PATH}/{IMAGE_PREFIX}_frame_{count}.jpg', frame)
        count += 1
    video.release()

if __name__ == '__main__':
    main()