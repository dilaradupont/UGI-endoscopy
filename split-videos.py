import cv2
from moviepy.editor import *
  
## Finding out the FPS - source: https://www.geeksforgeeks.org/moviepy-fps-of-video-file-clip/?ref=lbp
clip = VideoFileClip('/Users/dilaradupont/Desktop/UCL/Year 3/UGI-endoscopy/OGD_1.mov').subclip(0, 5)
rate = clip.fps
print("FPS : " + str(rate)) 

## Splitting the video into frames - source: https://techtutorialsx.com/2021/04/29/python-opencv-splitting-video-frames/
count = 0
video = cv2.VideoCapture('/Users/dilaradupont/Desktop/UCL/Year 3/UGI-endoscopy/OGD_1.mov')
while(True):
    success, frame = video.read()
    if success:
        pass
    else:
        break
    count = count+1
    if count % 10 == 0:
        cv2.imwrite(f'/Users/dilaradupont/Desktop/UCL/Year 3/UGI-endoscopy/video-split/frame_{count}.jpg', frame)

video.release()