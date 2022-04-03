import cv2
# from moviepy.editor import *

STARTING_FRAME = 393
IN_PATH = '/home/dilara/UGI-endoscopy/videos/OGD_1.mov'
OUT_PATH = '/home/dilara/UGI-endoscopy/video-split-images/video1-split'

# ## Finding out the FPS - source: https://www.geeksforgeeks.org/moviepy-fps-of-video-file-clip/?ref=lbp
# clip = VideoFileClip('/home/dilara/UGI-endoscopy/videos/OGD_1.mov').subclip(0, 5)
# rate = clip.fps
# print("FPS : " + str(rate)) 

# count = STARTING_FRAME
# ## Splitting the video into frames - source: https://techtutorialsx.com/2021/04/29/python-opencv-splitting-video-frames/
# video = cv2.VideoCapture(IN_PATH)
# while(True):
#     success, frame = video.read()
#     if success:
#         pass
#     else:
#         break
#     count +=1
#     if count % 10 == 0:
#         cv2.imwrite(f'./video-split-images/video1-split/frame_{count}.jpg', frame)

# video.release()