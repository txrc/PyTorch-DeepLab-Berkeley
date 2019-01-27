import cv2
import numpy as np
import os

# Playing video from file:
capture = cv2.VideoCapture('./SG_Driving.MOV')

try:
    if not os.path.exists('SG_Driving'):
        os.makedirs('SG_Driving')
except OSError:
    print ('Error: Creating directory of data')

# frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
# fps = int(capture.get(cv2.CAP_PROP_FPS))
# print(frame_count)
# print(fps)

currentFrame = 0
while(True):
    # Capture frame-by-frame
    ret, frame = capture.read()

    # Check for end of image frames
    if not ret: 
    	break
    frame = cv2.flip(frame, -1)
    # Saves image of the current frame in jpg file # Also ensure that padding is done to 
    name = './SG_Driving/frame' + str(currentFrame).zfill(10) + '.png'

    print ('Creating...' + name)
    cv2.imwrite(name, frame)

    # To stop duplicate images
    currentFrame += 1

# When everything done, release the capture
capture.release()
cv2.destroyAllWindows()