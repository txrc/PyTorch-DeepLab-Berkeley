import cv2
import os

DIR_PATH = "D:/PyTorch-DeepLab-Berkeley/SGDrivingOverlayed" # Absolute Path
IMAGE_EXT = "png"
OUTPUT_FILE = "SG_Driving_overlayed.mp4"
images = []
for image in os.listdir(DIR_PATH):
	if image.endswith(IMAGE_EXT):
		images.append(image)

# Determine height and width of images
image_path = os.path.join(DIR_PATH, images[0])
image = cv2.imread(image_path)

# Check Image
# cv2.imshow("First Image", image)
# # WaitKey Displays the image for specified milliseconds or if set to 0 any key that is pressed
# cv2.waitKey(0)
height, width, channels = image.shape
# #print(image.shape)


fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_FILE, fourcc, 30.0, (width,height))


for image in images:
	image_path = os.path.join(DIR_PATH, image)
	image_frame = cv2.imread(image_path)

	out.write(image_frame)

	# cv2.imshow("Video Sequence", image_frame)

	# If key entered is 'q', exit
	# if (cv2.waitKey(1) & 0xFF) == ord('q'):
	# 	break


out.release()
cv2.destroyAllWindows()