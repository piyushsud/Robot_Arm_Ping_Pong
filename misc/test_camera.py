import cv2

cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

im_path_left = '/Robot_Arm/stereo_sbgm_test_pictures/left/'
im_path_right = '/Robot_Arm/stereo_sbgm_test_pictures/right/'

# while (True):

# Capture the video frame
# by frame
ret, frame = cam.read()

print(frame.shape)

cv2.imwrite(im_path_left + "im1.jpg", frame[:, 0:640, :])
cv2.imwrite(im_path_right + "im1.jpg", frame[:, 640:1280, :])

# cv2.imshow("frame", frame)
# the 'q' button is set as the
# quitting button you may use any
# desired button of your choice
# if cv2.waitKey(1) & 0xFF == ord('q'):
#     break

# # After the loop release the cap object
# cam.release()
# # Destroy all the windows
# cv2.destroyAllWindows()