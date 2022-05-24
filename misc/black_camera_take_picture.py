import cv2
import time

TOLERANCE = 0.05

cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)

#cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

im_path_left = 'C:/Users/piyus/Robot_Arm_Ping_Pong/'
# im_path_right = '/Robot_Arm/stereo_sbgm_test_pictures/right/'

start_time = time.time()
picture_number = 163
busy = False

while True:

    # Capture the video frame
    # by frame

    ret, frame = cam.read()

    # print(frame.shape)
    curr_time = time.time()
    print(curr_time - start_time)

    if busy is False:
        if abs(curr_time - start_time - 1) < TOLERANCE:
            busy = True
            print(picture_number)
            cv2.imwrite("C:/Users/piyus/Robot_Arm_Ping_Pong/camera_calibration/black_camera_images/left/checkerboard_" +
                str(picture_number) + ".png", frame[:, 0:1280, :])
            cv2.imwrite("C:/Users/piyus/Robot_Arm_Ping_Pong/camera_calibration/black_camera_images/right/checkerboard_" +
                str(picture_number) + ".png", frame[:, 1280:2560, :])
            picture_number += 1
            start_time = curr_time

    if curr_time - start_time > TOLERANCE:
        busy = False

    cv2.imshow("img", frame)
# the 'q' button is set as the
# quitting button you may use any
# desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
cam.release()
# Destroy all the windows
cv2.destroyAllWindows()