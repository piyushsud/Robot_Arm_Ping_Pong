import cv2

# intel realsense
realsense = cv2.VideoCapture(3, cv2.CAP_DSHOW)
realsense.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
realsense.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, img = realsense.read()
    cv2.imshow("img", img)
    cv2.imwrite("C:/Users/piyus/Robot_Arm_Ping_Pong/camera_calibration/realsense_images/checkerboard_upright.png", img)

    # the 'q' button is set as the
    # quitting button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
