import cv2
import numpy as np

def get_chessboard(columns, rows, show=False):
    """
    Take a picture with a chessboard visible in both captures.

    ``columns`` and ``rows`` should be the number of inside corners in the
    chessboard's columns and rows. ``show`` determines whether the frames
    are shown while the cameras search for a chessboard.
    """
    frames_stereo = cv2.imread('/misc/checkerboard_image_camera2.jpg')
    img_gray = cv2.cvtColor(frames_stereo, cv2.COLOR_BGR2GRAY)
    img = img_gray[380:450, 313:413]
    frame = cv2.resize(img, (img.shape[1]*3, img.shape[0]*3))
    if show:
        cv2.imshow("img", frame)
    found_chessboard, found_corners = cv2.findChessboardCorners(frame, (columns, rows), flags=cv2.CALIB_CB_FAST_CHECK +
                                                                cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                                cv2.CALIB_CB_NORMALIZE_IMAGE)
    print(found_chessboard)
    print(found_corners)
    cv2.waitKey(0)
    # corners = cv2.goodFeaturesToTrack(frame, 25, 0.01, 10)
    # corners = np.int0(corners)
    #
    # for i in corners:
    #     x, y = i.ravel()
    #     cv2.circle(frame, (x,y),3,(0,0,255),-1)
    #
    # cv2.imshow('Corners',frame)
if __name__ == "__main__":
    get_chessboard(6, 9, show=True)