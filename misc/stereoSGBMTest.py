import numpy as np
import cv2
import matplotlib.pyplot as plt

def downsample_image(image, reduce_factor):
	for i in range(0,reduce_factor):
		#Check if image is color or grayscale
		if len(image.shape) > 2:
			row,col = image.shape[:2]
		else:
			row,col = image.shape

		image = cv2.pyrDown(image, dstsize= (col//2, row // 2))
	return image

left_camera_intrinsic_matrix = np.array(
[
    [574.94583505, 0, 320.0],
    [0, 575.36383871, 240.0],
    [0, 0, 1]
])

left_camera_distortion_coefficients = np.array([[0.06088478,  0.00711266, -0.00356545, -0.00076046, -0.32904665]])

right_camera_intrinsic_matrix = np.array(
[
    [579.50288857, 0, 320.0],
    [0, 579.82851688, 240.0],
    [0, 0, 1]
])

right_camera_distortion_coefficients = np.array([[0.05268772,  0.0604447,  -0.00271751, -0.00077472, -0.38804226]])


#Specify image paths
img_path1 = 'C:/Users/piyus/GrabCAD/Robot_Arm/stereo_sbgm_test_pictures/left/im0.jpg'
img_path2 = 'C:/Users/piyus/GrabCAD/Robot_Arm/stereo_sbgm_test_pictures/right/im0.jpg'

#Load pictures
img_1 = cv2.imread(img_path1)
img_2 = cv2.imread(img_path2)

#Get height and width. Note: It assumes that both pictures are the same size. They HAVE to be same size
h,w = img_2.shape[:2]

#Get optimal camera matrix for better undistortion
new_camera_matrix_left, roi_left = cv2.getOptimalNewCameraMatrix(left_camera_intrinsic_matrix, left_camera_distortion_coefficients,(w,h),1,(w,h))
new_camera_matrix_right, roi_right = cv2.getOptimalNewCameraMatrix(right_camera_intrinsic_matrix, right_camera_distortion_coefficients,(w,h),1,(w,h))

#Undistort images
img_1_undistorted = cv2.undistort(img_1, left_camera_intrinsic_matrix, left_camera_distortion_coefficients, None, new_camera_matrix_left)
img_2_undistorted = cv2.undistort(img_2, right_camera_intrinsic_matrix, right_camera_distortion_coefficients, None, new_camera_matrix_right)

#cv2.imwrite('C:/Users/piyus/GrabCAD/Robot_Arm/stereo_sbgm_test_pictures/left/im0_undistorted.jpg', img_1_undistorted)
#cv2.imwrite('C:/Users/piyus/GrabCAD/Robot_Arm/stereo_sbgm_test_pictures/right/im0_undistorted.jpg', img_2_undistorted)

#Downsample each image 3 times (because they're too big)
img_1_downsampled = downsample_image(img_1_undistorted,1)
img_2_downsampled = downsample_image(img_2_undistorted,1)

#Set disparity parameters
#Note: disparity range is tuned according to specific parameters obtained through trial and error.
win_size = 3
min_disp = 2 #-1
max_disp = 20 #63
num_disp = max_disp - min_disp # Needs to be divisible by 16
#Create Block matching object.
stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
 numDisparities = num_disp,
 blockSize = 3,
 uniquenessRatio = 5,
 speckleWindowSize = 5,
 speckleRange = 5,
 disp12MaxDiff = 1,
 P1 = 8*3*win_size**2,#8*3*win_size**2,
 P2 =32*3*win_size**2) #32*3*win_size**2)
#Compute disparity map
print ("\nComputing the disparity  map...")
disparity_map = stereo.compute(img_1_downsampled, img_2_downsampled)

#Show disparity map before generating 3D cloud to verify that point cloud will be usable.
plt.imshow(disparity_map,'gray')
plt.show()