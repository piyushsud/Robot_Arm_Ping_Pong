import os
import cv2
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

images_path = "/Robot_Arm/ping_pong_photos_labeled_clean/Ping-Pong-Robot-PascalVOC-export/JPEGImages/"
train_images_dest = "C:/Users/piyus/GrabCAD/Robot_Arm/yolo_formatted_data/train_image_folder/"
valid_images_dest = "C:/Users/piyus/GrabCAD/Robot_Arm/yolo_formatted_data/valid_image_folder/"
train_annot_dest = "C:/Users/piyus/GrabCAD/Robot_Arm/yolo_formatted_data/train_annot_folder/"
valid_annot_dest = "C:/Users/piyus/GrabCAD/Robot_Arm/yolo_formatted_data/valid_annot_folder/"
annotations_path = "/Robot_Arm/ping_pong_photos_labeled_clean/Ping-Pong-Robot-PascalVOC-export/Annotations/"
train_annot_dest_darknet = "C:/Users/piyus/GrabCAD/Robot_Arm/code/darknet-master/build/darknet/x64/data/obj/"
train_images_dest_darknet = train_annot_dest_darknet
train_file_path = "/Robot_Arm/code/darknet-master/build/darknet/x64/data/train.txt"
yolo_eval_path = "/Robot_Arm/code/tensorflow-yolov4-tflite/data/dataset/val_ping_pong.txt"

IMAGE_SIZE = 192
NEW_IMAGE_SIZE = 96

def train_test_split():
    i = 0

    # train_file = open(train_file_path, "w")
    val_file = open(yolo_eval_path, "w")
    for filename in os.listdir(annotations_path):

        mytree = ET.parse(annotations_path + filename)
        myroot = mytree.getroot()
        for bndbox in myroot.iter('bndbox'):
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))

        bbox_width = xmax - xmin
        bbox_height = ymax - ymin

        filename_obj = Path(filename)
        filename_with_suffix = filename_obj.with_suffix('.jpg')
        image_path = images_path + str(filename_with_suffix)
        img = cv2.imread(image_path)
        height = img.shape[0]
        cropped_img = img[0:(height - height % 32), :, :]
        cropped_height = cropped_img.shape[0]
        cropped_width = cropped_img.shape[1]

        left = max(0, xmin - (IMAGE_SIZE - bbox_width))
        top = max(0, ymin - (IMAGE_SIZE - bbox_height))
        if left == 0:
            x = 0
        else:
            x = np.random.randint(left, min(cropped_width - IMAGE_SIZE, xmin))
        if top == 0:
            y = 0
        else:
            y = np.random.randint(top, min(cropped_height - IMAGE_SIZE, ymin))

        xmin -= x
        xmax -= x
        ymin -= y
        ymax -= y
        cropped_image_aoi = cropped_img[y:y+IMAGE_SIZE, x:x+IMAGE_SIZE]

        for width in myroot.iter('width'):
            width.text = str(IMAGE_SIZE)
        for height in myroot.iter('width'):
            height.text = str(IMAGE_SIZE)

        img1, img1_bbox, img2, img2_bbox, img3, img3_bbox, img4, img4_bbox = augment_data(cropped_image_aoi, xmin, ymin, xmax, ymax)

        augmented_images = [img1, img2, img3, img4]
        augmented_bboxes = [img1_bbox, img2_bbox, img3_bbox, img4_bbox]

        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2

        rot_str = ["", "rot90", "rot180", "rot270"]

        for j in range(len(augmented_images)):
            filename_str = "img" + str(i) + rot_str[j]
            xmin = augmented_bboxes[j][0][0]//2
            ymin = augmented_bboxes[j][0][1]//2
            xmax = augmented_bboxes[j][1][0]//2
            ymax = augmented_bboxes[j][1][1]//2

            for filename in myroot.iter('filename'):
                filename.text = filename_str + ".jpg"
            for bndbox in myroot.iter('bndbox'):
                bndbox.find('xmin').text = str(xmin)
                bndbox.find('ymin').text = str(ymin)
                bndbox.find('xmax').text = str(xmax)
                bndbox.find('ymax').text = str(ymax)

            x_middle = ((xmin + xmax) / 2) / NEW_IMAGE_SIZE
            y_middle = ((ymin + ymax) / 2) / NEW_IMAGE_SIZE
            new_bbox_width = (xmax - xmin) / NEW_IMAGE_SIZE
            new_bbox_height = (ymax - ymin) / NEW_IMAGE_SIZE

            img = cv2.resize(augmented_images[j], (NEW_IMAGE_SIZE, NEW_IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            if i % 20 == 0:
                for path in myroot.iter('path'):
                    path.text = valid_images_dest + filename_str + ".jpg"
                #mytree.write(valid_annot_dest + filename_str + ".xml")
                #cv2.imwrite(valid_images_dest + filename_str + ".jpg", img)
                val_str = valid_images_dest + filename_str + ".jpg " + str(xmin) + "," + str(ymin) + "," + str(xmax) + "," + str(ymax) + "," + "0\n"
                val_file.write(val_str)
            else:
                for path in myroot.iter('path'):
                    path.text = train_images_dest + filename_str + ".jpg"
                # train_file.write("data/obj/" + filename_str + ".jpg\n")
                # file1 = open(train_annot_dest_darknet + filename_str + ".txt", "w")
                # bbox_str = "0 " + str(x_middle) + " " + str(y_middle) + " " + str(new_bbox_width) + " " + str(new_bbox_height)
                # file1.write(bbox_str)
                # file1.close()
                # #mytree.write(train_annot_dest + filename_str + ".xml")
                # cv2.imwrite(train_images_dest_darknet + filename_str + ".jpg", img)



        # # Draw a rectangle with blue line borders of thickness of 2 px
        # cropped_image_aoi = cv2.rectangle(img1, img1_bbox[0], img1_bbox[1], color, thickness)
        # cropped_image_aoi_rot90 = cv2.rectangle(img2, img2_bbox[0], img2_bbox[1], color, thickness)
        # cropped_image_aoi_rot180 = cv2.rectangle(img3, img3_bbox[0], img3_bbox[1], color, thickness)
        # cropped_image_aoi_rot270 = cv2.rectangle(img4, img4_bbox[0], img4_bbox[1], color, thickness)


        # cv2.imshow("cropped image 1", cropped_image_aoi)
        # cv2.imshow("cropped image 2", cropped_image_aoi_rot90)
        # cv2.imshow("cropped image 3", cropped_image_aoi_rot180)
        # cv2.imshow("cropped image 4", cropped_image_aoi_rot270)
        #
        # cv2.waitKey()
        # if i % 20 == 0:
        #     cv2.imwrite(validation_images_dest + str(filename_with_suffix), cropped_img)
        #     os.rename(annotations_path + filename, valid_annot_dest + filename)
        # else:
        #     cv2.imwrite(train_images_dest + str(filename_with_suffix), cropped_img)
        #     os.rename(annotations_path + filename, train_annot_dest + filename)

        i += 1

    # train_file.close()
    val_file.close()

def augment_data(img, xmin, ymin, xmax, ymax):

    cx = int(IMAGE_SIZE/2)
    cy = int(IMAGE_SIZE/2)

    deg_90_cw_rot_mat = np.array([
        [0, 1],
        [-1, 0]
    ])
    deg_180_rot_mat = np.array([
        [-1, 0],
        [0, -1]
    ])
    deg_90_ccw_rot_mat = np.array([
        [0, -1],
        [1, 0]
    ])
    bbox_mat = np.array([
        [xmin - cx, xmin - cx, xmax - cx, xmax - cx],
        [ymin - cy, ymax - cy, ymin - cy, ymax - cy]
    ])

    img1 = img
    img1_bbox = ((xmin, ymin), (xmax, ymax))

    img2 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img2_bbox_mat = np.matmul(deg_90_ccw_rot_mat, bbox_mat)
    img2_bbox = ((img2_bbox_mat[0][1] + cx, img2_bbox_mat[1][1] + cy), (img2_bbox_mat[0][2] + cx, img2_bbox_mat[1][2] + cy))

    img3 = cv2.rotate(img, cv2.ROTATE_180)
    img3_bbox_mat = np.matmul(deg_180_rot_mat, bbox_mat)
    img3_bbox = ((img3_bbox_mat[0][3] + cx, img3_bbox_mat[1][3] + cy), (img3_bbox_mat[0][0] + cx, img3_bbox_mat[1][0] + cy))

    img4 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img4_bbox_mat = np.matmul(deg_90_cw_rot_mat, bbox_mat)
    img4_bbox = ((img4_bbox_mat[0][2] + cx, img4_bbox_mat[1][2] + cy), (img4_bbox_mat[0][1] + cx, img4_bbox_mat[1][1] + cy))

    return img1, img1_bbox, img2, img2_bbox, img3, img3_bbox, img4, img4_bbox

if __name__ == "__main__":
    train_test_split()

