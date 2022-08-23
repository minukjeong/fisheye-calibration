import cv2
import numpy as np
import os

path = "/home/mujung/PycharmProjects/lens/"
img_infos = os.listdir(path)
#cap = cv2.VideoCapture(path + "103.mp4")

framenum = 0

for img_info in img_infos:
        framenum += 1
        #ret, img = cap.read()
        f = 100
        img = cv2.imread(path + img_info)
        cy = img.shape[0] / 2
        cx = img.shape[1] / 2
        # cx = img_info['cx']
        # cy = img_info['cy']

        if img is not None:
            mtx = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
            dist = np.array([0.0, 0.0, 0.0, 0.0])

            scale = 1
            balance = 4
            if scale is not None:
                f = f * scale * balance
                cx = cx * scale
                cy = cy * scale

            mtxnew = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
            R = np.eye(3)
            R[2,2] = 2
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(mtx, dist, R, mtxnew, (img.shape[1], img.shape[0]), cv2.CV_16SC2)

            print("imgshape:{}".format(img.shape))
            #img = cv2.warpAffine(img, rotMat, (img.shape[1],img.shape[0]))
            #cv2.line(img, (1531, 2667), (1275, 2560),color=(255,255,255), thickness=10)
            #cv2.line(img, (1275, 2560), (986, 2086), color=(255, 255, 255), thickness=10)
            #cv2.line(img, (986, 2086), (1246, 2230), color=(255, 255, 255), thickness=10)
            #cv2.line(img, (1246, 2230), (1531, 2667), color=(255, 255, 255), thickness=10)
            cv2.imshow("INP", cv2.resize(img, (320,240)))
            cv2.waitKey(1)
            _undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            cv2.imshow("", _undistorted_img)
            cv2.waitKey(1)
            cv2.imwrite("/home/mujung/PycharmProjects/bikeimage/%s.%d.jpg" %(img_info,framenum), _undistorted_img)

            print("Done")
