#形态学处理
import cv2
import numpy as np
imgpath='img.png'
img=cv2.imread(imgpath,cv2.IMREAD_GRAYSCALE)
#cv2.imwrite('imgGr.png',img)

ret, grethe_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('img',img)
cv2.imshow('img2',grethe_img)
#cv2.imwrite('imgGrE.png',grethe_img)
kernel = np.ones((5, 5), np.uint8)#核大小
res_img = cv2.dilate(grethe_img, kernel)  # 膨胀
cv2.imshow('img3',res_img)
#cv2.imwrite('imgRes.png',res_img)
cv2.waitKey(0)
