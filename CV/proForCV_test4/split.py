import cv2
import numpy as np
imgpath='img.png'


#sobel
img = cv2.imread(imgpath, 0)

x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

absX = cv2.convertScaleAbs(x)

absY = cv2.convertScaleAbs(y)

dst1 = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

cv2.imshow("Result", dst1)
cv2.imwrite('sobel.png',dst1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()




#Roberts

img = cv2.imread(imgpath)

grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
kernely = np.array([[0, -1], [1, 0]], dtype=int)
x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)

absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
dst2 = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

cv2.imshow("Result2", dst2)
cv2.imwrite('Roberts.png',dst2)
#cv2.waitKey(0)




#prewitt

img = cv2.imread(imgpath)



grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)

absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
dst3 = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

cv2.imshow("Result3", dst3)
cv2.imwrite('prewitt.png',dst3)
#cv2.waitKey(0)


#log

img = cv2.imread(imgpath)

grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

dst = cv2.Laplacian(grayImage, cv2.CV_16S, ksize=3)
dst4 = cv2.convertScaleAbs(dst)

cv2.imshow("Result4", dst4)
cv2.imwrite('log.png',dst4)
#cv2.waitKey(0)

#canny
img = cv2.imread(imgpath, 0)
img = cv2.GaussianBlur(img, (3, 3), 0)
dst5 = cv2.Canny(img, 50, 150)


cv2.imshow("Result5", dst5)
cv2.imwrite('canddy.png',dst5)
cv2.waitKey(0)