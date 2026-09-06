import cv2 as cv
import numpy as np
import os

# OPENING OF IMAGE
image = cv.imread(r"C:/Python/UAS PROJECT/input/IMAGE5.jpg")

# BLUE GREEN RED INTO HUE SATURATION VALUE
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

# masking of black bars
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 30])
black_mask = cv.inRange(image, lower_black, upper_black)

# masking of blue ellipse
lower_blue = np.array([200, 0, 60])
upper_blue = np.array([255, 60, 130])
blue_mask = cv.inRange(image, lower_blue, upper_blue)

mask = cv.bitwise_not(cv.bitwise_or(black_mask, blue_mask))

output_folder = r"C:/Python/UAS PROJECT/Output masking"

output_path = os.path.join(output_folder, "IMAGE5.jpg")

success = cv.imwrite(output_path, mask)

cv.imshow("Result", mask)
cv.waitKey(0)
cv.destroyAllWindows()

 
