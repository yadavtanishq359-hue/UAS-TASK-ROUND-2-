import cv2 as cv
import numpy as np

# Image load karo
image = cv.imread(r"C:/Python/UAS PROJECT/input/IMAGE5.jpg")

# HSV me convert karo
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

# Green land + green mountain dono hata do (broad range)
lower_green = np.array([30, 40, 20])
upper_green = np.array([80, 255, 255])
green_mask = cv.inRange(hsv, lower_green, upper_green)
nongreen_mask = cv.bitwise_not(green_mask)

# contour of outer region of shape only not inside one 
contours, _ = cv.findContours(nongreen_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

output = image.copy()
casuality_coordinates = []

for cnt in contours:
    area = cv.contourArea(cnt)
    if area < 300:       
 
        continue
    mask_single = np.zeros(image.shape[:2], np.uint8)
    cv.drawContours(mask_single, [cnt], -1, 255, -1)
    b, g, r, _ = cv.mean(image, mask=mask_single)
    brightness = (b + g + r) / 3

    if brightness < 25:
        continue                     #to skip black bars
    if b > 150 and g < 100:
        continue                     #to skip blue water body

    peri = cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) == 3:
        continue                     # triangle hai, skip

    x, y, w, h = cv.boundingRect(approx)
    cx, cy = x + w // 2, y + h // 2     # shape ka center point (x,y)

    casuality_coordinates.append((cx, cy))

    cv.drawContours(output, [approx], -1, (0, 255, 0), 2)
    cv.circle(output, (cx, cy), 4, (0, 0, 255), -1)
    cv.putText(output, f"({cx},{cy})", (cx - 30, cy - 15),
               cv.FONT_ITALIC, 0.4, (0, 0, 255), 1)

print("Total Casualties:", len(casuality_coordinates))
for i, (cx, cy) in enumerate(casuality_coordinates, 1):
    print(f"{i}: x={cx}, y={cy}")

cv.imshow("Detected Casualties", output)
cv.waitKey(0)
cv.destroyAllWindows()