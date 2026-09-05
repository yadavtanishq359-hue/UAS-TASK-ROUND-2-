import cv2 as cv
import numpy as np

# 1. Image read
image = cv.imread(r"C:/Python/UAS PROJECT/input/IMAGE5.jpg")

hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

# masking of green black blue
lower_green = np.array([30, 40, 20])
upper_green = np.array([80, 255, 255])
green_mask = cv.inRange(hsv, lower_green, upper_green)

lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 30])
black_mask = cv.inRange(hsv, lower_black, upper_black)

lower_blue = np.array([125, 50, 50])
upper_blue = np.array([150, 255, 255])
blue_mask = cv.inRange(hsv, lower_blue, upper_blue)

lower_orange = np.array([5, 100, 100])
upper_orange = np.array([25, 255, 255])
orange_mask = cv.inRange(hsv, lower_orange, upper_orange)

# ---- FIX: purple ka range tight kiya (140-160), taaki blue/water
# ellipses (jinka hue ~130 hai) ismein na aa jaye ----
lower_purple = np.array([140, 50, 50])
upper_purple = np.array([160, 255, 255])
purple_mask = cv.inRange(hsv, lower_purple, upper_purple)

unwanted = cv.bitwise_or(black_mask, blue_mask)
unwanted = cv.bitwise_or(unwanted, green_mask)
wanted = cv.bitwise_not(unwanted)

# 3. Terrain ke masks
lower_light_green = np.array([35, 100, 160])
upper_light_green = np.array([80, 255, 210])
light_green_mask = cv.inRange(hsv, lower_light_green, upper_light_green)

lower_mid_green = np.array([35, 150, 40])
upper_mid_green = np.array([80, 255, 170])
mid_green_mask = cv.inRange(hsv, lower_mid_green, upper_mid_green)

lower_dark_green = np.array([43, 200, 50])
upper_dark_green = np.array([100, 255, 82])
dark_green_mask = cv.inRange(hsv, lower_dark_green, upper_dark_green)

# 4. Casualty ka level find karne ka function
def get_level(x, y, w, h):

    points = [(x + w // 2, y - 5),
    (x + w // 2, y + h + 5),
    (x - 5, y + h // 2),
    (x + w + 5, y + h // 2)]

    points = [(dx, dy)
    for dx, dy in points
    if 0 <= dx < image.shape[1] and 0 <= dy < image.shape[0] ]

    for dx, dy in points:

        if dark_green_mask[dy, dx] == 255:
            return 3

        if mid_green_mask[dy, dx] == 255:
            return 2

        if light_green_mask[dy, dx] == 255:
            return 1

    return 0


# 5. Casualties detect karo
contours, _ = cv.findContours(wanted, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

output = image.copy()

# ---- START point (orange triangle) - sirf coordinate ke liye, casualty nahi ----
orange_contours, _ = cv.findContours(orange_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

for contour in orange_contours:
    area = cv.contourArea(contour)
    if area < 2000:            # ---- FIX: chhota noise (yellow shapes ke edges) skip ----
        continue

    M = cv.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        start_point = (cx, cy)

        cv.putText(output, "START", (cx, cy), cv.FONT_ITALIC, 0.7, (0, 140, 255), 2)
        print("START point:", start_point)

# ---- END point (purple triangle) - sirf coordinate ke liye, casualty nahi ----
purple_contours, _ = cv.findContours(purple_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

for contour in purple_contours:
    area = cv.contourArea(contour)
    if area < 300:            # ---- FIX: chhota noise skip ----
        continue

    M = cv.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        end_point = (cx, cy)

        cv.putText(output, "END", (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        print("END point:", end_point)


casualties = []

# 7. Har casualty ke liye
for contour in contours:

    area = cv.contourArea(contour)

    if area < 300:
        continue

    perimeter = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)

    if len(approx) == 3:
        continue          # this wont detect traingle as it is not a casualty

    x, y, w, h = cv.boundingRect(contour)
    cx = x + w // 2
    cy = y + h // 2

    level = get_level(x, y, w, h)

    casualties.append((cx, cy, level))

    cv.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.circle(output, (cx, cy), 4, (0, 0, 255), -1)
    cv.putText(output, f"({cx},{cy}) L{level}", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


# 8. Results print
print("Total Casualties:", len(casualties))

for i, (x, y, level) in enumerate(casualties, 1):
    print(f"Casualty {i}: Coordinates = ({x}, {y}), Level = {level}")


# 9. Show image
cv.imshow("Casualties", output)
cv.waitKey(0)
cv.destroyAllWindows()