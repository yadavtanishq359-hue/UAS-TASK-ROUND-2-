import cv2 as cv
import numpy as np

# 1. Image read
image = cv.imread(r"C:/Python/UAS PROJECT/input/IMAGE3.jpg")

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

unwanted = cv.bitwise_or(black_mask, blue_mask)
unwanted = cv.bitwise_or(unwanted, green_mask)
wanted = cv.bitwise_not(unwanted)


# 3. Terrain ke masks
lower_light_green = np.array([35, 100, 160])
upper_light_green = np.array([80, 255, 210])
light_green_mask = cv.inRange(hsv, lower_light_green, upper_light_green)

lower_mid_green = np.array([35, 150, 40])
upper_mid_green =np.array([80, 255, 170])
mid_green_mask = cv.inRange(hsv, lower_mid_green, upper_mid_green)

lower_dark_green = np.array([43, 200, 50])
upper_dark_green = np.array([100, 255, 82])
dark_green_mask = cv.inRange(hsv, lower_dark_green, upper_dark_green)

# 4. Casualty ka level find karne ka function
def get_level(x, y, w, h):

    area = dark_green_mask[y-10:y+10, x-10:x+10]
    if np.count_nonzero(area) > 0:
        return 3

    # Centre ke aas-paas 20 pixel ka area
    area = mid_green_mask[y-10:y+10, x-10:x+10]

    # Agar dark green mila
    if np.count_nonzero(area) > 0:
        return 2

    area = light_green_mask[y-10:y+10, x-10:x+10]

    # Agar light green mila
    if np.count_nonzero(area) > 0:
        return 1

    # Otherwise normal terrain
    return 0


# 5. Casualties detect karo
contours, _ = cv.findContours(
    wanted,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE
)


# 6. Output image
output = image.copy()

casualties = []


# 7. Har casualty ke liye
for contour in contours:

    area = cv.contourArea(contour)

    # Chhoti cheezein ignore
    if area < 300:
        continue

    # Triangle ignore
    perimeter = cv.arcLength(contour, True)

    approx = cv.approxPolyDP(
        contour,
        0.02 * perimeter,
        True
    )

    if len(approx) == 3:
        continue

    # Box aur centre
    x, y, w, h = cv.boundingRect(contour)

    cx = x + w // 2
    cy = y + h // 2

    # Terrain level
    level = get_level(cx, cy)

    # Data save
    casualties.append((cx, cy, level))

    # Box draw
    cv.rectangle(
        output,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    # Centre
    cv.circle(
        output,
        (cx, cy),
        4,
        (0, 0, 255),
        -1
    )

    # Image par text
    cv.putText(
        output,
        f"({cx},{cy}) L{level}",
        (x, y - 10),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1
    )


# 8. Results print
print("Total Casualties:", len(casualties))

for i, (x, y, level) in enumerate(casualties, 1):

    print(
        f"Casualty {i}: "
        f"Coordinates = ({x}, {y}), "
        f"Level = {level}"
    )


# 9. Show image
cv.imshow("Casualties", output)

cv.waitKey(0)
cv.destroyAllWindows()