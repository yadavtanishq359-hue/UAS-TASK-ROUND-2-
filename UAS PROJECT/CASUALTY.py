import cv2 as cv
import numpy as np

# LOADING OF IMAGE AND BGR TO HSV
image = cv.imread(r"C:/Python/UAS PROJECT/input/IMAGE5.jpg")
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
output = image.copy()

# MASKING OF UNWANTED REGION
lower_green = np.array([30, 40, 20])
upper_green = np.array([80, 255, 255])
green_mask = cv.inRange(hsv, lower_green, upper_green)

lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 30])
black_mask = cv.inRange(hsv, lower_black, upper_black)

lower_blue = np.array([125, 50, 50])
upper_blue = np.array([150, 255, 255])
blue_mask = cv.inRange(hsv, lower_blue, upper_blue)

unwanted = cv.bitwise_or(green_mask, black_mask)
unwanted = cv.bitwise_or(unwanted, blue_mask)
wanted = cv.bitwise_not(unwanted)

# MASKING FOR TERRAIN 
lower_light_green = np.array([35, 100, 160])
upper_light_green = np.array([80, 255, 210])
light_green_mask = cv.inRange(hsv, lower_light_green, upper_light_green)

lower_mid_green = np.array([35, 150, 40])
upper_mid_green = np.array([80, 255, 170])
mid_green_mask = cv.inRange(hsv, lower_mid_green, upper_mid_green)

lower_dark_green = np.array([43, 200, 50])
upper_dark_green = np.array([100, 255, 82])
dark_green_mask = cv.inRange(hsv, lower_dark_green, upper_dark_green)

# MASKING FOR STARTING AND ENDING TRIANGLE
lower_orange = np.array([5, 100, 100])
upper_orange = np.array([25, 255, 255])
orange_mask = cv.inRange(hsv, lower_orange, upper_orange)

lower_purple = np.array([140, 50, 50])
upper_purple = np.array([160, 255, 255])
purple_mask = cv.inRange(hsv, lower_purple, upper_purple)

# DICTIONARIES TO CALCULATE PRIORITY SCORE 
SHAPE_SCORE = {"Circle": 3, "Star": 1, "Square": 2}
COLOR_SCORE = {"Red": 3, "Yellow": 2, "White": 1}

# LEVEL OF TERRAIN
def get_level(x, y, w, h):
    px1, py1 = x + w // 2, y - 5
    px2, py2 = x + w // 2, y + h + 5
    px3, py3 = x - 5, y + h // 2
    px4, py4 = x + w + 5, y + h // 2

    points = [(px1, py1), (px2, py2), (px3, py3), (px4, py4)]

 # TO CHECK POINT IS WITHIN IMAGE AND THEN DECIDE ITS TERRAIN LEVEL 
    for px, py in points:
        if px < 0 or py < 0 or px >= image.shape[1] or py >= image.shape[0]:
            continue
        if dark_green_mask[py, px] == 255:
            return 3
        if mid_green_mask[py, px] == 255:
            return 2
        if light_green_mask[py, px] == 255:
            return 1

    return 0

# TO IDENTIFY SHAPES USING CONTOUR 
def get_shape(approx, area, perimeter):
    corners = len(approx)
    circularity = (4 * np.pi * area) / (perimeter * perimeter)

    if corners == 4 and circularity > 0.7:
        return "Square"
    elif circularity < 0.5:
        return "Star"
    else:
        return "Circle"


# TO IDENTIFY COLOUR OF SHAPE
def get_color(contour):
    mask = np.zeros(image.shape[:2], np.uint8) #FULL IMAGE BLACK
    cv.drawContours(mask, [contour], -1, 255, -1) #WHITE BOUNDARY TO SHAPES 
    b, g, r, _ = cv.mean(image, mask=mask)

    if r > 180 and g < 100 and b < 100:
        return "Red"
    elif r > 180 and g > 180 and b < 130:
        return "Yellow"
    elif r > 180 and g > 180 and b > 180:
        return "White"
    else:
        return None

# START POINT ORANGE TRIANGLE 
orange_contours, _ = cv.findContours(orange_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
for c in orange_contours:
    if cv.contourArea(c) < 2000:
        continue
    M = cv.moments(c)
    sx, sy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    print("START point:", (sx, sy))
    cv.putText(output, "START", (sx - 30, sy - 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

# END POINT PURPLE TRIANGLE
purple_contours, _ = cv.findContours(purple_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
for c in purple_contours:
    if cv.contourArea(c) < 300:
        continue
    M = cv.moments(c)
    ex, ey = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    print("END point:", (ex, ey))
    cv.putText(output, "END", (ex - 20, ey - 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


# SPOTTING CASAULTIES(SHAPES)
contours, _ = cv.findContours(wanted, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

casualties = []

for contour in contours:
    area = cv.contourArea(contour)
    if area < 300: #IGNORING SMALL CONTOURS FOR BETTER SHAPE SPOTTING
        continue

    perimeter = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)

    if len(approx) == 3: #IGNORING TRIANGLE AS IT IS NOT A CASUALTY
        continue   

    color = get_color(contour)
    if color is None:
        continue

    shape = get_shape(approx, area, perimeter)

    x, y, w, h = cv.boundingRect(contour)
    cx, cy = x + w // 2, y + h // 2
    level = get_level(x, y, w, h)

    # SEVERITY SCORE FOR APPROACHING CASUALTY 
    severity = SHAPE_SCORE[shape] * COLOR_SCORE[color]
    casualties.append([cx, cy, shape, color, level, severity, (x, y, w, h)])

# CASUALTY INFO GETTING STORED IN LIST
casualties.sort(key=lambda c: c[5], reverse=True)


# LOOP FOR GIVING RANK TO CASUALTY 
rank = 1
for c in casualties:
    cx, cy, shape, color, level, severity, box = c
    x, y, w, h = box

    cv.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.putText( output, f"#{rank} {shape} L{level}",(x, y - 10),cv.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255),1)
    rank += 1


# PRINTING RESULT
print("\nTotal Casualties:", len(casualties))
print("\nRank (Severity = Shape score x Color score):")

rank = 1
for c in casualties:
    cx, cy, shape, color, level, severity, box = c
    print(f"Rank {rank}: Coordinates=({cx},{cy})  Shape={shape}  Color={color}  "
          f"Severity={severity}  Level={level}")
    rank += 1

cv.imshow("Casualties", output)
cv.waitKey(0)
cv.destroyAllWindows()
