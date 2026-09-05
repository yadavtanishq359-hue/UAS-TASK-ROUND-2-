import cv2 as cv
import numpy as np
import heapq

# ---------- Image load ----------
image = cv.imread(r"C:/Python/UAS PROJECT/input/IMAGE5.jpg")
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
output = image.copy()
img_h, img_w = image.shape[:2]

# ---------- Unwanted cheezein (background, black bar, water) ----------
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
fg_mask = cv.bitwise_not(unwanted)

# ---------- Terrain ke 3 levels (sirf display ke liye) ----------
lower_light_green = np.array([35, 100, 160])
upper_light_green = np.array([80, 255, 210])
light_green_mask = cv.inRange(hsv, lower_light_green, upper_light_green)

lower_mid_green = np.array([35, 150, 40])
upper_mid_green = np.array([80, 255, 170])
mid_green_mask = cv.inRange(hsv, lower_mid_green, upper_mid_green)

lower_dark_green = np.array([43, 200, 50])
upper_dark_green = np.array([100, 255, 82])
dark_green_mask = cv.inRange(hsv, lower_dark_green, upper_dark_green)

# ---------- Start/End markers ----------
lower_orange = np.array([5, 100, 100])
upper_orange = np.array([25, 255, 255])
orange_mask = cv.inRange(hsv, lower_orange, upper_orange)

lower_purple = np.array([140, 50, 50])
upper_purple = np.array([160, 255, 255])
purple_mask = cv.inRange(hsv, lower_purple, upper_purple)

# ---------- Score tables ----------
SHAPE_SCORE = {"Circle": 3, "Star": 1, "Square": 2}
COLOR_SCORE = {"Red": 3, "Yellow": 2, "White": 1}


# ---------- Level nikalne ka function ----------
def get_level(x, y, w, h):
    px1, py1 = x + w // 2, y - 5
    px2, py2 = x + w // 2, y + h + 5
    px3, py3 = x - 5, y + h // 2
    px4, py4 = x + w + 5, y + h // 2

    points = [(px1, py1), (px2, py2), (px3, py3), (px4, py4)]

    for px, py in points:
        if px < 0 or py < 0 or px >= img_w or py >= img_h:
            continue
        if dark_green_mask[py, px] == 255:
            return 3
        if mid_green_mask[py, px] == 255:
            return 2
        if light_green_mask[py, px] == 255:
            return 1

    return 0


# ---------- Shape pehchanne ka function ----------
def get_shape(approx, area, perimeter):
    corners = len(approx)
    circularity = (4 * np.pi * area) / (perimeter * perimeter)

    if corners == 4 and circularity > 0.7:
        return "Square"
    elif circularity < 0.5:
        return "Star"
    else:
        return "Circle"


# ---------- Color pehchanne ka function ----------
def get_color(contour):
    mask = np.zeros(image.shape[:2], np.uint8)
    cv.drawContours(mask, [contour], -1, 255, -1)
    b, g, r, _ = cv.mean(image, mask=mask)

    if r > 180 and g < 100 and b < 100:
        return "Red"
    if r > 180 and g > 180 and b < 130:
        return "Yellow"
    if r > 180 and g > 180 and b > 180:
        return "White"
    return None


# ============================================================
# DIJKSTRA - shortest path nikalne ke liye (obstacles avoid karke)
# ============================================================

# black bars ko obstacle bana diya, thoda dilate kiya safety margin ke liye
kernel = np.ones((5, 5), np.uint8)
obstacle_mask = cv.dilate(black_mask, kernel, iterations=2)

STEP = 10   # har 10 pixel ka ek grid-node (poore image pe Dijkstra chalana slow hota)
grid_h = img_h // STEP
grid_w = img_w // STEP

blocked = np.zeros((grid_h, grid_w), dtype=bool)
for gy in range(grid_h):
    for gx in range(grid_w):
        if obstacle_mask[gy * STEP, gx * STEP] == 255:
            blocked[gy, gx] = True


def to_grid(px, py):
    gy, gx = py // STEP, px // STEP
    gy, gx = min(gy, grid_h - 1), min(gx, grid_w - 1)
    return (gy, gx)


def to_pixel(gy, gx):
    return (gx * STEP + STEP // 2, gy * STEP + STEP // 2)


def dijkstra(start_grid, end_grid):
    dist = np.full((grid_h, grid_w), np.inf)
    dist[start_grid] = 0
    visited = np.zeros((grid_h, grid_w), dtype=bool)
    prev = {}
    pq = [(0, start_grid)]

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while pq:
        d, (y, x) = heapq.heappop(pq)
        if visited[y, x]:
            continue
        visited[y, x] = True
        if (y, x) == end_grid:
            break

        for dy, dx in moves:
            ny, nx = y + dy, x + dx
            if 0 <= ny < grid_h and 0 <= nx < grid_w and not blocked[ny, nx] and not visited[ny, nx]:
                cost = 1.4142 if dy != 0 and dx != 0 else 1.0
                new_dist = d + cost
                if new_dist < dist[ny, nx]:
                    dist[ny, nx] = new_dist
                    prev[(ny, nx)] = (y, x)
                    heapq.heappush(pq, (new_dist, (ny, nx)))

    # path ko wapas trace karo (end se start tak, phir reverse)
    path = []
    node = end_grid
    while node in prev:
        path.append(node)
        node = prev[node]
    path.append(start_grid)
    path.reverse()
    return path


# ---------- START point (orange triangle) ----------
start_point = None
orange_contours, _ = cv.findContours(orange_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
for c in orange_contours:
    if cv.contourArea(c) < 2000:
        continue
    M = cv.moments(c)
    sx, sy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    start_point = (sx, sy)
    print("START point:", start_point)
    cv.putText(output, "START", (sx - 30, sy - 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

# ---------- END point (purple triangle) ----------
end_point = None
purple_contours, _ = cv.findContours(purple_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
for c in purple_contours:
    if cv.contourArea(c) < 300:
        continue
    M = cv.moments(c)
    ex, ey = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    end_point = (ex, ey)
    print("END point:", end_point)
    cv.putText(output, "END", (ex - 20, ey - 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


# ---------- Casualties dhundo ----------
contours, _ = cv.findContours(fg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

casualties = []

for contour in contours:
    area = cv.contourArea(contour)
    if area < 300:
        continue

    perimeter = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)

    if len(approx) == 3:
        continue   # triangle hai, casualty nahi

    color = get_color(contour)
    if color is None:
        continue

    shape = get_shape(approx, area, perimeter)

    x, y, w, h = cv.boundingRect(contour)
    cx, cy = x + w // 2, y + h // 2
    level = get_level(x, y, w, h)

    severity = SHAPE_SCORE[shape] * COLOR_SCORE[color]

    casualties.append([cx, cy, shape, color, level, severity, (x, y, w, h)])


# ---------- Severity ke hisaab se sort karo (sabse zyada severe pehle) ----------
casualties.sort(key=lambda c: c[5], reverse=True)


# ---------- Boxes + rank draw karo ----------
rank = 1
for c in casualties:
    cx, cy, shape, color, level, severity, box = c
    x, y, w, h = box
    cv.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.putText(output, f"#{rank} L{level}", (x, y - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    rank += 1


# ============================================================
# START -> Casualty1 -> Casualty2 -> ... -> END ka path Dijkstra se
# ============================================================
waypoints = [start_point] + [(c[0], c[1]) for c in casualties] + [end_point]

for i in range(len(waypoints) - 1):
    g1 = to_grid(*waypoints[i])
    g2 = to_grid(*waypoints[i + 1])
    grid_path = dijkstra(g1, g2)
    pixel_path = [to_pixel(gy, gx) for gy, gx in grid_path]

    for j in range(len(pixel_path) - 1):
        cv.line(output, pixel_path[j], pixel_path[j + 1], (255, 0, 0), 2)


# ---------- Results print karo ----------
print("\nTotal Casualties:", len(casualties))
print("\nRover ka route (Severity ke hisaab se rank, phir Dijkstra se path):")

rank = 1
for c in casualties:
    cx, cy, shape, color, level, severity, box = c
    print(f"Rank {rank}: Coordinates=({cx},{cy})  Shape={shape}  Color={color}  "
          f"Severity={severity}  Level={level}")
    rank += 1


cv.imshow("Casualties with Rover Path", output)
cv.waitKey(0)
cv.destroyAllWindows()