from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import math
def color_matches(pixel, target_color, tolerance=15):
    return all(abs(pixel[i] - target_color[i]) <= tolerance for i in range(3))

def calculate_area_and_draw_shape(square_image_path, pure_color, real_edge_length_mm):
    image = Image.open(square_image_path)
    image_np = np.array(image)
    width, height = image.size
    dpi = width / real_edge_length_mm

    points = []
    for y in range(height):
        for x in range(width):
            if color_matches(image_np[y, x][:3], pure_color):
                points.append((x, y))

    if not points:
        print("未找到任何匹配的纯色像素！")
        return 0, None, 0

    if len(points) >= 3:
        hull = ConvexHull(points)
        hull_points = [(points[i][0], points[i][1]) for i in hull.vertices]
    else:
        hull_points = points

    hull_area = ConvexHull(points).volume / (dpi ** 2)

    max_distance = 0
    point1, point2 = None, None
    for i in range(len(hull_points)):
        for j in range(i + 1, len(hull_points)):
            distance = math.sqrt((hull_points[i][0] - hull_points[j][0]) ** 2 +
                                 (hull_points[i][1] - hull_points[j][1]) ** 2)
            if distance > max_distance:
                max_distance = distance
                point1, point2 = hull_points[i], hull_points[j]

    max_distance_mm = max_distance / dpi

    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    draw.polygon(hull_points, outline="red", fill=None)
    if point1 and point2:
        draw.line([point1, point2], fill="blue", width=2)

    # 显示图像
    plt.imshow(draw_image)
    plt.axis('off')
    plt.show()

    return hull_area, draw_image, max_distance_mm


square_image_path = 'd:\\user\\tc029861\\桌面\\面积改.png'


pure_color = (255, 255, 0)


real_edge_length_mm = 9
area, drawn_image, max_line_length = calculate_area_and_draw_shape(square_image_path, pure_color, real_edge_length_mm)
print("不规则形状的面积:", area, "平方毫米")
print("最长线段的长度:", max_line_length, "毫米")
print("概念直径:", (area/math.pi)**(1/2)*2, "毫米")
