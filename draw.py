import cv2
import numpy as np

def draw_bounding_boxes(image, bounding_boxes, color=(0, 0, 255)):
    for ((start_x, start_y), (end_x, end_y)) in bounding_boxes:
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color, 4)

    return image
