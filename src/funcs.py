import yaml
import cv2
import numpy as np

def bb_intersection_over_union(boxA: list[int], boxB: list[int]) -> float:
    """
    SOURCE: https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    Compute the intersection over union of two bounding boxes.
    The bounding boxes are provided as [x, y, x2, y2] where the bounding box coordinates
    are x1, y1, x2, y2, where the left top and right bottom coordinates are x1, y1 and x2, y2.
    Args:
        boxA: Bounding box A [x, y, x2, y2]
        boxB: Bounding box B [x, y, x2, y2]
    Returns:
        float: The intersection over union value.
    """

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def load_settings(setting_file: str) -> dict:
    """
    Load the settings from the yaml file
    Args:
        setting_file: The path to the yaml file
    Returns:
        dict: The settings dictionary
    """
    with open(setting_file, "r") as f:
        settings = yaml.safe_load(f)
    return settings

def draw_detected_object(frame: np.ndarray, 
                          bbox: list[int],
                          detected_class: str = None,
                          color: tuple[int, int, int] = (0, 255, 0)) -> None:
    """
    Draw the detected object on the frame
    Args:
        frame: The frame to draw on
        bbox: The bounding box coordinates [x, y, w, h]
        detected_class: The class of the detected object
        color: The color of the bounding box
    Returns:
        None
    """

    x, y, x2, y2 = bbox

    cv2.rectangle(frame, (x, y), (x2, y2), color, 2)

    if detected_class is not None:
        cv2.putText(frame, f"{detected_class}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
def draw_stats_dict(
    img: np.ndarray,
    stats_dict: dict,
    title: str = "",
    color: tuple[int, int, int] = (255, 255, 255),
    x: int = 50,
    y: int = 50
) -> np.ndarray:
    """
    Draws a dictionary of statistics on an image using OpenCV putText.

    Args:
        img: The image (numpy array) to draw on.
        stats_dict: Dictionary containing key-value pairs to display.
        title: Optional title to display above the stats.
        x: X-coordinate for the starting position of the text.
        y: Y-coordinate for the starting position of the text.

    Returns:
        np.ndarray: The image with the statistics drawn on it.
    """
    delta_y = 30
    cv2.putText(img, title, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    # draw a line to highlight the title from stats
    cv2.line(img, (x, y+10), (x + 200, y + 10), (200, 200, 200), 2)
    y += delta_y
    for key, value in stats_dict.items():
        cv2.putText(img, f"{key}: {value}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += delta_y
    return img