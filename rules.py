"""
EAS 510 - Expert System Rules
"""
import os
import cv2


from PIL import Image

def rule1_metadata(target_info, input_path):
    """Rule 1: Compare file sizes and dimensions."""
    input_size = os.path.getsize(input_path)
    with Image.open(input_path) as img:
        input_width, input_height = img.size
        input_mode = img.mode

    target_size = target_info['size']
    target_width = target_info['width']
    target_height = target_info['height']
    target_mode = target_info['mode']

    score = 0

    # Size comparison
    if target_size > 0 and input_size > 0:
        size_ratio = min(input_size, target_size) / max(input_size, target_size)
        score += int(size_ratio * 10)
    else:
        size_ratio = 0

    # Dimension comparison
    if target_width > 0 and target_height > 0:
        width_ratio = min(input_width, target_width) / max(input_width, target_width)
        height_ratio = min(input_height, target_height) / max(input_height, target_height)
        dim_ratio = (width_ratio + height_ratio) / 2
        score += int(dim_ratio * 10)
    else:
        dim_ratio = 0

    # Mode comparison
    if input_mode == target_mode:
        score += 10

    fired = score > 10
    evidence = f"Size {size_ratio:.2f}, Dim {dim_ratio:.2f}, Mode match: {input_mode == target_mode}"


    # Calculate ratio (0.0 to 1.0)
    if target_size > 0 and input_size > 0:
        ratio = min(input_size, target_size) / max(input_size, target_size)
    else:
        ratio = 0

    # Convert to score (0-30 points)
    score = int(ratio * 30)
    fired = ratio > 0.3
    evidence = f"Size ratio {ratio:.2f}"

    return score, fired, evidence

def rule2_histogram(target_info, input_path):
    """Rule 2: Compare color histograms."""
    # Load both images
    target_img = cv2.imread(target_info['path'])
    input_img = cv2.imread(input_path)

    # Check if images loaded successfully
    if target_img is None or input_img is None:
        return 0, False, "Could not load images"

    # Calculate histograms for each color channel (B, G, R)
    # Parameters: [image], [channel], mask, [bins], [range]
    target_hist = cv2.calcHist([target_img], [0, 1, 2], None,
                                [8, 8, 8], [0, 256, 0, 256, 0, 256])
    input_hist = cv2.calcHist([input_img], [0, 1, 2], None,
                               [8, 8, 8], [0, 256, 0, 256, 0, 256])

    # Normalize histograms (scale to 0-1)
    cv2.normalize(target_hist, target_hist)
    cv2.normalize(input_hist, input_hist)

    # Compare histograms using correlation method
    # Returns value between -1 (opposite) and 1 (identical)
    similarity = cv2.compareHist(target_hist, input_hist, cv2.HISTCMP_CORREL)

    # Convert to score (0-30 points)
    # similarity ranges from -1 to 1, we map 0-1 to 0-30
    score = int(max(0, similarity) * 30)
    fired = similarity > 0.5
    evidence = f"Histogram correlation {similarity:.3f}"

    return score, fired, evidence