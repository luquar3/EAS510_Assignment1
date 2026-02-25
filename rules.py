"""
EAS 510 - Expert System Rules
"""
import os
import cv2
import numpy as np

from PIL import Image

def rule1_metadata(target_info, input_path):
    """Rule 1: Compare file sizes and dimensions (and basic properties)."""
    input_size = os.path.getsize(input_path)
    with Image.open(input_path) as img:
        input_width, input_height = img.size
        input_mode = img.mode

    target_size = target_info['size']
    target_width = target_info['width']
    target_height = target_info['height']
    target_mode = target_info['mode']

    score = 0

    # Size comparison (0-10)
    if target_size > 0 and input_size > 0:
        size_ratio = min(input_size, target_size) / max(input_size, target_size)
        score += int(size_ratio * 10)
    else:
        size_ratio = 0.0

    # Dimension comparison (0-10)
    if target_width > 0 and target_height > 0:
        width_ratio = min(input_width, target_width) / max(input_width, target_width)
        height_ratio = min(input_height, target_height) / max(input_height, target_height)
        dim_ratio = (width_ratio + height_ratio) / 2
        score += int(dim_ratio * 10)
    else:
        dim_ratio = 0.0

    # Mode match (0 or 10)
    mode_match = (input_mode == target_mode)
    if mode_match:
        score += 10

    fired = score >= 15  
    evidence = f"Size ratio {size_ratio:.2f}"

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
    evidence = f"Correlation {similarity:.2f}"

    return score, fired, evidence

def rule3_template(target_info, input_path):
    """Rule 3: Template Matching (crops)"""

    # Load grayscale images
    target = cv2.imread(target_info["path"], cv2.IMREAD_GRAYSCALE)
    inp = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if target is None or inp is None:
        return 0, False, "Could not load images"

    th, tw = target.shape[:2]
    ih, iw = inp.shape[:2]

    # Template must be smaller than the image being searched
    # We'll search the smaller inside the larger.
    if (th * tw) <= (ih * iw):
        big = inp
        template = target
    else:
        big = target
        template = inp

    bh, bw = big.shape[:2]
    th, tw = template.shape[:2]

    # If still not valid, bail out
    if th > bh or tw > bw:
        return 0, False, "Template larger than target"

    # Do template matching
    result = cv2.matchTemplate(big, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)  # max_val is best similarity (0..1)

    # Score: 0-40
    score = int(max(0.0, min(1.0, max_val)) * 40)
    fired = max_val > 0.50  # you can tune this threshold
    evidence = f"Match score {max_val:.2f}"

    return score, fired, evidence

def rule4_orb(target_info, input_path):
    """Rule 4: ORB keypoint matching (more robust to resize/rotation/crop)."""
    target = cv2.imread(target_info["path"], cv2.IMREAD_GRAYSCALE)
    inp = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if target is None or inp is None:
        return 0, False, "Could not load images"

    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(target, None)
    kp2, des2 = orb.detectAndCompute(inp, None)

    if des1 is None or des2 is None:
        return 0, False, "No descriptors"

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if not matches:
        return 0, False, "0 matches"

    # Sort matches by distance (lower is better)
    matches = sorted(matches, key=lambda m: m.distance)

    good = [m for m in matches if m.distance < 60]  # threshold you can tune
    good_count = len(good)

    # Convert count into 0-40 score (cap at 40)
    score = min(40, int(good_count / 25 * 40))  # 25 good matches => ~40 points
    fired = good_count >= 10
    evidence = f"Good matches {good_count}"

    return score, fired, evidence