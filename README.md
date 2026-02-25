# EAS 510 Assignment 1: Digital Forensics Apprentice

Dataset for building a rule-based expert system that matches modified images back to their originals.

## Dataset Structure

```
EAS510_Assignment1/
├── originals/        # 10 original JPEG images
├── modified_images/  # 60 "easy" cases (single transformations)
├── hard/             # 60 "hard" cases (combined transformations)
└── random/           # 15 unrelated images (should be rejected)
```

## Transformations

### Easy Cases (modified_images/)
Each original has 6 modifications:
- Brightness enhancement
- JPEG compression
- 25% crop (center)
- 50% crop (center)
- 75% crop (center)
- PNG format conversion

### Hard Cases (hard/)
Each original has 6 challenging modifications:
- **v1**: Off-center crop + compression
- **v2**: Crop + brightness + compression
- **v3**: Resize + compression
- **v4**: Rotation + compression
- **v5**: Contrast + compression
- **v6**: Crop + resize + compression

## Ground Truth

The filename prefix indicates which original each image was derived from:
- `modified_03_brightness.jpg` → `original_03.jpg`
- `original_03__rotate6deg__compress__q50__v4.jpg` → `original_03.jpg`

Images in `random/` are not derived from any original and should be rejected.

## Setup

```bash
pip install pillow opencv-python numpy
```

## Usage

Clone this repository and use the images to build and test your forensic matching system:

```bash
git clone https://github.com/delveccj/EAS510_Assignment1.git
cd EAS510_Assignment1
```

Your system should:
1. Register the 10 original images
2. For each test image, determine which original it came from (or reject it)
3. Display transparent reasoning showing how each rule contributed to the decision

## There were 31 instances where the system could not match an image to an original and 22 of these were for template matching. Thus, most of what V1 struggled with was template matching in instances of cropping and resizing. 

Template matching only works well when images are the same size. When images are resized or cropped and rotated, the pixel positions change, so template matching gives a low score. Because of this, Rule 3 sometimes fails to help the system make the correct decision.

Examples: 
Processing: original_01__resize_scale75__compress__q30__v3.jpg
Rule 1 (Metadata):  FIRED - Size ratio 0.20 -> 21/30 points
Rule 2 (Histogram): FIRED - Correlation 0.99 -> 29/30 points
Rule 3 (Template):  NO MATCH - Match score 0.27 -> 10/40 points

Processing: original_01__crop_keep50__resized__q75__v6.jpg
Rule 1 (Metadata):  FIRED - Size ratio 0.81 -> 28/30 points
Rule 2 (Histogram): FIRED - Correlation 0.83 -> 24/30 points
Rule 3 (Template):  NO MATCH - Match score 0.10 -> 4/40 points
Final Score: 56/100 -> MATCH to original_01.jpg

For rule 4 I chose ORB, because it is a feature matching method that works well for resized and rotated images.

