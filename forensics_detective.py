"""
EAS 510 - Digital Forensics Detective
"""
import os
from PIL import Image 
from rules import rule1_metadata, rule2_histogram, rule3_template, rule4_orb

class SimpleDetective:
    """An expert system that matches modified images to originals."""

    def __init__(self):
        self.targets = {}  # filename -> signature

    def register_targets(self, folder):
        """Load original images and compute signatures."""
        print(f"Loading targets from: {folder}")

        for filename in os.listdir(folder):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(folder, filename)
                file_size = os.path.getsize(filepath)
                with Image.open(filepath) as img:
                    width, height = img.size
                    mode = img.mode
                    fmt = img.format

                self.targets[filename] = {
                    'path': filepath,
                    'size': file_size,
                    'width': width,
                    'height': height,
                    'mode': mode,
                    'format': fmt
                    }
                print(f"  Registered: {filename} ({file_size} bytes)")

        print(f"Total targets: {len(self.targets)}")

    def find_best_match(self, input_image_path):
        filename = os.path.basename(input_image_path)
        print(f"Processing: {filename}")
        results = []

        for target_name, target_info in self.targets.items():
            # Apply Rule 1: Metadata
            score1, fired1, evidence1 = rule1_metadata(target_info, input_image_path)

            # Apply Rule 2: Histogram
            score2, fired2, evidence2 = rule2_histogram(target_info, input_image_path)

            # Apply Rule 3: Template
            score3, fired3, evidence3 = rule3_template(target_info, input_image_path)

            # Apply Rule 4: Edges
            score4, fired4, evidence4 = rule4_orb(target_info, input_image_path)

            # Combine scores
            total_score = score1 + score2 + score3 + score4
            max_possible = 140
            results.append({
                'target': target_name,
                'score': total_score,
                'max_score': max_possible,
                'rules': [(fired1, evidence1, score1), (fired2, evidence2, score2), (fired3, evidence3, score3), (fired4, evidence4, score4)]
            })

        # Sort by score, highest first
        results.sort(key=lambda x: x['score'], reverse=True)
        best = results[0]
    
        r1, r2, r3, r4 = best["rules"]
        
        # Exact required format
        print(f"Rule 1 (Metadata):  {'FIRED' if r1[0] else 'NO MATCH'} - {r1[1]} -> {r1[2]}/30 points")
        print(f"Rule 2 (Histogram): {'FIRED' if r2[0] else 'NO MATCH'} - {r2[1]} -> {r2[2]}/30 points")
        print(f"Rule 3 (Template):  {'FIRED' if r3[0] else 'NO MATCH'} - {r3[1]} -> {r3[2]}/40 points")
        print(f"Rule 4 (ORB):     {'FIRED' if r4[0] else 'NO MATCH'} - {r4[1]} -> {r4[2]}/40 points")

        # Decide match vs rejected
        if best['score'] >= 84:  
            print(f"Final Score: {best['score']}/{best['max_score']} -> MATCH to {best['target']}")
            return {'best_match': best['target'], 'confidence': best['score']}
        else:
            print(f"Final Score: {best['score']}/{best['max_score']} -> REJECTED")
            return {'best_match': None, 'confidence': best['score']}


if __name__ == "__main__":
    def run_folder(folder, detective):
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                detective.find_best_match(os.path.join(folder, f))

    print("=" * 50)
    print("SimpleDetective - Prototype v0.1")
    print("=" * 50)

    detective = SimpleDetective()
    detective.register_targets("originals")

    print("\n" + "=" * 50)
    print("TESTING")
    print("=" * 50)

    test_images = [
        "modified_images/modified_00_bright_enhanced.jpg",
        "modified_images/modified_03_compressed.jpg",
    ]
    # Run full folders (this is what you need for results_v1.txt)
    run_folder("hard", detective)

    print("\n" + "=" * 50)
    print("PROTOTYPE COMPLETE!")
    print("=" * 50)