"""
EAS 510 - System Test Runner
Runs the forensic system on required folders and produces output logs.
"""

from forensics_detective import SimpleDetective
import os


def run_folder(folder, detective):
    """Run the system on every image in a folder."""
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder, filename)
            detective.find_best_match(path)


if __name__ == "__main__":
    print("=" * 50)
    print("SimpleDetective - System Evaluation")
    print("=" * 50)

    # Initialize system
    detective = SimpleDetective()
    detective.register_targets("originals")

    print("\n" + "=" * 50)
    print("RUNNING EASY CASES (modified_images)")
    print("=" * 50)
    run_folder("modified_images", detective)

    print("\n" + "=" * 50)
    print("RUNNING HARD CASES (hard)")
    print("=" * 50)
    run_folder("hard", detective)

    print("\n" + "=" * 50)
    print("RUNNING RANDOM CASES (random)")
    print("=" * 50)
    run_folder("random", detective)

    print("\n" + "=" * 50)
    print("SYSTEM RUN COMPLETE")
    print("=" * 50)