# Pipette Angle Detector

Measure the angle of a pipette tip from horizontal using segmentation masks.

---

## What It Does

Input: Image + segmentation mask → Output: Angle of the pipette tip

---

## How I Got the Masks

I had only **9 images** from different angles, distances, and lighting.

1. Uploaded images to [Roboflow](https://roboflow.com)
2. Used **Smart Polygon** tool (auto-annotation with pre-trained model)
3. Made small fixes where needed
4. Exported as **YOLOv8 format** → Got the `.txt` label files

No manual drawing needed - Roboflow's pre-trained model did most of the work.

---

## How It Works
video: https://www.loom.com/share/2e5cdac43adc4567a5c98d7f70cbce22
### Step 1: Create Mask
Convert polygon annotation to binary mask.

### Step 2: Find Skeleton
The skeleton is the centerline - runs through the middle and stays inside the shape.

### Step 3: Find Tip and Handle
Trace the skeleton path and measure width at both ends:
- **Wide end** = Handle (P1)
- **Narrow end** = Tip (P2)

### Step 4: Calculate Angle
Fit a line through the **last 30%** of the path (near tip) to get the angle.

**Why trace the whole path if we only use 30%?**
- We need the full path to find both ends
- We measure width to know which end is the tip
- Then we use the tip portion for the angle (in case the pipette curves)

---

## Why Not Keypoints?

- Tip is often hidden (inside tubes, behind hands)
- Different pipette models look different
- Different camera angles
- No consistent landmarks like faces have

**Segmentation works** - just outline whatever is visible.

---

## Why Not Detect the Tube?

- Tubes are vertical - don't show pipette angle
- Well plates have 96 tubes - which one?
- The pipette shape already contains the angle info

---

## Why Not Hand Detection?

- Hand grip varies - doesn't match pipette angle
- Lab gloves make detection harder
- Why add complexity? Measure the pipette directly.

---

## Installation

```bash
pip install opencv-python numpy scikit-image
```

---

## How to Run

### Step 1: Set Up Your Folders

### Step 2: Edit the Script

Open `pipette_angle_detector.py` and find this section at the bottom:

```python
if __name__ == "__main__":
    process_dataset(
        image_folder="./images",      # ← change to your images folder
        label_folder="./labels",      # ← change to your labels folder
        output_folder="./output",     # ← change to your output folder
        csv_path="./results.csv"      # ← change or remove if you don't want CSV
    )
```

### Step 3: Run

```bash
python pipette_angle_detector.py
```

### What You Get

- Annotated images in `output/` folder showing:
  - Yellow outline (mask)
  - Green line (centerline)
  - Green dot P1 (handle)
  - Red dot P2 (tip)
  - Angle value
- `results.csv` with all angles

---

## Input Format

**Image:** `photo.jpg`

**Label:** `photo.txt` (same name, different extension)

Label file format:
```
0 0.45 0.12 0.48 0.15 0.52 0.35 0.50 0.58
```
- First number = class (0)
- Rest = x,y pairs (normalized 0-1)

---

## FAQ

**Q: What if my pipette is straight?**

A: Works fine. The last 30% gives the same angle as the whole thing.

**Q: What if the tip is hidden?**

A: It measures the angle of whatever part is visible and annotated.

---

## License

MIT - Free to use and modify.
