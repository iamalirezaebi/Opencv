# Vehicle Counting with OpenCV — Traffic Analysis in Laramie

**A computer-vision pipeline that detects moving vehicles in fixed-camera traffic footage and counts the cars heading toward downtown Laramie, using background subtraction, contour analysis and a virtual counting line.**

Built in Python with OpenCV as a video-processing exercise. The two notebooks apply the same pipeline to two different camera views of the same main street, and each one is calibrated for its own scene.

<!-- Add a GIF or screenshot of the annotated output here, e.g.:
![Vehicle counting demo](docs/demo.gif)
-->

---

## The problem

A fixed traffic camera films a main street with traffic in both directions. The task was to:

1. detect every moving vehicle in the video, and
2. count only the vehicles travelling **toward downtown**, ignoring traffic in the other direction,

and show the running count live on the video.

---

## How it works

```
Video frame
   │
   ▼
Crop region of interest (the road only)
   │
   ▼
Background subtraction (MOG2)  →  foreground mask
   │
   ▼
Binary threshold (drop shadows / noise)
   │
   ▼
Contour detection  →  filter by area (> 4000 px²)
   │
   ▼
Bounding box + centroid for each vehicle
   │
   ▼
Centroid enters the counting zone on the downtown-bound lane?  →  count += 1
   │
   ▼
Overlay count and rate on the frame, show frame + mask
```

**1. Region of interest.** Each frame is cropped to the road area, which removes sidewalks, buildings and sky. This cuts false detections and processing time.

**2. Background subtraction.** OpenCV's `BackgroundSubtractorMOG2` (Gaussian mixture model) learns the static background and outputs a mask of moving pixels. History and variance threshold were tuned for this footage (`history=100`, `varThreshold=5`). `KNN` subtraction can be selected as an alternative.

**3. Cleaning the mask.** A binary threshold at 254 keeps only confident foreground pixels. MOG2 marks shadows as grey (127), so this step removes them.

**4. Detection.** External shapes are found with `cv.findContours`, and any contour smaller than 4000 px² is discarded as noise (leaves, pedestrians, compression artefacts). Each remaining contour gets a bounding box and a centroid.

**5. Directional counting.** A virtual line is drawn across the road, and a vehicle is counted when its centroid passes through a narrow zone on the downtown-bound side. Because the zone sits on one lane only, traffic in the opposite direction is not counted. The zone coordinates are calibrated separately for each camera view, which is why there are two notebooks.

**6. Output.** The live video shows bounding boxes, centroids, the counting line, the total number of cars moving toward the centre, and a cars-per-minute figure. A second window shows the foreground mask, which is useful for tuning.

---

## Repository contents

| File | Description |
|---|---|
| `Notebook1.ipynb` | Pipeline applied to camera view 1 (`Traffic_Laramie_1.mp4`) |
| `Notebook2.ipynb` | Same pipeline, recalibrated counting zone for camera view 2 (`Traffic_Laramie_2.mp4`) |

> **Note:** The source videos are not included because of GitHub's file-size limits. To run the notebooks, place your own fixed-camera traffic video in the project folder and change the `--input` default, then recalibrate the ROI and counting zone for your scene.

---

## Tech stack

`Python 3.11` · `OpenCV 4.9` · `NumPy` · `Jupyter Notebook`

Techniques: background subtraction (MOG2 / KNN), binary thresholding, contour detection, bounding boxes and centroids, region-of-interest cropping, line-based counting.

---

## Run it

```bash
git clone https://github.com/iamalirezaebi/Opencv.git
cd Opencv
pip install opencv-python numpy jupyter
jupyter notebook Notebook1.ipynb
```

Put the video file next to the notebook and run all cells. Press **Esc** to close the video windows.

---

## What I learned

- How background subtraction models separate moving objects from a static scene, and how their parameters trade sensitivity against noise
- Why region-of-interest cropping and area filtering matter as much as the detector itself
- That counting logic is scene-specific: the same code needs different calibration for each camera angle
- The limits of a detection-only approach. Without per-object tracking, a car can be missed or counted twice if its centroid jumps across the counting zone between frames, and vehicles that overlap merge into one contour.

## Possible improvements

- Add a centroid tracker with persistent IDs, and count by line crossing direction instead of a fixed zone
- Use morphological opening/closing to separate vehicles that merge in the mask
- Replace background subtraction with a pretrained detector (e.g. YOLO) for robustness to lighting changes and stopped traffic
- Move the pipeline into a reusable Python script with command-line arguments and export results to CSV

---

## Author

**Alireza Ebrahimi** — [@iamalirezaebi](https://github.com/iamalirezaebi)
Background-subtraction setup adapted from the official OpenCV tutorial.
