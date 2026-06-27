<img width="981" height="327" alt="outputimg" src="https://github.com/user-attachments/assets/da9d57d3-edc9-49c7-8b60-298ff2dae48b" />

# Detection Without ML 👁️

Real-world object detection and counting using pure classical 
computer vision. No neural networks. No training data. No pretrained models.
Just math, edges, and geometry.

## What it does
Detects and counts PVC pipes loaded in a tempo/truck by analyzing 
the circular cross-sections of pipe ends facing the camera.

## Pipeline
1. User selects ROI around the pipe bundle
2. CLAHE — normalizes contrast for harsh outdoor lighting
3. Gaussian Blur — removes pixel noise before edge detection
4. Canny Edge Detection — finds boundaries between bright and dark regions
5. findContours — traces every closed outline in the edge map
6. Circularity Filter — keeps only circular shapes using 4π×area/perimeter²
7. Count rendered on output image with labeled contours

## Why no ML?
Most people jump straight to YOLO for detection tasks.
This project explores how far pure classical CV can go —
and where it hits its fundamental limits.

Spoiler: it works well until pipes touch each other.
Merged edges = inseparable contours = the exact problem
deep learning was built to solve.

## Accuracy
80-90% on real-world truck images with clearly visible pipe ends.

## Stack
- Python
- OpenCV
- NumPy

## Run it
pip install -r requirements.txt
python main.py

