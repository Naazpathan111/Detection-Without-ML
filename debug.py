import cv2
import numpy as np

image = cv2.imread("images/pipes2.png")
image = cv2.resize(image, (1280, 720))

print("Draw box around pipes then press ENTER")
roi_box = cv2.selectROI("Select ROI", image, fromCenter=False)
cv2.destroyAllWindows()

x, y, w, h = roi_box
roi = image[y:y+h, x:x+w]

# Show each stage
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
gray = clahe.apply(gray)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
edges = cv2.Canny(blurred, 15, 60)

# Show all stages so we can see what's happening
cv2.imshow("1 - ROI", roi)
cv2.waitKey(0)
cv2.imshow("2 - Gray", gray)
cv2.waitKey(0)
cv2.imshow("3 - Edges", edges)
cv2.waitKey(0)

# Print all contours found with their properties
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"\nTotal contours found: {len(contours)}")
print("\nTop 20 contours by area:")
contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
for i, c in enumerate(contours_sorted[:20]):
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    if perimeter > 0:
        circularity = (4 * np.pi * area) / (perimeter ** 2)
    else:
        circularity = 0
    print(f"  Contour {i+1}: area={int(area)}, circularity={circularity:.3f}")

cv2.destroyAllWindows()