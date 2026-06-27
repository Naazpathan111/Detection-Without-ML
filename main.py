import cv2
import numpy as np

def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def find_pipes(preprocessed, original):
    edges = cv2.Canny(preprocessed, 15, 60)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pipe_count = 0
    result = original.copy()

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 5000 or area > 12000:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = (4 * np.pi * area) / (perimeter ** 2)

        if circularity > 0.20:
            bx, by, bw, bh = cv2.boundingRect(contour)
            if bw < 30 or bh < 30:
                continue
            pipe_count += 1
            cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(result, str(pipe_count), (cx - 8, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.putText(result, f"Total pipes: {pipe_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    return result, pipe_count

# ── Load ──
image = cv2.imread("images/pipes2.png")
if image is None:
    print("Error: Could not load image.")
    exit()

image = cv2.resize(image, (1280, 720))

# ── ROI ──
print("Draw box around pipes then press ENTER")
roi_box = cv2.selectROI("Select pipe area - press ENTER", image, fromCenter=False)
cv2.destroyAllWindows()

x, y, w, h = roi_box
roi = image[y:y+h, x:x+w]

preprocessed = preprocess(roi)
result, count = find_pipes(preprocessed, roi)

print(f"Total pipes: {count}")
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()