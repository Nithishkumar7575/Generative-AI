import cv2
import pytesseract

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the image
image = cv2.imread('car_image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to remove noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect edges using Canny
edges = cv2.Canny(blur, 50, 150)

# Find contours in the edges
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area to find the potential number plate region
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) == 4:  # Assuming number plate has 4 corners
        number_plate = approx
        break

# Draw contours on the original image
cv2.drawContours(image, [number_plate], -1, (0, 255, 0), 3)

# Crop the detected number plate region
x, y, w, h = cv2.boundingRect(number_plate)
cropped_plate = image[y:y+h, x:x+w]

# Convert the cropped image to grayscale
gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)

# Use pytesseract to perform OCR
plate_text = pytesseract.image_to_string(gray_plate, config='--psm 8')

# Print the detected text
print("Detected number plate:", plate_text)

# Display the result
cv2.imshow("Number Plate Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()