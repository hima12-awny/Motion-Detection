# Import necessary libraries
import cv2 as cv
import numpy as np
import time

# Create a 2x2 kernel for use in dilation operation
kernel = np.ones((2, 2), np.uint8)

# Open a video capture object using default camera
cap = cv.VideoCapture(0)

# Set the width and height of the captured frames
cap.set(3, 1200)
cap.set(4, 720)

previous_time = 0

font = cv.FONT_HERSHEY_SIMPLEX
fontColor = (255, 0, 0)  # (b, g, r)

# Loop to continuously capture frames and detect motion
while True:
    # Read two consecutive frames from the camera
    success1, mainFrame = cap.read()
    mainFrame = cv.flip(mainFrame, 1)

    success2, secondFrame = cap.read()
    secondFrame = cv.flip(secondFrame, 1)

    # Calculate absolute difference between frames
    diff = cv.absdiff(mainFrame, secondFrame)

    # Convert to grayscale and apply Gaussian blur
    diff_gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
    diff_blur = cv.GaussianBlur(diff_gray, (5, 5), 0)

    # Threshold the image to create a binary image
    _, thresh_bin = cv.threshold(diff_blur, 20, 255, cv.THRESH_BINARY)

    # Apply Canny edge detection filter and dilation operation
    canny = cv.Canny(thresh_bin, 150, 200)
    dilate = cv.dilate(canny, kernel, iterations=1)

    # Find contours in the dilated image
    contours, hierarchy = cv.findContours(dilate, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # If contours are found, check if their area is greater than a certain threshold
    if contours:
        for obj in contours:
            if cv.contourArea(obj) > 6:
                # Draw a rectangle around the object
                x, y, w, h = cv.boundingRect(obj)
                cv.rectangle(mainFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Add text indicating motion to the frame with color red
                fontColor = (0, 0, 255)
                cv.putText(mainFrame, 'Warning: Something Moving.', (10, 70), font, 1, fontColor, 2)
    else:
        fontColor = (255, 0, 0)

    # Get number of frame per second and put it in main frame
    current_time = time.time()
    FPS = int(1.0 / (current_time - previous_time))
    previous_time = current_time
    cv.putText(mainFrame, f'FPS: {FPS}', (10, 30), font, 1, fontColor, 2)

    # Display the original frame with detected motion and the dilated image
    cv.imshow('main frame', mainFrame)

    # Exit the loop if the 'q' key is pressed
    if cv.waitKey(1) & 0xff == ord('q'):
        break

# Release the capture object and close all windows
cap.release()
cv.destroyAllWindows()
