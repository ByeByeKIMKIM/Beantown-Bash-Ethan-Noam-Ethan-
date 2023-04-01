import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import random

def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
    denom = (x1-x2) * (x1-x3) * (x2-x3);
    A = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom;
    B = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom;
    C = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom;

    return A,B,C

class TrackedPoint:
    def __init__(self, point_id, x, y):
        self.point_id = point_id
        self.x = x
        self.y = y

tracked_points_x = []
tracked_points_y = []
# Load the video file
cap = cv2.VideoCapture('IMG_0578.mp4')

# Define the color range for the dot you want to track
lower_red = np.array([0, 0, 0])
upper_red = np.array([200, 200, 200])

# Set the initial position of the point
x, y = None, None

# Loop through frames of the video
while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the frame using the color range
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Apply a morphological opening to the thresholded image
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find the contours in the thresholded image
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through each contour and find the centroid of the point
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # If the point has not been set yet, set it to the current position
            if x is None and y is None:
                x, y = cx, cy
                tracked_points_x.append(x)
                tracked_points_y.append(y)
                # new_point = TrackedPoint(len(tracked_points), cx, cy)
                # tracked_points.append(new_point)

            # Otherwise, update the position of the point
            else:
                x, y = 0.5 * (x + cx), 0.5 * (y + cy)
                tracked_points_x.append(x)
                tracked_points_y.append(y)
                # new_point = TrackedPoint(len(tracked_points), cx, cy)
                # tracked_points.append(new_point)

    #Draw a circle around the tracked point
    if x is not None and y is not None:
        cv2.circle(frame, (int(x), int(y)), 10, (0, 0, 255), -1)

    
    #Display the video with the circle drawn around the tracked point
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and destroy all windows
# cap.release()
# cv2.destroyAllWindows()

# for point in tracked_points:
#     print(f"({point.x}, {point.y})")

for point in tracked_points_y:
    point = -1 * point
for a in tracked_points_y:
  tracked_points_y[tracked_points_y.index(a)]=tracked_points_y[tracked_points_y.index(a)]*-1

# plt.scatter(tracked_points_x, tracked_points_y)
# plt.set_xticklabels([])
# # plt.set_yticklabels([])
# plt.show()

fig, ax = plt.subplots()

# Plot the data
ax.scatter(tracked_points_x, tracked_points_y)

# Remove the axis labels
# ax.set_xticklabels([])
# ax.set_yticklabels([])

beginning = 0
end = len(tracked_points_x) - 1
middle = len(tracked_points_x)/2

coef = calc_parabola_vertex(tracked_points_x[0], tracked_points_y[0], tracked_points_x[int(middle)], tracked_points_y[int(middle)], tracked_points_x[end], tracked_points_y[end])
x_parabola = np.linspace(np.min(tracked_points_x), np.max(tracked_points_x))
y = coef[0] * x_parabola**2 + coef[1] * x_parabola + coef[2]
# Show the plot
plt.plot(x_parabola, y)
text = str(str(round(coef[0], 10))) + "x^2 + " + str(str(round(coef[1], 2)) + "x + " + str(str(round(coef[2], 2))))
x_lim = plt.xlim()
y_lim = plt.ylim()
plt.text(x_lim[0] + 20, y_lim[0] + 20,text)
#print(str(coef[0]) + str(coef[1]) + str(coef[2]))
plt.show()
