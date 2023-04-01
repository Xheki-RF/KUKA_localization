import numpy as np
import cv2


video_path = r"C:\Users\pasho\Downloads\test_2.webm"
video = cv2.VideoCapture(video_path)

object_detector = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=500)

prev_contours = None

while True:
    _, window = video.read()

    points_to_transform = np.float32([[336, 233], [644, 225], [112, 470], [880, 450]])
    transformed_points = np.float32([[0, 0], [450, 0], [0, 750], [450, 750]])

    transform_matrix = cv2.getPerspectiveTransform(points_to_transform, transformed_points)
    final_image = cv2.warpPerspective(window, transform_matrix, (450, 750))

    mask = object_detector.apply(final_image)
    mask = cv2.GaussianBlur(mask, (3, 3), cv2.BORDER_DEFAULT)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        contours = np.concatenate(contours)
        x, y, width, height = cv2.boundingRect(contours)

        cv2.rectangle(final_image, (x, y), (x + width, y + height), (255, 255, 0), 1)
        cv2.putText(final_image, f"{x + width / 2}, {y + height / 2}", (x, y - 2), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 0))

        prev_contours = contours
    else:
        x, y, width, height = cv2.boundingRect(prev_contours)

        cv2.rectangle(final_image, (x, y), (x + width, y + height), (255, 255, 0), 2)
        cv2.putText(final_image, f"{x + width / 2}, {y + height / 2}", (x, y - 2), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 0))

    cv2.imshow("Transformed view", final_image)
    # cv2.imshow("Mask", mask)

    if cv2.waitKey(1) == ord("q"):
        break

    cv2.waitKey(20)

video.release()
cv2.destroyAllWindows()
