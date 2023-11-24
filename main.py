import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0)

# Create a face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

font = cv2.FONT_HERSHEY_PLAIN

def getCenterPoint(value1, value2):
    center = ((value1 + value2) / 2)
    return int(center)

def get_blanking_ratio(eye_points, landmarks):

    # horizontal line two points
    left_point = (landmarks.part(eye_points[0]).x, landmarks.part(eye_points[0]).y)
    right_point = (landmarks.part(eye_points[3]).x, landmarks.part(eye_points[3]).y)

    # call function that calculate the center points
    mid_point_x_up = getCenterPoint(landmarks.part(eye_points[1]).x, landmarks.part(eye_points[2]).x)
    mid_point_y_up = getCenterPoint(landmarks.part(eye_points[1]).y, landmarks.part(eye_points[2]).y)

    # call function that calculate the center points
    mid_point_x_down = getCenterPoint(landmarks.part(eye_points[5]).x, landmarks.part(eye_points[4]).x)
    mid_point_y_down = getCenterPoint(landmarks.part(eye_points[5]).y, landmarks.part(eye_points[4]).y)

    # vertical line two points
    mid_point_up = (mid_point_x_up, mid_point_y_up)
    mid_point_down = (mid_point_x_down, mid_point_y_down)

    # # drawing the horizontal line
    # cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    # # drawing the vertical line
    # cv2.line(frame, mid_point_up, mid_point_down, (0, 0, 255), 2)

    vert_line_length = hypot((mid_point_x_up - mid_point_x_down), (mid_point_y_up - mid_point_y_down))
    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))

    ratio = hor_line_length / vert_line_length
    return ratio

def get_gaze_ratio(eye_points,  landmarks):
    left_eye_region = np.array([(landmarks.part(eye_points[0]).x, landmarks.part(eye_points[1]).y),
                                (landmarks.part(eye_points[1]).x, landmarks.part(eye_points[1]).y),
                                (landmarks.part(eye_points[2]).x, landmarks.part(eye_points[2]).y),
                                (landmarks.part(eye_points[3]).x, landmarks.part(eye_points[3]).y),
                                (landmarks.part(eye_points[4]).x, landmarks.part(eye_points[4]).y),
                                (landmarks.part(eye_points[5]).x, landmarks.part(eye_points[5]).y)], np.int32)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)

    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)

    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y:max_y, min_x:max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape

    left_side_threshold = threshold_eye[0: height, 0:int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2):width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if right_side_white == 0 and left_side_white == 0:
        return -1

    if right_side_white != 0:
        gaze_ratio = left_side_white / right_side_white
    else:
        gaze_ratio = 3

    return gaze_ratio

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not _:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()

        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        landmarks = predictor(gray, face)

        left_eye_ratio = get_blanking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blanking_ratio([42, 43, 44, 45, 46, 47], landmarks)

        if left_eye_ratio > 4.5 and right_eye_ratio > 4.5:
            cv2.putText(frame, "BLinking", (100, 50), font, 7, (255, 0, 0))

        # Gaze detection
        gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
        gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2

        if gaze_ratio == -1:
            cv2.putText(frame, "Can not detect gaze", (10, 100), font, 3, (0, 0, 255), 3)
        cv2.putText(frame, str(gaze_ratio), (50, 150), font, 3, (0, 0, 255), 3)

        if gaze_ratio > 0:
            if gaze_ratio <= 0.5:
                cv2.putText(frame, "LEFT", (200, 300), font, 3, (0, 0, 255), 3)
            elif 0.5 < gaze_ratio < 2.5:
                cv2.putText(frame, "CENTER", (200, 300), font, 3, (0, 0, 255), 3)
            else:
                cv2.putText(frame, "RIGHT", (200, 300), font, 3, (0, 0, 255), 3)
        #
        # cv2.putText(frame, str(gaze_ratio_left_eye), (200, 300), font, 3, (0, 0, 255), 3)
        # cv2.putText(frame, str(gaze_ratio_right_eye), (200, 340), font, 3, (0, 0, 255), 3)



    cv2.imshow('eye tracker', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
