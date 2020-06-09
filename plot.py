
import numpy as np
import cv2


def bird_eye_view(frame, distances_mat, bottom_points, scale_w, scale_h, risk_count):
    h = frame.shape[0]
    w = frame.shape[1]

    red = (0, 0, 255)
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    white = (200, 200, 200)

    blank_image = np.zeros((int(h * scale_h), int(w * scale_w), 3), np.uint8)
    blank_image[:] = white
    warped_pts = []
    r = []
    g = []
    y = []
    for i in range(len(distances_mat)):

        if distances_mat[i][2] == 0:
            if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                r.append(distances_mat[i][0])
            if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                r.append(distances_mat[i][1])

            blank_image = cv2.line(blank_image, (int(distances_mat[i][0][0] * scale_w), int(distances_mat[i][0][1] * scale_h)), (int(
                distances_mat[i][1][0] * scale_w), int(distances_mat[i][1][1] * scale_h)), red, 2)

    for i in range(len(distances_mat)):

        if distances_mat[i][2] == 1:
            if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                y.append(distances_mat[i][0])
            if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                y.append(distances_mat[i][1])

            blank_image = cv2.line(blank_image, (int(distances_mat[i][0][0] * scale_w), int(distances_mat[i][0][1] * scale_h)), (int(
                distances_mat[i][1][0] * scale_w), int(distances_mat[i][1][1] * scale_h)), yellow, 2)

    for i in range(len(distances_mat)):

        if distances_mat[i][2] == 2:
            if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                g.append(distances_mat[i][0])
            if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                g.append(distances_mat[i][1])

    for i in bottom_points:
        blank_image = cv2.circle(
            blank_image, (int(i[0] * scale_w), int(i[1] * scale_h)), 5, green, 10)
    for i in y:
        blank_image = cv2.circle(
            blank_image, (int(i[0] * scale_w), int(i[1] * scale_h)), 5, yellow, 10)
    for i in r:
        blank_image = cv2.circle(
            blank_image, (int(i[0] * scale_w), int(i[1] * scale_h)), 5, red, 10)

    return blank_image


def social_distancing_view(frame, distances_mat, boxes, risk_count):

    red = (0, 0, 255)
    green = (0, 255, 0)
    yellow = (0, 255, 255)

    for i in range(len(boxes)):

        x, y, w, h = boxes[i][:]
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), green, 2)

    for i in range(len(distances_mat)):

        per1 = distances_mat[i][0]
        per2 = distances_mat[i][1]
        closeness = distances_mat[i][2]

        if closeness == 1:
            x, y, w, h = per1[:]
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), yellow, 2)

            x1, y1, w1, h1 = per2[:]
            frame = cv2.rectangle(frame, (x1, y1), (x1+w1, y1+h1), yellow, 2)

            frame = cv2.line(frame, (int(x+w/2), int(y+h/2)),
                             (int(x1+w1/2), int(y1+h1/2)), yellow, 2)

    for i in range(len(distances_mat)):

        per1 = distances_mat[i][0]
        per2 = distances_mat[i][1]
        closeness = distances_mat[i][2]

        if closeness == 0:
            x, y, w, h = per1[:]
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), red, 2)

            x1, y1, w1, h1 = per2[:]
            frame = cv2.rectangle(frame, (x1, y1), (x1+w1, y1+h1), red, 2)

            frame = cv2.line(frame, (int(x+w/2), int(y+h/2)),
                             (int(x1+w1/2), int(y1+h1/2)), red, 2)

    cv2.putText(frame, "HIGH RISK : " + str(risk_count[0]) + " people",
                (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    cv2.putText(frame, "LOW RISK : " + str(risk_count[1]) + " people",
                (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(frame, "SAFE : " + str(risk_count[2]) + " people",
                (50,  100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    return frame
