import argparse
import time
import numpy as np
import cv2

import utills
import plot

confid = 0.5
thresh = 0.5
mouse_pts = []

# Get mouse points from user
# Order is
# bottom left
# bottom right
# top right
# top left
# point 1
# point 2
# point 3
# The last 3 points mark a distance of 6 feet in image vertically and horizontally


def get_mouse_points(event, x, y, flags, param):

    global mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_pts) < 4:
            cv2.circle(image, (x, y), 5, (0, 255, 0), 10)
        else:
            cv2.circle(image, (x, y), 5, (0, 0, 255), 10)

        if len(mouse_pts) >= 1 and len(mouse_pts) <= 3:
            cv2.line(image, (x, y), (mouse_pts[len(
                mouse_pts)-1][0], mouse_pts[len(mouse_pts)-1][1]), (70, 70, 70), 2)
            if len(mouse_pts) == 3:
                cv2.line(image, (x, y),
                         (mouse_pts[0][0], mouse_pts[0][1]), (70, 70, 70), 2)

        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))

# Function to calculate social distancing violations


def calculate_social_distancing(vid_path, net, output_dir, output_vid, ln1):

    count = 0
    vs = cv2.VideoCapture(vid_path)

    # Get video height, width and fps
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(vs.get(cv2.CAP_PROP_FPS))

    # Set scale for birds eye view
    scale_w, scale_h = utills.get_scale(width, height)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    # Initialize writer objects
    output_movie = cv2.VideoWriter("Output.avi", fourcc, fps, (width, height))
    output_movie2 = cv2.VideoWriter("Output2.avi", fourcc, fps, (1920, 1080))
    bird_movie = cv2.VideoWriter("./output_vid/bird_eye_view.avi",
                                 fourcc, fps, (int(width * scale_w), int(height * scale_h)))

    points = []
    global image

    while True:
        # Read frames
        (grabbed, frame) = vs.read()

        if not grabbed:
            print('here')
            break

        (H, W) = frame.shape[:2]

        if count == 0:
            while True:
                image = frame
                cv2.imshow("image", image)
                cv2.waitKey(1)
                if len(mouse_pts) == 8:
                    cv2.destroyWindow("image")
                    break

            points = mouse_pts

        src = np.float32(np.array(points[:4]))
        dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
        # Transform perspective using opencv method
        prespective_transform = cv2.getPerspectiveTransform(src, dst)

        # using next 3 points for horizontal and vertical unit length(in this case 6 Feets ~= 180 cm)
        pts = np.float32(np.array([points[4:7]]))
        warped_pt = cv2.perspectiveTransform(pts, prespective_transform)[0]

        # Calculate distance scale using marked points by user
        distance_w = np.sqrt(
            (warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
        distance_h = np.sqrt(
            (warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
        pnts = np.array(points[:4], np.int32)
        cv2.polylines(frame, [pnts], True, (70, 70, 70), thickness=2)

        # Using YOLO v3 model using dnn method
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln1)
        end = time.time()
        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # detecting humans in frame
                if classID == 0:

                    if confidence > confid:
                        # Finding bounding boxes dimensions
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
        # Applying Non Maximum Suppression to remove multiple bounding boxes around same object
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)
        font = cv2.FONT_HERSHEY_PLAIN
        boxes1 = []
        for i in range(len(boxes)):
            if i in idxs:
                boxes1.append(boxes[i])
                x, y, w, h = boxes[i]

        if len(boxes1) == 0:
            count = count + 1
            continue

        # Get transformed points using perspective transform
        person_points = utills.get_transformed_points(
            boxes1, prespective_transform)

        # Get distances between the points
        distances_mat, bxs_mat = utills.get_distances(
            boxes1, person_points, distance_w, distance_h)

        # Get the risk counts
        risk_count = utills.get_count(distances_mat)

        frame1 = np.copy(frame)

        bird_image = plot.bird_eye_view(
            frame, distances_mat, person_points, scale_w, scale_h, risk_count)
        img = plot.social_distancing_view(frame1, bxs_mat, boxes1, risk_count)
        if count != 0:

            bird_movie.write(bird_image)

            cv2.imshow('Social Distancing Detect', img)
            output_movie.write(img)
            output_movie2.write(img)
            cv2.imwrite(output_dir+"frame%d.jpg" % count, img)
            cv2.imwrite(output_dir+"bird_eye_view/frame%d.jpg" %
                        count, bird_image)

        count = count + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        output_movie.write(img)

    vs.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    video_path = './data/example.mp4'
    model_path = './models/'

    output_dir = './output/'
    output_vid = './output_vid/'

    # load Yolov3 weights

    weightsPath = model_path + "yolov3.weights"
    configPath = model_path + "yolov3.cfg"

    # Initializing yolov3 weights
    net_yl = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net_yl.getLayerNames()
    ln1 = [ln[i[0] - 1] for i in net_yl.getUnconnectedOutLayers()]

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", get_mouse_points)
    np.random.seed(42)

    # Start the detection
    calculate_social_distancing(
        video_path, net_yl, output_dir, output_vid, ln1)
