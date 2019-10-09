from imutils.video import FileVideoStream
from UI import Text
from imutils.video import FPS
import numpy as np
import imutils
import cv2
import pafy as pafy
from collections import Counter

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load youtube video using pafy and youtube_dl
url = 'https://youtu.be/25EgbhdVESE'
vPafy = pafy.new(url)
play = vPafy.getbest()

print("[INFO] starting video file thread...")
fvs = FileVideoStream(play.url).start()

# start the FPS timer
fps = FPS().start()

# loop over frames from the video file stream
while fvs.more():
    frame = fvs.read()
    frame = imutils.resize(frame, width=1200)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.dstack([frame, frame, frame])

    height, width, channels = frame.shape

    frame = cv2.UMat(frame)

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00055, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    labels = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w // 2)
                y = int(center_y - h // 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

    labels = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            labels.append(label)
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 1.5,
                        color, 2)

    counter = dict(Counter(labels))
    text = Text().draw(counter, frame, (0, 145, 255))

    cv2.imshow("Frame", frame)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

    fps.update()

fps.stop()

print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
fvs.stop()