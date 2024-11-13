import cv2
import numpy as np

# Paths to YOLO files
weights_path = "yolo_files/yolov3-pokemon_final.weights"  # Use your trained weights file here
config_path = "yolo_files/yolov3-pokemon.cfg"  # The modified .cfg file for binary classification
names_path = "yolo_files/pokemon.names"  # Contains "Not Pokemon Card" and "Pokemon Card"

# Load YOLO model
net = cv2.dnn.readNet(weights_path, config_path)

# Load class labels
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]  # Should contain "Not Pokemon Card" and "Pokemon Card"

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Set colors for each class
colors = [(0, 0, 255), (0, 255, 0)]  # Red for Not Pokemon Card, Green for Pokemon Card

# Initialize the video feed (use 0 for webcam or a video file path)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the image for YOLO
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Process YOLO outputs
    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Adjust threshold as needed
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])  # Should be "Not Pokemon Card" or "Pokemon Card"
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the output frame
    cv2.imshow("YOLO Object Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
