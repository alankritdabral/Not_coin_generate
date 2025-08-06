import cv2
import numpy as np

# Load YOLOv7 model and class labels
net = cv2.dnn.readNet("object_detection_model/weights/yolov4-tiny.weights",
                      "object_detection_model/config/yolov4-tiny.cfg")

with open("object_detection_model/objectLabels/coco.names", "r") as file:
    label_classes = [name.strip() for name in file.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] if isinstance(i, (list, np.ndarray)) else layer_names[i - 1] 
                 for i in net.getUnconnectedOutLayers()]

def detectObject(frame):
    """Detect objects in the given frame using YOLOv7."""
    labels_this_frame = []
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (220, 220), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    if len(indexes) > 0:
        for i in indexes.flatten():
            label = str(label_classes[class_ids[i]])
            labels_this_frame.append((label, confidences[i]))
    
    return labels_this_frame

def main():
    """Main function to capture video and detect objects."""
    cap = cv2.VideoCapture(0)  # Use webcam; replace 0 with a video file path if needed
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects
        detected_objects = detectObject(frame)
        print(f"Detected Objects: {detected_objects}")
        
        # Display frame
        cv2.imshow("Object Detection", frame)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
