#app.py
import cv2
import imutils
import time
from facial_detections import detectFace
from blink_extra_detection import isBlinking
from mouth_tracking import mouthTrack
from object_detection import detectObject
from eye_tracker import gazeDetection
from head_pose_estimation import head_pose_detection 
import os
from datetime import datetime

global data_record
data_record = []

# For Beeping on Ubuntu
def beep():
    os.system('beep -f 3000 -l 100')

# OpenCV videocapture for the webcam
cam = cv2.VideoCapture(0)

# Face Count If-else conditions
def faceCount_detection(faceCount):
    if faceCount > 1:
        time.sleep(5)
        remark = "Multiple faces detected."
        beep()
    elif faceCount == 0:
        remark = "No face detected."
        time.sleep(3)
        beep()
    else:
        remark = "Face detected properly."
    return remark

# Main function 
def proctoringAlgo():
    blinkCount = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        record = []
        current_time = datetime.now().strftime("%H:%M:%S.%f")
        record.append(current_time)

        faceCount, faces = detectFace(frame)
        remark = faceCount_detection(faceCount)
        record.append(remark)

        if faceCount == 1:
            # Blink Detection
            blinkStatus = isBlinking(faces, frame)
            if blinkStatus[2] == "Blink":
                blinkCount += 1
                record.append(f"Blink count: {blinkCount}")
            else:
                record.append(blinkStatus[2])

            # Gaze Detection
            eyeStatus = gazeDetection(faces, frame)
            record.append(eyeStatus)

            # Mouth Detection
            mouth_status = mouthTrack(faces, frame)
            record.append(mouth_status)

            # Object detection
            objectName = detectObject(frame)
            record.append(objectName)

            if len(objectName) > 1:
                time.sleep(4)
                beep()
                continue

            # Head Pose estimation
            head_pose_status = head_pose_detection(faces, frame)
            record.append(head_pose_status)
        
        data_record.append(record)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    proctoringAlgo()
    activityVal = "\n".join(map(str, data_record))
    with open('activity.txt', 'w') as file:
        file.write(str(activityVal))