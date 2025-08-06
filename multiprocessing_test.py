import cv2
import dlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from object_detection import detectObject
from mouth_tracking import mouthTrack
from head_pose_estimation import get_head_pose, initialize_face_detector
import time

# Initialize dlib's face detector and landmark predictor
predictorModel = 'shape_predictor_model/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictorModel)

# Initialize face detector and landmark predictor for head pose estimation
face_detector, landmark_predictor = initialize_face_detector(predictorModel)

# --- Modified task functions to return results ---
def mouth_tracking_task(faces, frame):
    return ("Mouth Statuses", mouthTrack(faces, frame))

def object_detection_task(frame):
    return ("Detected Objects", detectObject(frame))

def head_pose_task(frame, face_detector, landmark_predictor):
    return ("Head Pose", get_head_pose(frame, face_detector, landmark_predictor))
# --- End modifications ---

def main():
    cap = cv2.VideoCapture(0)  # Use webcam; change the source if needed
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Create a persistent thread pool with 3 workers
    with ThreadPoolExecutor(max_workers=3) as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert 3-channel images to 1-channel image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 0)

            # Count the number of faces
            faceCount = len(faces)
        
            # Always submit object detection task
            fut1 = executor.submit(object_detection_task, frame)
            
            if faceCount == 0:
                print("No face detected")
                # Process only object detection future
                print("Detected Objects:", fut1.result())
            elif faceCount == 1:
                fut2 = executor.submit(mouth_tracking_task, faces, frame)
                fut3 = executor.submit(head_pose_task, frame, face_detector, landmark_predictor)
                futures = [fut2, fut1, fut3]
                # Process tasks as they complete
                for future in futures:
                    task, result = future.result()
                    print(f"{task}: {result}")
            else:  # faceCount > 1
                print("Multiple faces detected", fut1.result())

            # Display the normal frame
            cv2.imshow("Combined Feed", frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
