import cv2
import dlib
from math import hypot

# Initialize dlib's face detector and landmark predictor
predictorModel = 'shape_predictor_model/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictorModel)

def calcDistance(pointA, pointB):
    # Calculate the Euclidean distance between two points
    return hypot(pointA[0] - pointB[0], pointA[1] - pointB[1])

def mouthTrack(faces, frame):
    statuses = []

    for face in faces:
        landmarks = predictor(frame, face)
        # Outer lip top (landmark 51) and bottom (landmark 57)
        outerTop = (landmarks.part(51).x, landmarks.part(51).y)
        outerBottom = (landmarks.part(57).x, landmarks.part(57).y)
        distance = calcDistance(outerTop, outerBottom)

        if distance > 23:
            statuses.append("Mouth Open")
        else:
            statuses.append("Mouth Closed")
    return statuses

def main():
    # Main loop to capture video and apply mouth tracking
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        # Process the frame for mouth tracking and get status output
        statuses = mouthTrack(faces, frame)
        print("Status:", statuses)
        
        # Optionally, you can still display the frame without text overlays.
        cv2.imshow("Webcam Feed", frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
