import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('face_spoofing_model.h5')

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (160, 160))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to detect face spoofing
def detect_face_spoofing(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Could not read the image.")
        return

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Predict using the model
    prediction = model.predict(preprocessed_image)
    if prediction[0][0] > 0.5:
        print("Real face detected.")
    else:
        print("Spoofed face detected.")

# Example usage
image_path = 'path_to_your_image.jpg'
detect_face_spoofing(image_path)