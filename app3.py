import streamlit as st
import numpy as np
import speech_recognition as sr
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import cv2
import time

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

def preprocess_image(image):
    """
    Preprocess the captured image for the MobileNetV2 model.
    Converts RGBA to RGB if necessary and resizes the image.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((224, 224))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    
    return image_array

def run_deep_learning_model(image):
    """
    Takes an image, preprocesses it, and makes predictions using the MobileNetV2 model.
    Returns the top 3 predicted labels and their associated probabilities.
    """
    image_array = preprocess_image(image)
    predictions = model.predict(image_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    
    return decoded_predictions

def recognize_speech_and_trigger_camera():
    """
    Recognizes speech and triggers the camera input if the correct keyword is recognized.
    """
    recognizer = sr.Recognizer()

    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        st.info("Say 'capture' to trigger the camera...")

        # Adjust the recognizer sensitivity to ambient noise
        recognizer.adjust_for_ambient_noise(source)
        
        # Listen for speech
        audio = recognizer.listen(source)

        try:
            # Recognize speech using Google Web Speech API
            speech_text = recognizer.recognize_google(audio)
            st.write(f"Recognized Speech: {speech_text}")
            
            # If the recognized word is 'capture', trigger the camera input
            if 'capture' in speech_text.lower():
                st.success("Keyword 'capture' recognized! Capturing image...")

                # Trigger the camera capture using OpenCV
                return capture_image()

            else:
                st.error("No valid keyword recognized. Please try again and say 'capture'.")

        except sr.UnknownValueError:
            st.error("Sorry, could not understand the speech.")
        except sr.RequestError as e:
            st.error(f"Could not request results from the speech recognition service; {e}")
    
    return None

def capture_image():
    """
    Captures an image using the OpenCV library from the webcam.
    """
    cap = cv2.VideoCapture(1)  # Open the webcam (0 is usually the default webcam)

    if not cap.isOpened():
        st.error("Could not open the webcam.")
        return None

    st.info("Preparing to capture image...")

    # Adding a delay to give the webcam time to adjust
    time.sleep(1)  # 1-second delay for the camera to adjust

    # Capture several frames to ensure real-time capture
    frames = []
    for _ in range(5):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    if not frames:
        st.error("Failed to capture image.")
        return None

    # Average the frames to get a stable image
    frame_avg = np.mean(frames, axis=0).astype(np.uint8)

    # Convert the image from OpenCV's BGR to RGB format
    frame_rgb = cv2.cvtColor(frame_avg, cv2.COLOR_BGR2RGB)
    
    # Close the webcam
    cap.release()

    # Convert the OpenCV image to a PIL image for further processing
    pil_image = Image.fromarray(frame_rgb)
    
    # Display the captured image
    st.image(pil_image, caption="Captured Image", use_column_width=True)

    return pil_image

# Streamlit app
st.title("Speech-Activated Image Classifier with OpenCV")

st.write("This app listens for the word 'capture' and automatically captures an image using OpenCV. The captured image will then be classified using a deep learning model.")

if st.button('Start Speech Recognition'):
    captured_image = recognize_speech_and_trigger_camera()

    if captured_image is not None:
        # Run the deep learning model on the captured image
        predictions = run_deep_learning_model(captured_image)

        # Display predictions
        st.write("Predictions:")
        for i, (imagenet_id, label, confidence) in enumerate(predictions):
            st.write(f"{i+1}: {label} ({confidence * 100:.2f}%)")
