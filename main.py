import cv2
import numpy as np
#import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Function to get the object classification
def classify_image(frame):
    # Convert frame to image format compatible with the model
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))  # Resize to 224x224
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess for the model
    
    # Get predictions
    predictions = model.predict(img_array)
    
    # Decode predictions and get the most likely object
    decoded_predictions = decode_predictions(predictions, top=1)[0][0]
    
    return decoded_predictions

# Function to run the webcam and get user input for noun
def run_webcam():
    # Open the webcam (default is 0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Please enter a noun for classification:")
    noun = input("Enter noun: ").lower()

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        # Classify the current frame from the webcam
        prediction = classify_image(frame)
        predicted_class = prediction[1].lower()  # Predicted object label
        confidence = prediction[2]  # Confidence score
        
        # Check if the predicted class matches the input noun
        match = "yes" if noun in predicted_class else "no"
        
        # Display the classification result
        cv2.putText(frame, f"Predicted: {predicted_class} ({confidence*100:.2f}%)", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(frame, f"Match with '{noun}': {match}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show the frame
        cv2.imshow('Webcam Feed', frame)
        
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_webcam()
