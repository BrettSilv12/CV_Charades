import cv2
import numpy as np
import random
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
#from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from fuzzywuzzy import fuzz

nounList = [
    "elephant",
    "guitar",
    "bicycle",
    "shark",
    "tree",
    "banana",
    "monkey",
    "rocket",
    "camera",
    "balloon",
    "fish",
    "turtle",
    "flute",
    "ghost",
    "lion",
    "horse",
    "pizza",
    "helicopter",
    "cactus",
    "scissors",
    "penguin",
    "umbrella",
    "piano",
    "giraffe",
    "airplane",
    "glove",
    "trophy",
    "sunglasses",
    "ice cream",
    "hammer",
    "lizard",
    "chair",
    "whistle",
    "dog",
    "cat"
]

# Load the pre-trained MobileNetV2 model
model = ResNet50(weights='imagenet')
#model = MobileNetV2(weights='imagenet')

def guess_in_guesses(noun, listOG, threshold=80):
    for guess in listOG:
        # Calculate fuzzy match score between the input noun and predicted class
        score = fuzz.partial_ratio(noun.lower(), guess.lower())
        if score >= threshold:
            return True, guess  # Return true and details of the match
    
    return False, "NaN" # No match found


# Function to get the confidence score for the noun match
def get_confidence_for_noun(frame, noun):
    # Convert frame to image format compatible with the model
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))  # Resize to 224x224
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess for the model
    
    # Get predictions
    predictions = model.predict(img_array)
    
    # Decode predictions and get the most likely object
    decoded_predictions = decode_predictions(predictions, top=1000)[0]
    object_list = [pred[1] for pred in decoded_predictions]
    confidence_list = [pred[2] for pred in decoded_predictions]
    #confidence_list = decoded_predictions[1]  # Confidence score
    # Check if the predicted class matches the input noun
    exists, word = guess_in_guesses(noun, object_list)
    if exists:
        print(f"-------------------------------------\nFOUND THIS FRAME: {word} :: {confidence_list[object_list.index(word)]}\n-------------------------------------\n\n")
        return confidence_list[object_list.index(word)]
    else:
        return 0.0  # If no match, return 0.0 confidence

# Function to run the webcam and get user input for noun
def round(noun):
    # Open the webcam (default is 0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    

    maxConfidence = 0.00

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        # Get the confidence score for the noun match
        confidence = get_confidence_for_noun(frame, noun)
        maxConfidence = max(maxConfidence, confidence)
        # Display the confidence score
        cv2.putText(frame, f"Confidence for '{noun}': {confidence*100:.2f}%", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Max Confidence: {maxConfidence*100:.2f}%", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
        # Show the frame
        cv2.imshow('Webcam Feed', frame)
        
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    round(nounList[random.randint(0, len(nounList) - 1)])
