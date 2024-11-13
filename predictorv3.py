import cv2
import numpy as np
import random
import threading
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from fuzzywuzzy import fuzz

# List of nouns (targets)
nounList = [
    "elephant", "guitar", "bicycle", "shark", "tree", "banana", "monkey", "rocket",
    "camera", "balloon", "fish", "turtle", "flute", "ghost", "lion", "horse", "pizza",
    "helicopter", "scissors", "penguin", "umbrella", "piano", "giraffe", "airplane", 
    "glove", "trophy", "sunglasses", "ice cream", "hammer", "lizard", "chair", "whistle", 
    "dog", "cat"
]

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Global variable to hold the current target word
current_target_word = random.choice(nounList)

# Lock for safe thread communication (to avoid race conditions)
lock = threading.Lock()

# Function to change the target word every 30 seconds
def change_target_word():
    global current_target_word
    while True:
        with lock:  # Ensures the target word change is thread-safe
            current_target_word = random.choice(nounList)
            print(f"New target word: {current_target_word}")  # You can log it or update a GUI here
        threading.Event().wait(30)  # Wait for 30 seconds before changing the word

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
    
    # Check if the predicted class matches the input noun
    exists, word = guess_in_guesses(noun, object_list)
    if exists:
        print(f"-------------------------------------\nFOUND THIS FRAME: {word} :: {confidence_list[object_list.index(word)]}\n-------------------------------------\n\n")
        return confidence_list[object_list.index(word)]
    else:
        return 0.0  # If no match, return 0.0 confidence

# Function to display the current target word and process webcam feed
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
        
        # Display the confidence score and current target word
        cv2.putText(frame, f"Confidence for '{noun}': {confidence*100:.2f}%", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Max Confidence: {maxConfidence*100:.2f}%", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Current Target: {noun}", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Show the frame
        cv2.imshow('Webcam Feed', frame)
        
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Check if the target word has changed
        with lock:  # Access the updated target word safely
            updated_target = current_target_word
        
        if updated_target != noun:
            print(f"Target word changed to: {updated_target}")
            round(updated_target)  # Start a new round with the updated target word
            return  # Exit the current round to restart with the new target word

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Start a background thread to change the target word every 30 seconds
    threading.Thread(target=change_target_word, daemon=True).start()
    
    # Start the round with the initial target word
    round(current_target_word)
