import cv2
import numpy as np
import random
import threading
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from fuzzywuzzy import fuzz

# List of nouns (targets)
nounList = [
    "elephant", "guitar", "bicycle", "shark", "tree", "banana", "monkey", "rocket", "camera", "balloon",
    "fish", "turtle", "flute", "ghost", "lion", "horse", "pizza", "helicopter", "scissors", "penguin",
    "umbrella", "piano", "giraffe", "airplane", "glove", "trophy", "sunglasses", "ice cream", "hammer",
    "lizard", "chair", "whistle", "dog", "cat"
]

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Global variables to store player scores, current round state, and number of players
player_scores = []
num_players = 0
current_player = 1
target_word = ''
round_started = False

# Function to change the target word every 30 seconds
def change_target_word():
    return random.choice(nounList)

def guess_in_guesses(noun, listOG, threshold=80):
    for guess in listOG:
        score = fuzz.partial_ratio(noun.lower(), guess.lower())
        if score >= threshold:
            return True, guess
    return False, "NaN"

def get_confidence_for_noun(frame, noun):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1000)[0]
    object_list = [pred[1] for pred in decoded_predictions]
    confidence_list = [pred[2] for pred in decoded_predictions]
    exists, word = guess_in_guesses(noun, object_list)
    if exists:
        return confidence_list[object_list.index(word)]
    else:
        return 0.0

def play_round(player_id):
    global target_word
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    maxConfidence = 0.00
    start_time = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        confidence = get_confidence_for_noun(frame, target_word)
        maxConfidence = max(maxConfidence, confidence)

        # Display instructions and current status on the screen
        cv2.putText(frame, f"Player {player_id} - Confidence for '{target_word}': {confidence*100:.2f}%",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Max Confidence: {maxConfidence*100:.2f}%",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Current Target: {target_word}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Display game instructions
        cv2.putText(frame, "Press 'Q' to quit, 'N' to start a new round",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # Check if 30 seconds has passed
        elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        if elapsed_time > 30:
            break

        cv2.imshow('Webcam Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return maxConfidence

def main():
    global num_players, current_player, player_scores, round_started, target_word

    while True:
        if not round_started:
            # Ask the user how many players when the game starts or restarts
            num_players = int(input("Enter number of players: "))
            round_started = True
            player_scores.clear()  # Reset scores for the new game
            current_player = 1  # Start from player 1

        # Start rounds for each player
        for current_player in range(1, num_players + 1):
            target_word = change_target_word()  # New word for each round
            print(f"Player {current_player}, the target word is: {target_word}")
            score = play_round(current_player)
            player_scores.append(score)
            print(f"Player {current_player} finished with a score of {score*100:.2f}%")

        # Determine the winner (highest score)
        winner_score = max(player_scores)
        winner_player = player_scores.index(winner_score) + 1
        print(f"\nPlayer {winner_player} wins with the highest score of {winner_score*100:.2f}%!")

        # Ask for next action inside the OpenCV window (no CLI prompts)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            break

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Display the winner and options
            cv2.putText(frame, f"Player {winner_player} wins with {winner_score*100:.2f}%!",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Press 'Q' to quit, 'N' to play another round, 'R' to restart.",
                        (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow("Game Over", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Exiting game...")
                cap.release()
                cv2.destroyAllWindows()
                return  # Exit the game
            elif key == ord('n'):
                print("Starting a new round with the same players...")
                break  # Start a new round
            elif key == ord('r'):
                print("Restarting the game...")
                round_started = False
                break  # Restart the game

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
