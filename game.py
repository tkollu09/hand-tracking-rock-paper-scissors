import cv2
import mediapipe as mp
import math

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def get_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)

def get_gesture(hand, handedness):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    if handedness == "Right":
        # For right hand: thumb tip.x < thumb joint.x means open
        fingers.append(1 if hand.landmark[tips[0]].x < hand.landmark[tips[0] - 1].x else 0)
    else:
        # For left hand: thumb tip.x > thumb joint.x means open
        fingers.append(1 if hand.landmark[tips[0]].x > hand.landmark[tips[0] - 1].x else 0)

    for tip in tips[1:]:
        fingers.append(1 if hand.landmark[tip].y < hand.landmark[tip - 2].y else 0)

    if fingers == [0,0,0,0,0]:
        return "rock"

    if fingers == [1,1,1,1,1] or fingers == [0,1,1,1,1]:
        return "paper"

    if fingers == [0,1,1,0,0]:
        return "scissors"

    return "none"

def get_computer_choice(cheat, player_choice):
    import random
    return random.choice(["rock", "paper", "scissors"]) if not cheat else \
        {"rock": "paper", "paper": "scissors", "scissors": "rock"}[player_choice]

def get_winner(player, computer):
    if player == computer:
        return "Tie"
    elif (player == "rock" and computer == "scissors") or  (player == "paper" and computer == "rock") or (player == "scissors" and computer == "paper"):
        return "Player Wins"
    else:
        return "Computer Wins"

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
    left_hand_base = None
    right_hand_base = None
    right_hand_gesture = "none"
    computer_choice = None
    last_gesture = None
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        left_hand_base = None
        right_hand_base = None
        right_hand_gesture = "none"

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label
                
                if hand_label == "Left": 
                    left_hand_base = hand_landmarks.landmark[0]
                elif hand_label == "Right":
                    right_hand_base = hand_landmarks.landmark[0]
                    right_hand_gesture = get_gesture(hand_landmarks, "Right")
            
            # Check if both hands are close together
            if left_hand_base and right_hand_base:
                distance = get_distance(left_hand_base, right_hand_base)
                if distance < 0.15:  # Adjust threshold as needed
                    right_hand_gesture = "thinking"
            
            # Get computer choice only when gesture changes and is valid
            if right_hand_gesture != last_gesture and right_hand_gesture not in ["none", "thinking"]:
                computer_choice = get_computer_choice(False, right_hand_gesture)
            
            last_gesture = right_hand_gesture
            
            cv2.putText(frame, f"Gesture: {right_hand_gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            if computer_choice:
                cv2.putText(frame, f"Computer: {computer_choice}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if right_hand_gesture in ["rock", "paper", "scissors"]:
                    result = get_winner(right_hand_gesture, computer_choice)
                    cv2.putText(frame, f"Result: {result}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    

            
        cv2.imshow("Hand Tracking (Press q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()