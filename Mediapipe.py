import cv2
import pandas as pd
import pyttsx3
import pyautogui
import mediapipe as mp

# Loading the labeled data
data = pd.read_csv("C:/Users/ADMIN/Documents/Dverse/labels.csv")

# Initializing the text-to-speech engine
engine = pyttsx3.init()

# Function to read out the label name given in the image
def read_label(label):
    engine.say(label)
    engine.runAndWait()

# Function to move mouse cursor based on hand position
def move_cursor(x, y):
    screen_width, screen_height = pyautogui.size()
    pyautogui.moveTo(int(x * screen_width), int(y * screen_height))

# Function to handle mouse events
def mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        for _, row in data.iterrows():
            # Check if the mouse cursor is within the bounding box
            if row['bbox_x'] <= x <= row['bbox_x'] + row['bbox_width'] and \
               row['bbox_y'] <= y <= row['bbox_y'] + row['bbox_height']:
                # Read out the label name
                read_label(row['label_name'])
                break

# Loading the hand tracking module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Webcam initialization
cap = cv2.VideoCapture(0)
hand_detector = mp_hands.Hands()
drawing_utils = mp_drawing

# Seting up variables for smoothing mouse movement
prev_x, prev_y = 0, 0
smooth_alpha = 0.5

# Displaying the image for label prediction
image = cv2.imread("C:/Users/ADMIN/Documents/Dverse/heart-diagram_page-0001.jpg")
cv2.namedWindow("Heart Image")
cv2.setMouseCallback("Heart Image", mouse_event)

# Display the video with hand tracking
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks
    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                if id == 8:
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255))
                    x_smooth = smooth_alpha * prev_x + (1 - smooth_alpha) * x
                    y_smooth = smooth_alpha * prev_y + (1 - smooth_alpha) * y
                    move_cursor(x_smooth / frame_width, y_smooth / frame_height)
                    prev_x, prev_y = x_smooth, y_smooth
    
    # Displaying the frame with hand tracking
    cv2.imshow('Hand Tracking', frame)
    # Displaying the heart image
    cv2.imshow("Heart Image", image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
