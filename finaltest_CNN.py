from keras.models import load_model
import cv2
import numpy as np
import pyautogui
import pyttsx3
import pandas as pd

# Loading the trained model
model = load_model('Fingers_Detection_CNN_Tensorflow_Keras.model')

IMG_SIZE = 128
label_list = ['0L', '1L', '2L', '3L', '4L', '5L', '0R', '1R', '2R', '3R', '4R', '5R']
min_area_threshold = 4000

pyautogui.FAILSAFE = False

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
CURSOR_SPEED = 10 

prev_hand_x, prev_hand_y = 0, 0

# Function to preprocess the frame
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    normalized_frame = gray_frame / 255.0
    reshaped_frame = normalized_frame.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    return reshaped_frame

# Initializing webcam
cap = cv2.VideoCapture(0)

# Loading the labeled data
data = pd.read_csv("C:/Users/ADMIN/Documents/Dverse/labels.csv")

# Initializing the text-to-speech engine
engine = pyttsx3.init()

def read_label(label):
    engine.say(label)
    engine.runAndWait()

# Function to move cursor based on hand position
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
                read_label(row['label_name'])
                break

# Seting up variables for smoothing mouse movement
prev_x, prev_y = 0, 0
smooth_alpha = 0.5

# Displaying the image
image = cv2.imread("C:/Users/ADMIN/Documents/Dverse/heart-diagram_page-0001.jpg")
cv2.namedWindow("Heart Image")
cv2.setMouseCallback("Heart Image", mouse_event)

while True:
    ret, frame = cap.read()
    
    # Converting frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range of skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Threshold the HSV image to get only skin color
    mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)

    # Finding contours in the mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area_threshold:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculating hand position (centroid of bounding box)
            hand_x = x + w // 2
            hand_y = y + h // 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (hand_x, hand_y), 5, (0, 255, 0), -1)
            
            # Calculate cursor movement based on hand position only if the predicted label is '5R' or '5L'
            if predicted_label == '5R' or predicted_label == '5L':
                cursor_dx = (hand_x - prev_hand_x) * CURSOR_SPEED
                cursor_dy = (hand_y - prev_hand_y) * CURSOR_SPEED
                pyautogui.moveRel(cursor_dx, cursor_dy)
            
            prev_hand_x, prev_hand_y = hand_x, hand_y
    

    preprocessed_frame = preprocess_frame(frame)
    prediction = model.predict(preprocessed_frame)
    predicted_label_index = np.argmax(prediction)
    predicted_label = label_list[predicted_label_index]

    cv2.putText(frame, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Displaying the frame
    cv2.imshow('Hand Gesture Recognition', frame)
    
    # Displaying the heart image
    cv2.imshow("Heart Image", image)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
