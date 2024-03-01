# Label-prediction-handtracking-CNN-mediapipe

Task Overview:
Developing a computer vision program with a GUI, that loads the given image (Image source)
onto the screen, and when the user shows his hands in front of the in-built webcam,
exploring the diagram on the screen with their fingers, the program should be able to track
finger movement, and when the user points at each of the labeled regions of the image (not
on side labels), the labels needs to be readout with synthesized speech.

In this task I have approached the statement in two different approaches 
1. Using existing module-Mediapipe
2. Building own model to predict the hand movement

Using existing module-Mediapipe
Using the existing module named mediapipe i was able to predict the movement of the hand. When I point my index finger it could move the mouse. For the heart picture I labelled the data using data annotations and fixed the axis with its label. So when I move the cursor with the hand on the heart picture it would produce the label names which produces speaker output with the help of pyttsx3 module.
 ![Screenshot 2024-03-01 123348](https://github.com/navaneeth8056/Label-prediction-handtracking-CNN-mediapipe/assets/126904083/94bdd7f0-90ad-4293-8cce-b745f21c7242)


Building own model to predict the hand movement
Built own model using the CNN with 4 layers which could predict the finger count. Using the model I build a system which could move the cursor if I show the 5 fingers in either left or right hand. By calculating the central index of the hand it creates a centroid point by which it moves the mouse cursor.
Other process remains same by labelling the heart image with the corresponding axis and predicts the label when we move the cursor on the image and produces the output in an audio source.

Bugs and difficulties:
For building own model the dataset was not able to produce an accurate source of predictions. The model tends to overfit due to the lack of proper dataset. Due to this the prediction of hand was low compared to the inbuild models prediction.

Future optimization:
The accuracy of the model could be further developed by adding more relevant dataset and the prediction could be further optimized by adding more layers and by adding more epoch to the training model.
