Sign Language Recognition System
Project Overview-
The Sign Language Recognition System is a desktop application designed to detect and interpret hand gestures for sign language communication. 
This system leverages computer vision techniques and deep learning models to recognize hand signs and convert them into corresponding text. 
The application is built using OpenCV for image processing and Keras for model training.

Features-
Hand-Gesture Detection: Uses a webcam to capture and detect hand signs.
Custom Dataset: Collects hand-gesture images to create a personalized sign language dataset.
Real-time Recognition: Recognizes signs in real-time and converts them to text using pre-trained models.
Model Training: The system uses a Keras-based neural network to train on the collected dataset.

Key Files-
main.py:
Used for data collection and creation of the sign language dataset.
Captures hand gestures and stores them in specific folders for each sign.

text.py and convertword.py:
Responsible for recognizing and converting the detected gestures into text with high accuracy.

Dataset Creation-
Data Collection:
Use main.py to capture hand-gesture images and store them in predefined folders based on the sign they represent.
Each folder corresponds to a specific sign (e.g., a folder for each letter in English).

Model Training:
The collected dataset is used to train a model using Keras.
The model is saved as keras.h5.
Labels for the gestures are stored in label.txt.

Technologies Used-
OpenCV: For image capture and hand-gesture detection.
Keras: For building and training the neural network model.
NumPy: For handling data arrays and operations.
