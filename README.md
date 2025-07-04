Handwritten Digit Recognizer - An AI Web Application
Live Repository: https://github.com/ashleywyattjapan/DigitRecognizer

This project is a web-based application that uses a custom-built Convolutional Neural Network (CNN) to recognize and predict handwritten digits drawn by a user in real-time. It demonstrates a full end-to-end development cycle, from training a deep learning model to deploying it in an interactive web interface.

Features
Interactive Drawing Canvas: Users can draw a digit (0-9) directly in the browser using their mouse.

Real-Time AI Prediction: With a single click, the user's drawing is sent to a Python backend where a trained TensorFlow model predicts the digit.

Prediction Confidence: The application not only predicts the digit but also displays the model's confidence level in its prediction.

Clean, Responsive Interface: A simple and intuitive frontend built with HTML, CSS, and vanilla JavaScript.

Technologies Used
Backend: Python, Flask, TensorFlow, Keras, NumPy, Pillow

Frontend: HTML, CSS, JavaScript (with Fetch API)

Dataset: MNIST

How It Works
The application is composed of three main parts:

Frontend (Client-Side): An index.html file with an HTML5 canvas allows the user to draw. When the "Predict" button is clicked, JavaScript converts the canvas drawing into a Base64 image string and sends it to the backend via a POST request.

Backend (Server-Side): A Flask server (app.py) receives the image data. It uses the Pillow library to pre-process the image—resizing it to 28x28 pixels, converting it to grayscale, and inverting the colors to match the format of the training data. The processed image is then fed into the AI model.

AI Model: A Convolutional Neural Network (CNN) built and trained in train_digit_model.py using TensorFlow/Keras on the MNIST dataset. The trained model (digit_model.h5) is loaded by the Flask app and used to predict the digit from the processed image. The prediction and confidence score are sent back to the frontend as a JSON response.

Setup and Installation
To run this project on your local machine, follow these steps:

Clone the repository:

git clone https://github.com/ashleywyattjapan/DigitRecognizer.git
cd DigitRecognizer

Set up a virtual environment (recommended):

python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required libraries:

pip install tensorflow numpy Pillow flask

Train the AI Model:
If the digit_model.h5 file is not included, run the training script once to generate it:

python train_digit_model.py

Run the Flask Application:

python app.py

Open your web browser and navigate to http://127.0.0.1:5000/.

Areas for Improvement
Increase Model Accuracy: The model could be made more robust by training for more epochs or by using data augmentation techniques to create more varied training examples.

Live Deployment: Deploy the Flask application to a cloud service like PythonAnywhere or Heroku to make it publicly accessible.

UI Enhancements: Add animations or a more dynamic display for the prediction results to improve the user experience.
