#necessary libraries
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image
import base64
import io

#initialize the flask app and load AI model
app = Flask(__name__)
model = tf.keras.models.load_model('digit_model.h5')

#main route
#function when someone visits the homepage
@app.route('/')
def home():
    #renders index.html file from templates
    return render_template('index.html')

# define the prediction route
#runs when front end sends predict request
@app.route('/predict', methods=['POST'])
def predict():

    #gets image from request
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    
    #pre process the image
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))

    # --- Start of Corrections ---
    # The following block is updated to correctly handle the image from the canvas.

    # First, handle transparency (the 'A' in RGBA) from the canvas
    # by creating a white background and pasting the image on top.
    bg = Image.new('RGB', image.size, 'WHITE')
    bg.paste(image, (0, 0), image)

    # convert the image to grayscale L for luminance
    image = bg.convert('L')

    #resize the image 28x28
    image = image.resize((28,28))

    #convert the image to numpy array of #'s
    image_array = np.array(image)

    # Normalize the pixel values after creating the array
    image_array = image_array / 255.0

    # Invert the image colors (model expects white digit on black background)
    image_array = 1 - image_array

    # --- End of Corrections ---

    #reshapes the data to format to our model (1,28,28,1)
    image_array = np.expand_dims(image_array, axis = 0) #adds batch dimension
    image_array = np.expand_dims(image_array, axis = -1) #add channel dimensions

    prediction = model.predict(image_array)[0] # Get the first (and only) prediction
    predicted_digit = int(np.argmax(prediction))
    
    # Get the confidence level for the predicted digit
    confidence = float(prediction[predicted_digit]) * 100 # Convert to percentage
    #send prediction back
    return jsonify({'prediction': predicted_digit, 'confidence': f'{confidence:.2f}'})

#run the app
if __name__ == '__main__':
    app.run(debug=True)