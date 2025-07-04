import tensorflow as tf
import numpy as np #numerical operations in python

print("Loading MNIST dataset...")

mnist = tf.keras.datasets.mnist #splits data set into training and testing 
(x_train, y_train), (x_test, y_test) = mnist.load_data() #x_train are images to teach AI, y_train is the correct labels for the training images. 
print("Data set loaded")
print("Normalizing pixel values")
x_train = x_train / 255.0  #x_test images to test AI after trained, y_test labels for testing images. 
x_test = x_test / 255.0
print("Normalization complete.")

print("Reshaping data for the model...")
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("Reshaping complete. Final Shape:", x_train.shape)

print("Building the model..")
#sequential model for data to flow through layers after layers. 
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),     #32 filters, 3x3 pixel area, and to AI to learn faster, non-linearity
    tf.keras.layers.MaxPooling2D((2,2)), #shrinks the image to half its size to make program run faster. 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(), #flattens the data to a single long line of numbers instead of a grid. 
    tf.keras.layers.Dense(128, activation='relu'), #thinking layer
    tf.keras.layers.Dropout(0,2), #prevents the AI model from cheatingm turns off 20% of the model randomly. 
    tf.keras.layers.Dense(10, activation='softmax') #final output layer. 
])
print("Model built sucessfully.")

print("Compiling the model...")
model.compile(optimizer='adam', #adjusts models neurons during training, improves accuracy
              loss='sparse_categorical_crossentropy', #loss function
              metrics=['accuracy']) # % of images that are accurate in training 
print("Training the model...")
model.fit(x_train, y_train, epochs=10) #saves entire model 
print("Training Complete.")

print("Saving the model")
model.save('digit_model.h5')
print("Model saved as digit_model.h5")
