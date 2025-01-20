from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam
from PIL import Image
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('soldier_uniform_classifier.h5')

# Recompile the model to suppress warnings
model.compile(
    optimizer=Adam(learning_rate=0.001),  # Use the same optimizer as during training
    loss='categorical_crossentropy',      # Use the same loss function as during training
    metrics=['accuracy']
)

# Define class labels
classes = ['BSF', 'CRPF', 'JKP']

@app.route('/', methods=['GET', 'POST'])
def home():
    """Homepage for uploading an image and displaying the result."""
    if request.method == 'POST':
        # Check if an image is provided in the request
        if 'image' not in request.files:
            return render_template('index.html', error="No image uploaded. Please upload an image.")

        file = request.files['image']

        # Validate file
        if file:
            try:
                # Open the image file
                image = Image.open(file.stream)
                # Resize and preprocess the image
                image = image.resize((224, 224))
                img_array = img_to_array(image) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Make predictions
                predictions = model.predict(img_array)
                predicted_class = classes[np.argmax(predictions)]
                confidence = np.max(predictions)

                # Pass the prediction result to the HTML template
                result = f"The image is classified as {predicted_class}."
                return render_template('index.html', result=result)
            except Exception as e:
                return render_template('index.html', error=f"Error processing the image: {str(e)}")
        else:
            return render_template('index.html', error="Invalid file format. Please upload a valid image.")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

