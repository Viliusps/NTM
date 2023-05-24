from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load the saved model
model = load_model('model')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    image_file = request.files['image']

    # Load and preprocess the new image
    img = Image.open(image_file)
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255.0  # Normalize the image

    # Prepare the image data for prediction
    new_image = np.array([img])

    # Get the numeric values from the request
    kilometrage = int(request.form.get('kilometrage'))
    make_date_str = request.form.get('make_date')
    engine = int(request.form.get('engine'))

    # Parse the make_date string and extract the year
    make_date = int(make_date_str.split('-')[0])

    # Prepare the numeric data for prediction
    new_numeric_data = np.array([[kilometrage, make_date, engine]], dtype=np.float32)

    # Perform prediction on the new data
    prediction = float(model.predict([new_image, new_numeric_data])[0])

    # Return the predicted label as JSON response
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
