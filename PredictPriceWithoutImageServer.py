from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
CORS(app)

# Load the saved model
model = load_model('modelWithoutImages')

data_path = './data/cars.csv'
data_path1 = './data/cars1.csv'
df = pd.concat(map(pd.read_csv, [data_path, data_path1]))

selected_columns = ['price', 'kilometrage', 'make_date', 'engine']
x_columns = ['kilometrage', 'make_date', 'engine']
df = df[selected_columns]
df['price'] = df['price'].str.replace(' ', '').astype(float)
df['kilometrage'] = df['kilometrage'].str.replace(' ', '').astype(float)
df['make_date'] = df['make_date'].str.split('-').str[0].astype(int)
df['engine'] = df['engine'].str.extract(r'AG \((\d+)kW\)').astype(float)
df.dropna(inplace=True)
X = df[x_columns]
y = df['price']

# Normalize the input features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the numeric values from the request
    kilometrage = int(request.form.get('kilometrage'))
    make_date = request.form.get('make_date')
    engine = int(request.form.get('engine'))
    # Parse the make_date string and extract the year
    make_date = int(make_date.split('-')[0])

    # Prepare the numeric data for prediction
    new_numeric_data = np.array([[kilometrage, make_date, engine]], dtype=np.float32)

    # Normalize the new data using the same scaler
    new_numeric_data = scaler.transform(new_numeric_data)

    # Perform prediction on the new data
    prediction = float(model.predict(new_numeric_data)[0])
    print(model.predict(new_numeric_data))
    
    # Return the predicted label as JSON response
    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(debug=True)
