import pandas as pd
import numpy as np
from keras.models import load_model
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

# Load the saved model
model = load_model('modelWithoutImages')

# Define the directories and file paths for new data
new_image_dir = './data/images/'
new_data_path = './data/cars.csv'

# Load the new data
new_df = pd.read_csv(new_data_path)

# Perform the same preprocessing steps as before
selected_columns = ['image', 'kilometrage', 'make_date', 'engine']
new_df = new_df[selected_columns]
new_df['kilometrage'] = new_df['kilometrage'].str.replace(' ', '').astype(float)
new_df['make_date'] = new_df['make_date'].str.split('-').str[0].astype(int)
new_df['engine'] = new_df['engine'].str.extract(r'AG \((\d+)kW\)').astype(float)
new_df.dropna(inplace=True)
new_df = new_df.head(100)

scaler = MinMaxScaler()
new_df[selected_columns[1:]] = scaler.fit_transform(new_df[selected_columns[1:]])

# Load and preprocess the new images
new_images = []
for image_path in new_df['image']:
    img = Image.open(new_image_dir + image_path)
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255.0  # Normalize the image
    new_images.append(img)
new_images = np.array(new_images)

# Prepare the numeric data for prediction
new_numeric_data = new_df.loc[:, selected_columns[1:]].values.astype(np.float32)

# Perform prediction on the new data
predictions = model.predict(new_numeric_data)

# Print the predicted prices
for i, prediction in enumerate(predictions):
    print(f"Prediction for data point {i + 1}: {prediction[0]}")
