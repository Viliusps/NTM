import pandas as pd
import numpy as np
from keras.callbacks import CSVLogger
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import keras
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler

data_path = './data/cars.csv'
data_path1 = './data/cars1.csv'
data_path2 = './data/cars2.csv'
image_dir = './data/images/'
df = pd.concat(map(pd.read_csv, [data_path, data_path1, data_path2]))

selected_columns = ['price','image', 'kilometrage', 'make_date', 'engine']
x_columns = ['kilometrage', 'make_date', 'engine']
df = df[selected_columns]
df['price'] = df['price'].str.replace(' ', '').astype(float)
df['kilometrage'] = df['kilometrage'].str.replace(' ', '').astype(float)
df['make_date'] = df['make_date'].str.split('-').str[0].astype(int)
df['engine'] = df['engine'].str.extract(r'AG \((\d+)kW\)').astype(float)
df.dropna(inplace=True)
X = df[x_columns]
y = df['price']

# Check for outliers
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
outliers = ((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
df = df[~outliers]

# Normalize the input features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

image_height = 224
image_width = 224
image_channels = 3

modelImages = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height,image_width,image_channels)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(64, (3,3),activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(64, (3,3),activation='relu'),
    keras.layers.Flatten()
])

# Create the Keras model
model = keras.Sequential([
    keras.layers.Dense(128, input_dim=3, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
])

merged = concatenate([modelImages.output, model.output])

output = keras.layers.Dense(1, activation='linear')(merged)
combined_model = keras.Model(inputs=[modelImages.input, model.input], outputs=output)

combined_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['MeanAbsoluteError'])

combined_model.summary()

datagen = ImageDataGenerator(rescale=1.0/255.0)

image_data_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory=image_dir,
    x_col='image',
    y_col='price',
    target_size=(image_height, image_width),
    batch_size=len(df),
    class_mode='raw'
)

numeric_data = df.loc[:, selected_columns[1:]].copy()
numeric_data.drop('image', axis=1, inplace=True)
numeric_data = numeric_data.values.astype(np.float32)

csv_logger = CSVLogger("epochs_logs.txt", separator='\t', append=True)
combined_model.fit([image_data_generator.next()[0], numeric_data], df['price'], epochs=250, batch_size=30, verbose=2, callbacks=[csv_logger])

combined_model.save('model')