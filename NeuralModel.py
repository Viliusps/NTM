import pandas as pd
import numpy as np
import keras
from keras.layers import Input
import matplotlib.pyplot as plt
import seaborn as sns

data_path = './data/cars.csv'
data_path1 = './data/cars1.csv'
image_dir = './data/images/'
df = pd.concat(map(pd.read_csv, [data_path, data_path1]))

selected_columns = ['price', 'kilometrage', 'make_date', 'engine']

df = df[selected_columns]
df['price'] = df['price'].str.replace(' ', '').astype(float)
df['kilometrage'] = df['kilometrage'].str.replace(' ', '').astype(float)
df['make_date'] = df['make_date'].str.split('-').str[0].astype(int)
df['engine'] = df['engine'].str.extract(r'AG \((\d+)kW\)').astype(float)

# Convert specific columns to numeric
# columns_to_encode = []  # ['gearbox', 'fuel_id']
#
# # Encode categorical columns
# label_encoder = LabelEncoder()
# df[columns_to_encode] = df[columns_to_encode].apply(lambda x: label_encoder.fit_transform(x))

df.dropna(inplace=True)

num_features = len(selected_columns) - 1

input_features = Input(shape=(num_features,))
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(num_features,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='tanh'),
    keras.layers.Dense(1)
])

sns.heatmap(df.corr(), annot=True)
plt.show()

model.compile('adam', loss='mean_squared_error', metrics=['MeanAbsoluteError'])

model.summary()

numeric_data = df.loc[:, selected_columns[1:]].copy()
numeric_data = numeric_data.values.astype(np.float32)
numeric_data = np.reshape(numeric_data, (numeric_data.shape[0], num_features))

model.fit(numeric_data, df['price'], epochs=10000, batch_size=32, verbose=1)
