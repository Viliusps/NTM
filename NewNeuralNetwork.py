import pandas as pd
import keras
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler

data_path = './data/cars.csv'
data_path1 = './data/cars1.csv'
data_path2 = './data/cars2.csv'
df = pd.concat(map(pd.read_csv, [data_path, data_path1, data_path2]))

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

# Check for outliers
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
outliers = ((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
df = df[~outliers]

# Normalize the input features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Create the Keras model
model = keras.Sequential([
    keras.layers.Dense(128, input_dim=3, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    Dropout(0.2),
    keras.layers.Dense(1),
])

# Compile the model
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(), metrics=['mae'])

# Train the model
model.fit(X, y, epochs=1000, batch_size=32, verbose=2)

model.save('modelWithoutImages')

# Evaluate the model
mse = model.evaluate(X, y)
print('Mean Squared Error:', mse)
