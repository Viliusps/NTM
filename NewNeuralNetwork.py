import pandas as pd
import keras
from keras.layers import Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Keras model
model = keras.Sequential([
    keras.layers.Dense(64, input_dim=3, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    Dropout(0.2),
    keras.layers.Dense(1),
])

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=1)

model.save('modelWithoutImages')

# Evaluate the model
mse = model.evaluate(X_test, y_test)
print('Mean Squared Error:', mse)
