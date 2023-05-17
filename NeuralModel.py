import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder

data_path = './data/cars.csv'
image_dir = './data/images/'
df = pd.read_csv(data_path)

selected_columns = ['price', 'gearbox', 'kilometrage', 'make_date', 'fuel_id']
df = df.loc[:, selected_columns]
df['price'] = df['price'].str.replace(' ', '')
df['price'] = df['price'].astype(float)

for col in df.columns:
    if df[col].dtype == 'object':
        label_encoder = LabelEncoder()
        df[col] = label_encoder.fit_transform(df[col].astype(str))

df.dropna(inplace=True)

num_features = len(selected_columns) - 1

input_features = Input(shape=(num_features,))
dense1 = Dense(64, activation='relu')(input_features)

dense2 = Dense(128, activation='relu')(dense1)
output = Dense(1, activation='linear')(dense2)

model = Model(inputs=input_features, outputs=output)

model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error', metrics=['accuracy'])

model.summary()

numeric_data = df.loc[:, selected_columns[1:]].copy()
numeric_data = numeric_data.values.astype(np.float32)
print(numeric_data)

model.fit(numeric_data, df['price'], epochs=500, batch_size=50, verbose=2)