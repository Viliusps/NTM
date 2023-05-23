import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

data_path = './data/cars.csv'
data_path1 = './data/cars1.csv'
image_dir = './data/images/'
df = pd.concat(map(pd.read_csv,[data_path,data_path1]))

selected_columns = ['price', 'kilometrage', 'make_date', 'engine']

df = df[selected_columns]
df['price'] = df['price'].str.replace(' ', '').astype(float)
df['kilometrage'] = df['kilometrage'].str.replace(' ', '').astype(float)
df['make_date'] = df['make_date'].str.split('-').str[0].astype(int)
df['engine'] = df['engine'].str.extract(r'AG \((\d+)kW\)').astype(float)

#Convert specific columns to numeric
columns_to_encode = []#['gearbox', 'fuel_id']

#Encode categorical columns
label_encoder = LabelEncoder()
df[columns_to_encode] = df[columns_to_encode].apply(lambda x: label_encoder.fit_transform(x))

df.dropna(inplace=True)

num_features = len(selected_columns) - 1

input_features = Input(shape=(num_features,))
dense1 = Dense(64, activation='relu')(input_features)
dense2 = Dense(128, activation='relu')(dense1)
dense3 = Dense(128, activation='relu')(dense2)
dense4 = Dense(128, activation='relu')(dense3)
dense5 = Dense(128, activation='relu')(dense4)
output = Dense(1, activation='linear')(dense5)

print(df.head())
sns.heatmap(df.corr(), annot=True)
plt.show()

model = Model(inputs=input_features, outputs=output)

model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error', metrics=['MeanAbsoluteError'])

model.summary()

numeric_data = df.loc[:, selected_columns[1:]].copy()
numeric_data = numeric_data.values.astype(np.float32)
numeric_data = np.reshape(numeric_data, (numeric_data.shape[0], num_features))

model.fit(numeric_data, df['price'], epochs=1000, batch_size=20, verbose=1)