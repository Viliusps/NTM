import pandas as pd
import numpy as np
from keras.callbacks import CSVLogger
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder

data_path = './data/cars.csv'
data_path1 = './data/cars1.csv'
image_dir = './data/images/'
df = pd.concat(map(pd.read_csv,[data_path,data_path1]))

selected_columns = ['price', 'image', 'kilometrage', 'make_date', 'engine']
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

num_features = len(selected_columns) - 2

image_height = 224
image_width = 224
image_channels = 3

input_img = Input(shape=(image_height, image_width, image_channels))
conv1 = Conv2D(32, (3, 3), activation='relu')(input_img)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
flatten_img = Flatten()(pool1)

input_features = Input(shape=(num_features,))
dense1 = Dense(64, activation='relu')(input_features)
merged = concatenate([flatten_img, dense1])
dense2 = Dense(128, activation='relu')(merged)
dense3 = Dense(128, activation='relu')(dense2)
dense4 = Dense(128, activation='relu')(dense3)
dense5 = Dense(128, activation='relu')(dense4)
output = Dense(1, activation='linear')(dense5)

model = Model(inputs=[input_img, input_features], outputs=output)

model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error', metrics=['MeanAbsoluteError'])

model.summary()

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
model.fit([image_data_generator.next()[0], numeric_data], df['price'], epochs=1000, batch_size=30, verbose=1, callbacks=[csv_logger])

