import os
import pandas as pd
import numpy as np
from keras.callbacks import CSVLogger
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder

data_path = './data/cars.csv'
data_path1 = './data/cars1.csv'
image_dir = './cropped/'
df = pd.concat(map(pd.read_csv,[data_path,data_path1]))
df = df[df['image'].map(lambda x: os.path.isfile(image_dir + x))]

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

image_height = 128
image_width = 128
image_channels = 3

input_img = Input(shape=(image_height, image_width, image_channels))
conv1 = Conv2D(32, (3, 3), activation='relu')(input_img)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
batch1 = BatchNormalization()(pool1)
conv2 = Conv2D(64, (3, 3), activation='relu')(batch1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
batch2 = BatchNormalization()(pool2)
conv3 = Conv2D(96, (3, 3), activation='relu')(batch2)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
batch3 = BatchNormalization()(pool3)
conv4 = Conv2D(96, (3, 3), activation='relu')(batch3)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
batch4 = BatchNormalization()(pool4)
drop1 = Dropout(0.2)(batch4)
conv5 = Conv2D(64, (3, 3), activation='relu')(drop1)
pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
batch5 = BatchNormalization()(pool5)
drop2 = Dropout(0.2)(batch5)
flatten_img = Flatten()(drop2)
img_dense1 = Dense(256, activation='relu')(flatten_img)
img_dense2 = Dense(128, activation='relu')(img_dense1)
img_dense3 = Dense(64, activation='relu')(img_dense2)
img_dense4 = Dense(32, activation='relu')(img_dense3)
img_dense5 = Dense(16, activation='relu')(img_dense4)

input_features = Input(shape=(num_features,))
dense1 = Dense(16, activation='relu')(input_features)
merged = concatenate([img_dense5, dense1])
dense2 = Dense(16, activation='relu')(merged)
output = Dense(1, activation='linear')(dense2)

model = Model(inputs=[input_img, input_features], outputs=output)

model.compile(optimizer=Adam(lr=0.001), loss='mean_absolute_error', metrics=['MeanAbsoluteError'])

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
model.fit([image_data_generator.next()[0], numeric_data], df['price'], epochs=500, batch_size=30, verbose=1, callbacks=[csv_logger])

model.save('model')