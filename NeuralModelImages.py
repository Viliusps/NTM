import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder

data_path = './data/cars.csv'
image_dir = './data/images/'
df = pd.read_csv(data_path)

selected_columns = ['title', 'price', 'gearbox', 'image', 'kilometrage', 'make_date', 'fuel_id']
df = df.loc[:, selected_columns]

for col in df.columns:
    if df[col].dtype == 'object' and col not in ['image', 'title']:
        label_encoder = LabelEncoder()
        df[col] = label_encoder.fit_transform(df[col].astype(str))

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
output = Dense(1, activation='linear')(dense2)

model = Model(inputs=[input_img, input_features], outputs=output)

model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error', metrics=['accuracy'])

model.summary()

datagen = ImageDataGenerator(rescale=1.0/255.0)

image_data_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory=image_dir,
    x_col='image',
    y_col='price',
    target_size=(image_height, image_width),
    batch_size=1676,
    class_mode='raw'
)

numeric_data = df.loc[:, selected_columns[1:]].copy()
numeric_data.drop('image', axis=1, inplace=True)
numeric_data = numeric_data.values.astype(np.float32)

model.fit([image_data_generator.next()[0], numeric_data], df['price'], epochs=500, batch_size=50, verbose=1)

