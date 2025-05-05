import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

df = pd.read_csv('my_data.csv')
image_paths = df['filename'].tolist()
labels = df['label'].tolist()

images = []
encoded_labels = []


for path, label in zip(image_paths, labels):
    img = load_img(path, target_size=(128, 128))
    img_array = img_to_array(img)
    images.append(img_array)
    encoded_label = int(label) - 1
    encoded_labels.append(encoded_label)

X = np.array(images, dtype='float32') / 255.0
y = np.array(encoded_labels, dtype='int')

np.random.seed(42)
indices = np.arange(len(X))
np.random.shuffle(indices)
split_index = int(0.8 * len(X))
train_idx, val_idx = indices[:split_index], indices[split_index:]

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(X_train, y_train,
          epochs=10,
          validation_data=(X_val, y_val))

model.save('cnn_model.h5')