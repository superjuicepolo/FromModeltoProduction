import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load the dataset from CSV
df = pd.read_csv('my_data.csv')  # CSV should have 'filename' and 'label' columns
image_paths = df['filename'].tolist()
labels = df['label'].tolist()

# Lists to store image data and encoded labels
images = []
encoded_labels = []

# Load and preprocess each image
for path, label in zip(image_paths, labels):
    # Load image and resize to 128x128
    img = load_img(path, target_size=(128, 128))
    # Convert the image to a NumPy array
    img_array = img_to_array(img)
    images.append(img_array)
    # Encode label (string '1'->0, '2'->1, ..., '10'->9)
    encoded_label = int(label) - 1
    encoded_labels.append(encoded_label)

# Convert lists to NumPy arrays and normalize pixel values to [0, 1]
X = np.array(images, dtype='float32') / 255.0
y = np.array(encoded_labels, dtype='int')

# Shuffle and split data into training (80%) and validation (20%) sets
np.random.seed(42)  # For reproducibility
indices = np.arange(len(X))
np.random.shuffle(indices)
split_index = int(0.8 * len(X))
train_idx, val_idx = indices[:split_index], indices[split_index:]

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

# Define a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # 10 output classes for labels 0-9
])

# Compile the model with optimizer, loss, and metrics
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model on training data and validate on validation data
model.fit(X_train, y_train,
          epochs=10,
          validation_data=(X_val, y_val))

# Save the trained model to a file
model.save('cnn_model.h5')