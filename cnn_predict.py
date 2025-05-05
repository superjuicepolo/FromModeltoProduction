import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


model = load_model('cnn_model.h5')


def predict_images(image_paths):


    predictions = []
    for path in image_paths:
        img = load_img(path, target_size=(128, 128))
        img_array = img_to_array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        probs = model.predict(img_array)[0]
        pred_index = np.argmax(probs)
        pred_label = str(pred_index + 1)

        predictions.append((path, pred_label))

    return predictions

