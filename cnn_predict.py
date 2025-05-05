import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained CNN model
model = load_model('cnn_model.h5')


def predict_images(image_paths):
    """
    Predict the class label for each image in the list of file paths.

    Args:
        image_paths (List[str]): List of image file paths.

    Returns:
        List[Tuple[str, str]]: List of (image_path, predicted_label) tuples.
    """
    predictions = []
    for path in image_paths:
        # Load the image and resize to 128x128
        img = load_img(path, target_size=(128, 128))
        img_array = img_to_array(img)
        # Normalize pixel values to [0, 1]
        img_array = img_array.astype('float32') / 255.0
        # Add batch dimension: shape becomes (1, 128, 128, 3)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict class probabilities
        probs = model.predict(img_array)[0]
        # Determine the class with highest probability
        pred_index = np.argmax(probs)
        # Convert index to label string ('1' to '10')
        pred_label = str(pred_index + 1)

        predictions.append((path, pred_label))

    return predictions

# Example usage:
# image_list = ['archive/train/1/0a4f72f295b004edf174736390d625ca83da4383_1623146930_50414315ed754701ab973aadee1a67c6.jpg', 'archive/train/5/0a90f9cca5cfef4a390e6a83f799f142fb26ee37_1628546209_97b95c7641514e73b7c8f95039259221.jpg']
# results = predict_images(image_list)
# for img_path, label in results:
#     print(f"{img_path} -> Predicted label: {label}")
