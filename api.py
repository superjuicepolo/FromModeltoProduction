from cnn_predict import predict_images
from flask import Flask, request, jsonify
from utils import get_jpg_image_paths

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # print("printing from server")

    try:
        labels = predict_images(get_jpg_image_paths("returned_images_of_the_items"))
        return jsonify({"predicted_labels": labels})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False)