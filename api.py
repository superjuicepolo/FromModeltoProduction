from predict import predict_batch
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # print("printing from server")

    try:
        labels = predict_batch()
        return jsonify({"predicted_labels": labels})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False)