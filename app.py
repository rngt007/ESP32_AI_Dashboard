
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import os
import io
import time
import threading

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "traffic_cnn.tflite")

environment = {
    "temperature": None,
    "humidity": None,
    "pressure": None,
    "gas": None,
    "duty_ratio": None,
    "pm25": None,
    "voc_grade": None,
    "aqi_category": None,
    "environment_prediction": None,
    "sound_db": None,
    "noise_status": None,
    "sound_threshold": 65.0,
    "updated_at": None
}

interpreter = None
input_details = None
output_details = None
model_lock = threading.Lock()


def load_model():
    global interpreter, input_details, output_details

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    import tensorflow as tf

    interpreter = tf.lite.Interpreter(
        model_path=MODEL_PATH
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\n========== TRAFFIC MODEL ==========")
    print("Model:", MODEL_PATH)
    print("Input shape:", input_details[0]["shape"])
    print("Input type:", input_details[0]["dtype"])
    print("Input quantization:",
          input_details[0]["quantization"])
    print("Output shape:", output_details[0]["shape"])
    print("Output type:", output_details[0]["dtype"])
    print("===================================\n")


def predict_traffic(image):
    global interpreter

    if interpreter is None:
        load_model()

    input_info = input_details[0]
    shape = input_info["shape"]

    if len(shape) != 4 or int(shape[3]) != 3:
        raise ValueError(
            f"Unsupported model input shape: {shape}"
        )

    height = int(shape[1])
    width = int(shape[2])

    image = image.convert("RGB").resize(
        (width, height),
        Image.Resampling.BILINEAR
    )

    pixels = np.asarray(image, dtype=np.float32)
    dtype = input_info["dtype"]

    # Float input: do not normalize here.
    # This assumes the model includes its own preprocessing.
    if dtype == np.float32:
        pixels = pixels.astype(np.float32)

    elif dtype in (np.uint8, np.int8):
        scale, zero_point = input_info["quantization"]

        if scale <= 0:
            raise ValueError(
                "Invalid quantization scale. "
                "Check the model input details."
            )

        pixels = np.round(
            pixels / scale + zero_point
        )

        limits = np.iinfo(dtype)
        pixels = np.clip(
            pixels, limits.min, limits.max
        ).astype(dtype)

    else:
        raise ValueError(
            f"Unsupported input data type: {dtype}"
        )

    pixels = np.expand_dims(pixels, axis=0)

    with model_lock:
        interpreter.set_tensor(
            input_info["index"], pixels
        )
        interpreter.invoke()

        raw_output = interpreter.get_tensor(
            output_details[0]["index"]
        )

    scores = np.asarray(
        raw_output[0], dtype=np.float32
    ).flatten()

    print("\n========== PREDICTION DEBUG ==========")
    print("Image:", image.size)
    print("Input min/max:",
          float(np.min(pixels)),
          float(np.max(pixels)))
    print("Raw model output:", scores)
    print("======================================")

    if len(scores) == 1:
        # Single sigmoid output requires confirmation
        # of what class the positive output represents.
        score = float(scores[0])

        if not 0.0 <= score <= 1.0:
            raise ValueError(
                "Single output is not a probability. "
                "Check the training model output."
            )

        dense_probability = score

    elif len(scores) >= 2:
        # Convert logits to probabilities if necessary.
        if (
            np.all(scores >= 0)
            and np.all(scores <= 1)
            and np.isclose(
                np.sum(scores), 1.0, atol=0.03
            )
        ):
            probabilities = scores / np.sum(scores)
        else:
            exp_scores = np.exp(
                scores - np.max(scores)
            )
            probabilities = (
                exp_scores / np.sum(exp_scores)
            )

        # Assumes class index 0 = dense_traffic,
        # class index 1 = sparse_traffic.
        dense_probability = float(probabilities[0])

    else:
        raise ValueError(
            f"Unexpected model output: {scores}"
        )

    prediction = (
        "DENSE_TRAFFIC"
        if dense_probability >= 0.5
        else "SPARSE_TRAFFIC"
    )

    confidence = (
        dense_probability
        if prediction == "DENSE_TRAFFIC"
        else 1.0 - dense_probability
    )

    print("Dense probability:",
          round(dense_probability, 5))
    print("Prediction:", prediction)
    print("======================================\n")

    return {
        "prediction": prediction,
        "confidence": round(confidence * 100, 2),
        "dense_probability": round(
            dense_probability * 100, 2
        ),
        "input_size": f"{width}x{height}",
        "raw_output": [
            round(float(v), 5) for v in scores
        ]
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/environment", methods=["GET", "POST"])
def environment_api():
    if request.method == "GET":
        return jsonify(environment)

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Send sensor data as a JSON object."
        }), 400

    for key in environment:
        if key in data and key != "updated_at":
            environment[key] = data[key]

    environment["updated_at"] = time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    return jsonify({
        "status": "received",
        "updated_at": environment["updated_at"]
    })


@app.route("/upload", methods=["POST"])
def upload_image():
    uploaded = request.files.get("image")

    if uploaded is None or not uploaded.filename:
        return jsonify({
            "error": "Please select an image."
        }), 400

    try:
        image = Image.open(
            io.BytesIO(uploaded.read())
        )

        result = predict_traffic(image)
        result["filename"] = uploaded.filename

        return jsonify(result)

    except Exception as error:
        app.logger.exception(
            "Traffic prediction failed"
        )

        return jsonify({
            "error": str(error)
        }), 500


if __name__ == "__main__":
    print("Starting ESP32 AI Dashboard...")
    print("Dashboard: http://127.0.0.1:5000")

    if not os.path.isfile(MODEL_PATH):
        print("WARNING: traffic_cnn.tflite not found.")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
