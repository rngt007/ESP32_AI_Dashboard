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

# --------------------------------------------------
# ENVIRONMENT SENSOR DATA
# --------------------------------------------------

environment = {
    "temperature": None,
    "humidity": None,
    "pressure": None,
    "gas": None,
    "duty_ratio": None,
    "pm25": None,
    "voc_grade": None,
    "voc_level": None,
    "aqi": None,
    "aqi_label": "Waiting for sensor data",
    "aqi_category": None,
    "environment_prediction": None,
    "abnormal_probability": None,
    "sound_db": None,
    "noise_status": None,
    "sound_threshold": 65.0,
    "updated_at": None
}

# --------------------------------------------------
# PM2.5 AQI CALCULATION
# --------------------------------------------------

def calculate_aqi(pm25):
    """
    Calculate an estimated US EPA-style AQI from PM2.5.

    Important:
    This uses a single PM2.5 reading. Official AQI reporting
    requires appropriate averaging, so this is an estimate.
    """

    if pm25 is None:
        return None, "Waiting for sensor data"

    try:
        pm25 = float(pm25)
    except (TypeError, ValueError):
        return None, "Invalid PM2.5 reading"

    if not np.isfinite(pm25) or pm25 < 0:
        return None, "Invalid PM2.5 reading"

    # EPA PM2.5 concentrations are truncated to 1 decimal place.
    pm25 = int(pm25 * 10) / 10.0

    breakpoints = [
        (0.0, 9.0, 0, 50, "Good"),
        (9.1, 35.4, 51, 100, "Moderate"),
        (35.5, 55.4, 101, 150, "Unhealthy for Sensitive Groups"),
        (55.5, 125.4, 151, 200, "Unhealthy"),
        (125.5, 225.4, 201, 300, "Very Unhealthy"),
        (225.5, 325.4, 301, 500, "Hazardous")
    ]

    for c_low, c_high, i_low, i_high, label in breakpoints:
        if c_low <= pm25 <= c_high:
            aqi = round(
                ((i_high - i_low) / (c_high - c_low))
                * (pm25 - c_low)
                + i_low
            )
            return aqi, label

    if pm25 > 325.4:
        return 500, "Hazardous"

    # Covers small gaps caused by the defined breakpoint ranges.
    return None, "Outside AQI calculation range"


# --------------------------------------------------
# TRAFFIC MODEL
# --------------------------------------------------

interpreter = None
input_details = None
output_details = None
model_lock = threading.Lock()


def load_model():
    global interpreter, input_details, output_details

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Traffic model not found: {MODEL_PATH}"
        )

    import tensorflow as tf

    interpreter = tf.lite.Interpreter(
        model_path=MODEL_PATH
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Traffic model loaded successfully.")
    print("Model input:", input_details)
    print("Model output:", output_details)


def predict_traffic(image):
    global interpreter, input_details, output_details

    if interpreter is None:
        load_model()

    input_info = input_details[0]
    shape = input_info["shape"]

    if len(shape) != 4 or int(shape[3]) != 3:
        raise ValueError(
            f"Expected an image model input shaped [1, height, width, 3]. "
            f"Received: {shape}"
        )

    height = int(shape[1])
    width = int(shape[2])

    image = image.convert("RGB").resize(
        (width, height),
        Image.Resampling.BILINEAR
    )

    pixels = np.asarray(image, dtype=np.float32)
    dtype = input_info["dtype"]

    if dtype == np.float32:
        # Assumes any required normalization is already part
        # of the model, or that the model expects pixel values.
        pixels = pixels.astype(np.float32)

    elif dtype in (np.uint8, np.int8):
        scale, zero_point = input_info["quantization"]

        if scale <= 0:
            raise ValueError(
                "The quantized model input has an invalid scale."
            )

        pixels = np.round(pixels / scale + zero_point)

        limits = np.iinfo(dtype)
        pixels = np.clip(
            pixels,
            limits.min,
            limits.max
        ).astype(dtype)

    else:
        raise ValueError(
            f"Unsupported model input data type: {dtype}"
        )

    pixels = np.expand_dims(pixels, axis=0)

    with model_lock:
        interpreter.set_tensor(
            input_info["index"],
            pixels
        )
        interpreter.invoke()

        raw_output = interpreter.get_tensor(
            output_details[0]["index"]
        )

    scores = np.asarray(
        raw_output[0],
        dtype=np.float32
    ).flatten()

    if len(scores) == 0:
        raise ValueError("The traffic model returned empty output.")

    # Convert model output to probabilities.
    if len(scores) == 1:
        dense_probability = float(scores[0])

    elif (
        np.all(scores >= 0)
        and np.all(scores <= 1)
        and np.isclose(np.sum(scores), 1.0, atol=0.01)
    ):
        # Output already appears to be probabilities.
        dense_probability = float(scores[0])

    else:
        # Treat output as logits and apply softmax.
        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted)
        probabilities = exp_scores / np.sum(exp_scores)
        dense_probability = float(probabilities[0])

    dense_probability = float(
        np.clip(dense_probability, 0.0, 1.0)
    )

    # Assumes output class index 0 = dense traffic.
    # Assumes output class index 1 = sparse traffic.
    if dense_probability >= 0.5:
        prediction = "DENSE_TRAFFIC"
        confidence = dense_probability
    else:
        prediction = "SPARSE_TRAFFIC"
        confidence = 1.0 - dense_probability

    return {
        "prediction": prediction,
        "confidence": round(confidence, 6),
        "dense_probability": round(dense_probability, 6),
        "input_size": [width, height],
        "raw_output": np.asarray(raw_output).tolist()
    }


# --------------------------------------------------
# DASHBOARD HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# ESP32 ENVIRONMENT API
# --------------------------------------------------

@app.route("/api/environment", methods=["GET", "POST"])
def environment_api():

    # Return the latest sensor values.
    if request.method == "GET":
        return jsonify(environment)

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Send sensor data as a JSON object."
        }), 400

    # Update only recognized fields.
    # AQI is calculated on the server and cannot be overwritten
    # by a value sent by the ESP32.
    protected_fields = {
        "updated_at",
        "aqi",
        "aqi_label",
        "aqi_category"
    }

    for key in environment:
        if key in data and key not in protected_fields:
            environment[key] = data[key]

    # Calculate AQI from the latest PM2.5 reading.
    aqi, aqi_label = calculate_aqi(environment["pm25"])

    environment["aqi"] = aqi
    environment["aqi_label"] = aqi_label
    environment["aqi_category"] = aqi_label

    environment["updated_at"] = time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    app.logger.info(
        "Environment updated: PM2.5=%s, estimated AQI=%s, label=%s",
        environment["pm25"],
        aqi,
        aqi_label
    )

    return jsonify({
        "status": "received",
        "updated_at": environment["updated_at"],
        "pm25": environment["pm25"],
        "aqi": aqi,
        "aqi_label": aqi_label
    }), 200


# --------------------------------------------------
# TRAFFIC IMAGE UPLOAD API
# --------------------------------------------------

@app.route("/upload", methods=["POST"])
def upload_image():

    uploaded = request.files.get("image")

    if uploaded is None or not uploaded.filename:
        return jsonify({
            "error": "Please select an image."
        }), 400

    try:
        image_bytes = uploaded.read()

        if not image_bytes:
            return jsonify({
                "error": "The uploaded image is empty."
            }), 400

        image = Image.open(io.BytesIO(image_bytes))

        result = predict_traffic(image)
        result["filename"] = uploaded.filename

        app.logger.info(
            "Traffic prediction: %s, confidence=%s",
            result["prediction"],
            result["confidence"]
        )

        return jsonify(result), 200

    except Exception as error:
        app.logger.exception("Traffic prediction failed")

        return jsonify({
            "error": str(error)
        }), 500


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "ESP32 AI Dashboard"
    }), 200


# --------------------------------------------------
# START APPLICATION LOCALLY
# --------------------------------------------------

if __name__ == "__main__":
    print("Starting ESP32 AI Dashboard...")
    print("Dashboard: http://127.0.0.1:5000")

    if not os.path.isfile(MODEL_PATH):
        print(
            f"WARNING: Traffic model not found at {MODEL_PATH}"
        )

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
