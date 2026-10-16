from flask import Flask, render_template, request, jsonify, Response
from PIL import Image, UnidentifiedImageError
import numpy as np
import os
import io
import time
import threading

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "traffic_cnn.tflite")

# =========================================================
# ENVIRONMENT SENSOR DATA
# =========================================================

environment = {
    "temperature": None,
    "humidity": None,
    "pressure": None,
    "gas": None,
    "duty_ratio": None,
    "pm25": None,
    "voc_level": None,
    "voc_grade": None,
    "aqi": None,
    "aqi_label": "Waiting for sensor data",
    "aqi_category": None,
    "environment_prediction": None,
    "abnormal_probability": None,
    "sound_db": None,
    "noise_status": None,
    "sound_threshold": 80.0,
    "updated_at": None
}

environment_lock = threading.Lock()


# =========================================================
# AQI CALCULATION
# =========================================================

def calculate_aqi(pm25):
    try:
        if pm25 is None:
            return None, "Waiting for sensor data"

        pm25 = float(pm25)

        if not np.isfinite(pm25) or pm25 < 0:
            return None, "Invalid PM2.5 reading"

        pm25 = int(pm25 * 10) / 10.0

        breakpoints = [
            (0.0, 9.0, 0, 50, "Good"),
            (9.1, 35.4, 51, 100, "Moderate"),
            (
                35.5, 55.4, 101, 150,
                "Unhealthy for Sensitive Groups"
            ),
            (55.5, 125.4, 151, 200, "Unhealthy"),
            (125.5, 225.4, 201, 300, "Very Unhealthy"),
            (225.5, 325.4, 301, 500, "Hazardous")
        ]

        for c_low, c_high, i_low, i_high, label in breakpoints:
            if c_low <= pm25 <= c_high:
                aqi = round(
                    (i_high - i_low)
                    / (c_high - c_low)
                    * (pm25 - c_low)
                    + i_low
                )
                return aqi, label

        if pm25 > 325.4:
            return 500, "Hazardous"

        return None, "Outside AQI calculation range"

    except (TypeError, ValueError, OverflowError):
        return None, "Invalid PM2.5 reading"


# =========================================================
# TRAFFIC AI MODEL
# =========================================================

interpreter = None
input_details = None
output_details = None

# Keep the model lock separate from the sensor-data lock.
model_lock = threading.Lock()


def load_model():
    global interpreter, input_details, output_details

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            "traffic_cnn.tflite is missing from the repository root."
        )

    import tensorflow as tf

    interpreter = tf.lite.Interpreter(
        model_path=MODEL_PATH
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Traffic model loaded successfully.")


def predict_traffic(image):
    global interpreter, input_details, output_details

    with model_lock:
        if interpreter is None:
            load_model()

        info = input_details[0]
        shape = info["shape"]

        if (
            len(shape) != 4
            or shape[0] != 1
            or shape[3] != 3
        ):
            raise ValueError(
                f"Unsupported model input shape: {shape}"
            )

        height = int(shape[1])
        width = int(shape[2])

        image = image.convert("RGB").resize((width, height))

        pixels = np.asarray(image)
        dtype = info["dtype"]

        if dtype == np.float32:
            pixels = pixels.astype(np.float32)

        elif dtype in (np.uint8, np.int8):
            scale, zero_point = info["quantization"]

            if scale > 0:
                pixels = np.round(
                    pixels / scale + zero_point
                )
            else:
                pixels = pixels.astype(dtype)

            limits = np.iinfo(dtype)

            pixels = np.clip(
                pixels,
                limits.min,
                limits.max
            ).astype(dtype)

        else:
            raise ValueError(
                f"Unsupported model data type: {dtype}"
            )

        pixels = np.expand_dims(pixels, axis=0)

        interpreter.set_tensor(
            info["index"],
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
        raise ValueError("The model returned no prediction.")

    if len(scores) == 1:
        dense_probability = float(scores[0])

        if not 0 <= dense_probability <= 1:
            dense_probability = 1.0 / (
                1.0 + np.exp(
                    -np.clip(dense_probability, -80, 80)
                )
            )

    elif (
        np.all(scores >= 0)
        and np.all(scores <= 1)
        and np.isclose(np.sum(scores), 1.0, atol=0.01)
    ):
        # Class index 0 = dense_traffic
        dense_probability = float(scores[0])

    else:
        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted)
        probabilities = exp_scores / np.sum(exp_scores)

        dense_probability = float(probabilities[0])

    dense_probability = float(
        np.clip(dense_probability, 0.0, 1.0)
    )

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
        "input_size": [width, height]
    }


# =========================================================
# LATEST ESP32-CAM FRAME
# =========================================================

camera_lock = threading.Lock()

latest_camera_frame = None
latest_camera_prediction = None
latest_camera_time = None


# =========================================================
# DASHBOARD HOME
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# ENVIRONMENT API
# ESP32 POSTs readings; dashboard GETs readings.
# =========================================================

@app.route("/api/environment", methods=["GET", "POST"])
def environment_api():

    if request.method == "GET":
        with environment_lock:
            return jsonify(environment), 200

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Request must contain a JSON object."
        }), 400

    protected = {
        "updated_at",
        "aqi",
        "aqi_label",
        "aqi_category"
    }

    with environment_lock:
        for key in environment:
            if key in data and key not in protected:
                environment[key] = data[key]

        aqi, label = calculate_aqi(
            environment["pm25"]
        )

        environment["aqi"] = aqi
        environment["aqi_label"] = label
        environment["aqi_category"] = label

        environment["updated_at"] = time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        response = {
            "status": "received",
            "pm25": environment["pm25"],
            "aqi": aqi,
            "aqi_label": label,
            "updated_at": environment["updated_at"]
        }

    app.logger.info(
        "ESP32 sensor data received: %s",
        response
    )

    return jsonify(response), 200


# =========================================================
# MANUAL TRAFFIC IMAGE UPLOAD
# =========================================================

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
                "error": "The image is empty."
            }), 400

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image.load()

        result = predict_traffic(image)
        result["filename"] = uploaded.filename

        return jsonify(result), 200

    except (UnidentifiedImageError, OSError):
        return jsonify({
            "error": "The uploaded file is not a valid image."
        }), 400

    except Exception:
        app.logger.exception(
            "Traffic prediction failed"
        )

        return jsonify({
            "error": (
                "Traffic prediction failed. "
                "Check Render logs and model configuration."
            )
        }), 500


# =========================================================
# ESP32-CAM UPLOAD
#
# ESP32-CAM sends the raw JPEG bytes using:
# Content-Type: image/jpeg
# POST /api/camera/frame
# =========================================================

@app.route("/api/camera/frame", methods=["POST"])
def receive_camera_frame():
    global latest_camera_frame
    global latest_camera_prediction
    global latest_camera_time

    image_bytes = request.get_data(cache=False)

    if not image_bytes:
        return jsonify({
            "error": "No image data received."
        }), 400

    # Keep uploads small to protect the server.
    if len(image_bytes) > 1_500_000:
        return jsonify({
            "error": "Image is too large. Reduce camera resolution."
        }), 413

    try:
        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image.load()

        if image.format != "JPEG":
            return jsonify({
                "error": "Please upload a JPEG image."
            }), 400

        image = image.convert("RGB")

        # Run the existing traffic AI model.
        result = predict_traffic(image)

        with camera_lock:
            latest_camera_frame = image_bytes
            latest_camera_prediction = result
            latest_camera_time = time.time()

        app.logger.info(
            "Camera frame received; prediction=%s",
            result["prediction"]
        )

        return jsonify({
            "success": True,
            "message": "Camera frame received successfully.",
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "dense_probability": result["dense_probability"],
            "input_size": result["input_size"]
        }), 200

    except (UnidentifiedImageError, OSError):
        return jsonify({
            "error": "The received image is invalid."
        }), 400

    except Exception:
        app.logger.exception(
            "ESP32-CAM frame processing failed"
        )

        return jsonify({
            "error": (
                "Camera image processing failed. "
                "Check the Render logs."
            )
        }), 500


# =========================================================
# GET LATEST CAMERA IMAGE
# Dashboard refreshes this image periodically.
# =========================================================

@app.route("/api/camera/latest.jpg", methods=["GET"])
def latest_camera_image():

    with camera_lock:
        frame = latest_camera_frame

    if frame is None:
        return jsonify({
            "error": "Waiting for the first camera frame."
        }), 404

    return Response(
        frame,
        mimetype="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate"
        }
    )


# =========================================================
# GET CAMERA STATUS AND LATEST AI PREDICTION
# =========================================================

@app.route("/api/camera/status", methods=["GET"])
def camera_status():

    with camera_lock:
        prediction = latest_camera_prediction
        frame_time = latest_camera_time

    if frame_time is None:
        return jsonify({
            "connected": False,
            "message": "Waiting for ESP32-CAM uploads.",
            "prediction": None
        }), 200

    age = max(0, time.time() - frame_time)

    return jsonify({
        "connected": age < 15,
        "last_frame_seconds_ago": round(age, 1),
        "prediction": prediction
    }), 200


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "ESP32 AI Dashboard"
    }), 200


# =========================================================
# START LOCAL DEVELOPMENT SERVER
# Render normally starts the app using Gunicorn.
# =========================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
