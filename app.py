from flask import Flask, request, jsonify
import cv2
import numpy as np
import io
from PIL import Image
from gaze_tracking import GazeTracking

app = Flask(__name__)
gaze_tracker = GazeTracking()

@app.route("/predict-gaze", methods=["POST"])
def predict_gaze():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        gaze_tracker.refresh(frame)

        if not gaze_tracker.pupils_located:
            return jsonify({"status": "face_not_detected_or_pupils_not_found"}), 200

        return jsonify({
            "status": "ok",
            "is_blinking": gaze_tracker.is_blinking(),
            "is_looking_left": gaze_tracker.is_left(),
            "is_looking_right": gaze_tracker.is_right(),
            "is_looking_center": gaze_tracker.is_center(),
            "pupil_left_coords": gaze_tracker.pupil_left_coords(),
            "pupil_right_coords": gaze_tracker.pupil_right_coords(),
            "horizontal_ratio": gaze_tracker.horizontal_ratio(),
            "vertical_ratio": gaze_tracker.vertical_ratio()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
