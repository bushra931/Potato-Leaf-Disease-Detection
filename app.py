import os
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ---- CONFIG ----
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MODEL_PATH = "potato_disease_classification_model.keras"

# ---- INIT ----
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
model = load_model(MODEL_PATH)

# Class labels (must match your training order)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Knowledge base
DISEASE_INFO = {
    "Early Blight": {
        "desc": "Caused by Alternaria solani fungus. Produces concentric brown spots on older leaves.",
        "solution": [
            "Remove infected leaves",
            "Use crop rotation (2-3 years)",
            "Apply fungicides like Mancozeb or Chlorothalonil",
        ],
    },
    "Late Blight": {
        "desc": "Caused by Phytophthora infestans. Creates large, dark lesions; spreads rapidly in humid weather.",
        "solution": [
            "Destroy infected plants immediately",
            "Avoid overhead irrigation",
            "Use fungicides like Metalaxyl or Cymoxanil mixtures",
        ],
    },
    "Healthy": {
        "desc": "No disease detected.",
        "solution": [
            "Maintain proper irrigation",
            "Monitor regularly for early signs",
            "Ensure good soil nutrition",
        ],
    },
}

# ---- HELPERS ----
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_leaf(img_path):
    img = Image.open(img_path).resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    idx = np.argmax(preds)
    label = CLASS_NAMES[idx]
    confidence = float(preds[idx]) * 100
    return label, confidence

# ---- ROUTES ----
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            label, confidence = predict_leaf(filepath)
            info = DISEASE_INFO[label]

            return render_template("result.html",
                                   label=label,
                                   confidence=f"{confidence:.2f}%",
                                   info=info,
                                   img_path=filepath)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
