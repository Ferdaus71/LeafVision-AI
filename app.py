import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import time
import io

# ---------------- CONFIG ----------------
MODEL_PATH = "leaf_classifier.tflite"
CLASSES_FILE = "classes.txt"
IMG_SIZE = (224, 224)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_tflite_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model(MODEL_PATH)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------- LOAD CLASSES ----------------
with open(CLASSES_FILE, "r") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

# ---------------- FUNCTIONS ----------------
def preprocess(frame):
    img = cv2.resize(frame, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img / 127.5) - 1.0
    img = np.expand_dims(img, axis=0)
    return img

def predict(frame):
    inp = preprocess(frame)
    interpreter.set_tensor(input_details[0]["index"], inp)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]["index"])[0]
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs

def add_label_to_image(image, label, conf):
    img = image.copy()
    cv2.putText(img, f"{label} ({conf*100:.1f}%)", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 100), 2)
    return img

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="🌿 LeafVision AI", page_icon="🍃", layout="wide")

# ---------------- ANIMATED BACKGROUND ----------------
st.markdown("""
<style>
body {
    margin: 0;
    overflow: hidden;
    background-color: #001b12;
    color: #d9f7e8;
    font-family: 'Poppins', sans-serif;
}

canvas#leafCanvas {
    position: fixed;
    top: 0; left: 0;
    z-index: -1;
}

h1, h2, h3, h5, p {
    text-align: center;
}

.card {
    background: rgba(0, 40, 25, 0.7);
    border-radius: 20px;
    box-shadow: 0 0 30px rgba(0, 255, 100, 0.3);
    padding: 30px;
    margin: 20px auto;
    text-align: center;
    width: 90%;
    backdrop-filter: blur(10px);
}

.btn {
    background-color: #00d884;
    border: none;
    border-radius: 10px;
    padding: 10px 25px;
    color: #001b12;
    font-weight: bold;
    cursor: pointer;
    transition: 0.3s;
}
.btn:hover {
    background-color: #00ffa2;
    transform: scale(1.05);
}

.footer-card {
    background: linear-gradient(90deg, #00ff99, #00ffaa, #00ffcc);
    background-size: 400% 400%;
    animation: gradientGlow 8s ease infinite;
    color: #001b12;
    border-radius: 15px;
    text-align: center;
    padding: 25px;
    margin-top: 50px;
    box-shadow: 0 0 30px rgba(0,255,150,0.5);
}

@keyframes gradientGlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.social-icon {
    margin: 0 8px;
    transition: transform 0.3s;
}
.social-icon:hover {
    transform: scale(1.3);
}
</style>

<canvas id="leafCanvas"></canvas>
<script>
const canvas = document.getElementById('leafCanvas');
const ctx = canvas.getContext('2d');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

const leaves = [];
const leafCount = 30;
for (let i = 0; i < leafCount; i++) {
    leaves.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        size: 20 + Math.random() * 20,
        speed: 0.5 + Math.random(),
        drift: (Math.random() - 0.5) * 1.5,
        opacity: 0.4 + Math.random() * 0.6
    });
}

function drawLeaf(l) {
    ctx.beginPath();
    ctx.ellipse(l.x, l.y, l.size/2, l.size/3, Math.PI/4, 0, 2*Math.PI);
    ctx.fillStyle = `rgba(0,255,100,${l.opacity})`;
    ctx.fill();
}

function animate() {
    ctx.clearRect(0,0,canvas.width,canvas.height);
    leaves.forEach(l => {
        drawLeaf(l);
        l.y += l.speed;
        l.x += l.drift;
        if (l.y > canvas.height) l.y = -20;
        if (l.x > canvas.width || l.x < 0) l.x = Math.random() * canvas.width;
    });
    requestAnimationFrame(animate);
}
animate();

window.addEventListener('resize', () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
});
</script>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<h1>🍃 LeafVision AI</h1>", unsafe_allow_html=True)
st.markdown("<p>🌿 Identify tree species instantly using AI and your camera!</p>", unsafe_allow_html=True)

# ---------------- MAIN CARD ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
mode = st.radio("Choose Mode:", ["📁 Upload Image", "📷 Live Webcam"])

# ---------------- UPLOAD MODE ----------------
if mode == "📁 Upload Image":
    img_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
    if img_file:
        image = Image.open(img_file)
        st.image(image, caption="Uploaded Leaf", use_container_width=True)
        with st.spinner("Analyzing leaf... 🌱"):
            frame = np.array(image.convert("RGB"))
            label, conf, probs = predict(frame)
        st.success(f"🌳 **{label}** — Confidence: `{conf*100:.2f}%`")
        st.bar_chart(dict(zip(CLASS_NAMES, probs)))

        # ✅ Add label on image for download
        labeled_image = add_label_to_image(np.array(image.convert("RGB")), label, conf)
        img_pil = Image.fromarray(cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        byte_img = buf.getvalue()

        st.download_button(
            label="📸 Download Result Image",
            data=byte_img,
            file_name=f"{label}_result.png",
            mime="image/png",
        )

# ---------------- LIVE MODE ----------------
elif mode == "📷 Live Webcam":
    run = st.checkbox("Start Camera 🎥")
    FRAME_WINDOW = st.image([])
    cap = None
    capture_img = None

    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Webcam not available.")
        else:
            st.info("✅ Webcam active. Uncheck to stop.")

    while run and cap and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Frame not captured.")
            break

        label, conf, _ = predict(frame)
        frame_labeled = add_label_to_image(frame, label, conf)
        FRAME_WINDOW.image(cv2.cvtColor(frame_labeled, cv2.COLOR_BGR2RGB),
                           channels="RGB", use_container_width=True)
        capture_img = frame_labeled
        time.sleep(0.1)

    if cap and cap.isOpened():
        cap.release()
        cv2.destroyAllWindows()
        st.success("Camera stopped.")

    if capture_img is not None:
        img_pil = Image.fromarray(cv2.cvtColor(capture_img, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        byte_img = buf.getvalue()
        st.download_button(
            label="📸 Download Captured Result",
            data=byte_img,
            file_name="leafvision_result.png",
            mime="image/png",
        )

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER / DEVELOPER CARD ----------------
st.markdown("""
<div class="footer-card">
    <h3>🧑‍💻 Developed by: <b>Md. Ferdaus Hossen</b> 🧑‍💻</h3>
    <h5>Junior AI/ML Engineer at Zensoft Lab</h5>
    <p>
        <a href="https://github.com/Ferdaus71" target="_blank" class="social-icon">
            <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/github.svg" width="25" height="25" alt="GitHub">
        </a>
        <a href="https://www.linkedin.com/in/ferdaus70/" target="_blank" class="social-icon">
            <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg" width="25" height="25" alt="LinkedIn">
        </a>
    </p>
</div>
""", unsafe_allow_html=True)
