# app/app.py
import os, json
import gradio as gr
import onnxruntime as ort
from PIL import Image
import numpy as np

MODEL_PATH = "../experiments/model_best.onnx"
MAPPING_PATH = "../experiments/class_to_idx.json"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"ONNX model not found: {MODEL_PATH}")

if not os.path.exists(MAPPING_PATH):
    raise FileNotFoundError(f"class_to_idx.json not found: {MAPPING_PATH}")

with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    class_to_idx = json.load(f)
idx_to_class = {int(v): k for k, v in class_to_idx.items()}

session = ort.InferenceSession(MODEL_PATH)

def preprocess(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    image = image.convert("RGB").resize((224, 224))
    arr = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)[None, ...].astype(np.float32)
    return arr

def predict(image):
    arr = preprocess(image)
    outputs = session.run(None, {"input": arr})
    logits = np.squeeze(outputs[0]).astype(np.float32)
    probs = np.exp(logits) / np.sum(np.exp(logits))
    # build dict using idx_to_class
    label_scores = { idx_to_class[i]: float(probs[i]) for i in range(len(probs)) }
    top_idx = int(np.argmax(probs))
    top_name = idx_to_class[top_idx]
    top_str = f"🍝 {top_name} — {probs[top_idx]*100:.1f}%"
    return label_scores, top_str

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=[gr.Label(num_top_classes=len(idx_to_class)), gr.Markdown()],
    title="🍝 Pasta Classifier"
)

if __name__ == "__main__":
    demo.launch()
