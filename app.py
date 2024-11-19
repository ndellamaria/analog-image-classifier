import gradio as gr
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="ndellamaria/analog-image-classifier")

def predict(input_img):
    predictions = pipeline(input_img)
    return input_img, {p["label"]: p["score"] for p in predictions} 

gradio_app = gr.Interface(
    predict,
    inputs=gr.Image(label="Select analog image", sources=['upload'], type="pil"),
    outputs=[gr.Image(label="Processed Image"), gr.Label(label="Result", num_top_classes=5)],
    title="Quality?",
)

if __name__ == "__main__":
    gradio_app.launch()