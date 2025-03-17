import gradio as gr
from ultralytics import YOLO
import numpy as np

# Load the trained YOLO model
model = YOLO("C:/Users/Lenovo/OneDrive/Desktop/YoloV8/best_model.pt")

# Prediction function
def predict(image):
    results = model(image)
    result = results[0]  # Extract first result
    names_dict = model.names  

    # Get class probabilities
    if hasattr(result, "probs") and result.probs is not None:
        probs = result.probs.data.tolist()
        predicted_class = names_dict[np.argmax(probs)]
        confidence_score = max(probs) * 100
    else:
        predicted_class, confidence_score = "N/A", 0.0

    # Format output
    result_text = (
        f" Predicted Class: `{predicted_class}`\n"
        f" Confidence: `{confidence_score:.2f}%`"
    )
    
    return result_text

# Function to clear inputs and outputs
def clear_inputs():
    return None, ""

# Gradio Interface 
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("## YOLOv8 F1 Team Livery Detection")
    gr.Markdown("Upload an image and CLICK PREDICT to detect the F1 team based on livery")
    gr.Markdown("Model trained on McLaren, Ferrari, Redbull, Racing Point, Mercedes, and Williams F1 teams.")

    with gr.Row():
        image_input = gr.Image(
            type="pil", 
            label="Upload Image",
            sources=["upload"]  
        )
        prediction_output = gr.Textbox(label="🔍 Prediction Info", interactive=False)

    with gr.Row():
        predict_btn = gr.Button(" Predict")
        clear_btn = gr.Button(" Clear")

    predict_btn.click(predict, inputs=image_input, outputs=prediction_output)
    clear_btn.click(clear_inputs, inputs=[], outputs=[image_input, prediction_output])

# Run the app
if __name__ == "__main__":
    demo.launch()
