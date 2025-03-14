from ultralytics import YOLO
import numpy as np

# Loading our  model
model = YOLO("yolo11n-cls.pt")  
model = YOLO("C:/Users/Lenovo/OneDrive/Desktop/YoloV8/best_model.pt")  #Best model

image_paths = [
    "Predictions/Racing Point.jpg",
    "Predictions/Ferrari.jpg",
    "Predictions/Merc.jpg",
    "Predictions/Mclaren.jpg",
    "Predictions/Redbull.jpg",
    "Predictions/William.jpg",
]

results_list = model(image_paths)


# Print header
print(f"{'Image Path':<30} | {'Predicted Class':<30} | {'Confidence':<10}")
print("-" * 75)

# Print data rows
for i, results in enumerate(results_list):
    names_dict = results.names
    probs = results.probs.data.tolist()
    
    predicted_class = names_dict[np.argmax(probs)]
    confidence_score = max(probs) * 100
    
    print(f"{image_paths[i]:<30} | {predicted_class:<30} | {confidence_score:.2f}%")
