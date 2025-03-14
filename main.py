from ultralytics import YOLO
import os
import multiprocessing

def main():
    # Initialize model
    model = YOLO("yolo11n-cls.pt")
    
    data_path = "C:/Users/Lenovo/OneDrive/Desktop/YoloV8/Formula1"
    
    #train model
    results = model.train(
        data=data_path,
        epochs=20,
        imgsz=480,        # Reduced from 640
        batch=4,          #  reduced from 8
        lr0=0.01,
        lrf=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        save=True,
        save_period=5,
        patience=50,
        device=0,
        workers=2,        # Reduced from 4
        val=True,
        plots=True,
        cache=False       #  no caching in RAM
    )

    # Save model
    model.save('best_model.pt')

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()