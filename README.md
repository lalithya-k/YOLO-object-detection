# YOLO-object-detection

A deep learning-based system designed to detect pedestrian safety violations using YOLOv8 and YOLOv9 models. This project aims to improve urban safety through real-time monitoring and accurate violation detection.

## **Features**
- **Real-Time Detection** using YOLO models.
- **High Accuracy** with 91% detection performance.
- **Optimized for Scalability** in smart city surveillance systems.
- 
  ![Screenshot 2025-04-11 171144](https://github.com/user-attachments/assets/c1cc9219-08a5-45fa-991e-59b47ba6a019)

## **Technologies Used**
- **Languages:** Python  
- **Libraries:** YOLOv8, YOLOv9, TensorFlow, PyTorch, OpenCV, WandB  
- **Environment:** Kaggle Notebooks  

## **Dataset**
Annotated pedestrian images from urban environments, split into training, validation, and test sets (80-10-10 split).

## **Model Overview**
- **YOLOv8 & YOLOv9:** For object detection with optimized accuracy and speed.  
- **Techniques:** Data augmentation, hyperparameter tuning, and TensorFlow optimizations.

## **Installation & Usage**

1. **Clone the Repo & Install Dependencies:**
   ```bash
   git clone https://github.com/your-repo/pedestrian-safety-detection.git
   cd pedestrian-safety-detection
   pip install -r requirements.txt
   pip install ultralytics supervision datasets pyyaml
   ```

2. **Train the Model:**
   ```python
   from ultralytics import YOLO
   model = YOLO('yolov8n.pt')  # or 'yolov9n.pt'
   model.train(data='data.yaml', epochs=50, imgsz=640)
   ```

3. **Run Inference:**
   ```python
   results = model.predict(source='path_to_image_or_video')
   results.show()
   ```

## **Results**
- **Accuracy:** 91%  
- **Precision & Recall:** Optimized for minimal false positives/negatives  
- **Fast Inference:** Suitable for real-time deployment
