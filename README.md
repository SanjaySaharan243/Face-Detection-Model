🧑‍💻 Face Detection using OpenCV DNN

This project demonstrates face detection using OpenCV’s Deep Neural Network (DNN) module with a pre-trained Caffe model.

📌 Overview

Loads an image and applies deep learning–based face detection.

Uses OpenCV’s DNN face detector (res10_300x300_ssd_iter_140000.caffemodel) trained on the SSD (Single Shot Detector) framework with ResNet-10.

Draws bounding boxes around detected faces with a confidence score > 0.5.

🚀 Tech Stack

Python 3

OpenCV (cv2)

NumPy

Google Colab (for easy execution & file upload/preview)

📂 Project Structure 📁 Face-Detection-Project │── Face_Detection.ipynb # Main Colab notebook │── README.md # Project documentation

⚙️ Setup 1️⃣ Install dependencies pip install opencv-python opencv-python-headless numpy

2️⃣ Download Pre-trained Model Files

You need two files from OpenCV’s GitHub:

deploy.prototxt

res10_300x300_ssd_iter_140000.caffemodel

Place them in your project folder.

▶️ Usage Upload an image in Colab: from google.colab import files uploaded = files.upload()

image_path = list(uploaded.keys())[0] # Get uploaded image

Load model: net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

Detect faces: image = cv2.imread(image_path) (h, w) = image.shape[:2]

blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)) net.setInput(blob) detections = net.forward()

for i in range(0, detections.shape[2]): confidence = detections[0, 0, i, 2] if confidence > 0.5: box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) (startX, startY, endX, endY) = box.astype("int") cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

cv2_imshow(image) # Colab display

📊 Model Details

Model: ResNet-10 SSD

Framework: Caffe

Input Size: 300x300

Mean Subtraction Values: (104.0, 177.0, 123.0)

Output: Bounding boxes + confidence score

🎯 Example Output

✅ Upload an image → Faces detected → Green boxes drawn.

📝 Notes

Confidence threshold can be adjusted (0.5 → more strict/lenient).

Works well on frontal faces, struggles with extreme angles or occlusions.

You can extend this to real-time detection using a webcam (cv2.VideoCapture).

