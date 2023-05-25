import cv2
import os
from PIL import Image
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  # Update with your YOLO weights and cfg paths
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
classes = open('coco.names').read().strip().split('\n')  # Update with your classes file path

# Directory where your images are stored
image_directory = 'data/images/'
images = os.listdir(image_directory)

for img_file in images:
    img_path = os.path.join(image_directory, img_file)
    image = cv2.imread(img_path)

    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Object detected
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non Maximum Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # If car detected, crop and resize
    for i in range(len(boxes)):
      if i in indexes and classes[class_ids[i]] == 'car':
          x, y, w, h = boxes[i]

          # Ensure coordinates are within image bounds
          x = max(0, x)
          y = max(0, y)
          w = min(w, width - x)
          h = min(h, height - y)

          crop_img = image[y:y+h, x:x+w]

          # Resize to 128x128
          im_pil = Image.fromarray(crop_img)
          im_pil = im_pil.resize((128, 128), Image.ANTIALIAS)
          # Save the cropped image
          im_pil.save(f'cropped/{img_file}')