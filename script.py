from ultralytics import YOLO as YOLOv10
import cv2
import dlib
import os

face_detector = YOLOv10("C:/Users/guilh/Desktop/facial/clustering-de-images/media/faceTest3.pt")

base = "C:/Users/guilh/Desktop/facial/clustering-de-images/uploads/"
path1 = os.path.join(base, f"imagem1.jpg")
path2 = os.path.join(base, f"imagem2.jpg")
image_path = os.path.join(base, "20241127112433.jpg")

image = cv2.imread(image_path)

detector = dlib.get_frontal_face_detector()

faces = detector(image, 1)

for face in faces:
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv2.rectangle(image, (x,y), (x + w, y + h), (0,255,0), 2)
    
cv2.imwrite(path2, image)

image = cv2.imread(image_path)

detected_faces = face_detector(image)[0]

for box in detected_faces.boxes:
    cords = box.xywh[0].cpu().numpy().tolist()
    x_center, y_center, width, height = cords
    x1 = int(x_center - width / 2) 
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
cv2.imwrite(path1, image)
