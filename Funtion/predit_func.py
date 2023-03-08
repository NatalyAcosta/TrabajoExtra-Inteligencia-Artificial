from ultralytics import YOLO
import cv2
model = YOLO('Models/best.pt') 
result = model('Data/Dataset\Test/frame_252.jpg')
cv2.imwrite('result.jpg', result[0].plot())




