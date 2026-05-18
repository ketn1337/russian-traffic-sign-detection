from ultralytics import YOLO
import cv2


class TrafficSignDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, verbose=False)

        annotated_frame = results[0].plot()
        detections = []

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = self.model.names[cls_id]

            detections.append({
                "class": class_name,
                "confidence": conf
            })

        return annotated_frame, detections
