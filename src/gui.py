import cv2
import threading
from pathlib import Path

import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk

from detector import TrafficSignDetector


MODEL_PATH = "models/best.pt"


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Traffic Sign Recognition")
        self.geometry("1400x900")

        self.detector = TrafficSignDetector(MODEL_PATH)

        self.video_running = False
        self.cap = None

        self.create_ui()

    def create_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        # Левая панель
        self.left_panel = ctk.CTkFrame(self)
        self.left_panel.grid(row=0, column=0, sticky="nswe", padx=10, pady=10)

        self.image_btn = ctk.CTkButton(
            self.left_panel,
            text="Загрузить изображение",
            command=self.load_image
        )
        self.image_btn.pack(pady=10, fill="x")

        self.video_btn = ctk.CTkButton(
            self.left_panel,
            text="Открыть видео",
            command=self.load_video
        )
        self.video_btn.pack(pady=10, fill="x")

        # self.camera_btn = ctk.CTkButton(
        #     self.left_panel,
        #     text="Запустить камеру",
        #     command=self.start_camera
        # )
        # self.camera_btn.pack(pady=10, fill="x")

        self.stop_btn = ctk.CTkButton(
            self.left_panel,
            text="Остановить",
            command=self.stop_video
        )
        self.stop_btn.pack(pady=10, fill="x")

        self.info_box = ctk.CTkTextbox(self.left_panel, width=300)
        self.info_box.pack(pady=20, fill="both", expand=True)

        self.image_label = ctk.CTkLabel(self, text="")
        self.image_label.grid(row=0, column=1, sticky="nswe", padx=10, pady=10)

    def show_frame(self, frame, detections=None):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(rgb)
        image = image.resize((1000, 700))

        photo = ImageTk.PhotoImage(image=image)

        self.image_label.configure(image=photo)
        self.image_label.image = photo

        self.info_box.delete("0.0", "end")

        if detections:
            for det in detections:
                text = f"{det['class']} ({det['confidence']:.2f})\n"
                self.info_box.insert("end", text)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png")]
        )

        if not file_path:
            return

        frame = cv2.imread(file_path)

        annotated_frame, detections = self.detector.detect(frame)

        self.show_frame(annotated_frame, detections)

    def load_video(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video", "*.mp4 *.avi *.mov")]
        )

        if not file_path:
            return

        self.stop_video()

        self.cap = cv2.VideoCapture(file_path)
        self.video_running = True

        threading.Thread(target=self.video_loop, daemon=True).start()

    # def start_camera(self):
    #     self.stop_video()

    #     self.cap = cv2.VideoCapture(0)
    #     self.video_running = True

    #     threading.Thread(target=self.video_loop, daemon=True).start()

    def stop_video(self):
        self.video_running = False

        if self.cap:
            self.cap.release()
            self.cap = None

    def video_loop(self):
        while self.video_running and self.cap:
            ret, frame = self.cap.read()

            if not ret:
                break

            annotated_frame, detections = self.detector.detect(frame)

            self.show_frame(annotated_frame, detections)

        self.stop_video()


if __name__ == "__main__":
    app = App()
    app.mainloop()
