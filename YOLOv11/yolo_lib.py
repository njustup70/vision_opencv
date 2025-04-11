from ultralytics import YOLO
import numpy as np
import cv2
class MyYOLO():
    def __init__(self, model_path,show=False):
        self.model = YOLO(model_path)
        self.show=show
    def update(self,image:np.ndarray,content:dict):
        results=self.model(image,self.show)
        # results.print()