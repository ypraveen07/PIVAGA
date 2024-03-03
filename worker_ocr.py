from ai_controller.common_utils import *
import cv2
from datetime import datetime
from ai_controller.inference import *
from ai_controller.ai_settings import *
from pymongo import MongoClient
import torch
from paddleocr import PaddleOCR,draw_ocr 
import matplotlib.pyplot as plt
from datetime import datetime




class worker():
    def __init__(self) -> None:
        self.ocr = PaddleOCR(use_angle_cls=True) # Paddle OCR 
        self.predictor = Predictor()
        self.model_dir = "best.pt"               # checkpoint
        self.detector_dir = "ai_controller"      # main directory
        self.csv_file = 'D:\\lincode\\paddle_ocr\\interview\\data.csv' #output csv file
        self.unique_names = {}
        self.unique_names_list = []

    def load_model(self):
        # loading model
        global model
        model = torch.hub.load(self.detector_dir,'custom',path=self.model_dir,source = 'local',force_reload=True)

        #loading Video
        cap = cv2.VideoCapture('demo.mp4')
        if not cap.isOpened():
            print("Error opening video file")
            exit()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            #getting predictions
            predicted_frame, detector_predictions,cords  = self.predictor.run_inference_hub(model,frame)

            for cord in cords:
                cropped_img = crop_image(frame,cord) # crop the image
                ocr_result = ocr_prediction(cropped_img,self.ocr) # OCR Detection
                if ocr_result in txts1 or ocr_result in txts2:
                    ocr_result = refined_txt(ocr_result)
                else:
                    continue

                # creating time stamp
                ########################################
                if ocr_result in self.unique_names_list:
                    current_time = datetime.now().strftime('%H:%M:%S')
                    curr_val = self.unique_names[ocr_result]
                    st = curr_val[0]
                    et = current_time
                    self.unique_names[ocr_result] = [st,et]
                else:
                    current_time = datetime.now().strftime('%H:%M:%S')
                    self.unique_names[str(ocr_result)] = [current_time,current_time]
                    self.unique_names_list.append(ocr_result)

        write_to_csv(self.unique_names,self.csv_file)
    
        
        
if __name__ == "__main__":
    obj = worker()
    obj.load_model()