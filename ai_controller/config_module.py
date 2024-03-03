import os
import sys

from utils.torch_utils import select_device
from detect_module import *

class opt_config():
    def __init__(self):
        self.base_path = ""
        self.detector_weights_path = ""
        self.detector_mask_weights_path =""

        # self.crop_detector_weights_path = '/home/maini/main/aiEngine/aiworker/weights/best.pt'
        self.separate_crop_model = False
        self.classifier_weights = ""
        self.segmentor_weights = ""
        self.ocr_weights = ""
        # self.batch_size_images = 6
        self.detector_input_image_size = 1280
        self.detector_mask_input_image_size = 640
        self.common_conf_thres = 0.1
        self.iou_thres = 0.2
        self.max_det = 1000
        self.device = ""
        self.line_thickness = 2
        self.hide_labels = False
        self.hide_conf = True
        self.half = False
        self.crop = False
        self.cord = []
        self.crop_class = ""
        self.min_crop_size = None
        self.max_crop_size = None
        self.crop_conf = 0.25
        self.crop_iou = 0.25
        self.padding  = 50
        # self.crop_resize = (640,640)
        self.crop_hide_labels = True
        self.crop_hide_conf = True
        self.classes = None
        self.defects = []
        self.feature = []
        self.features_extra = []
        # self.detector_weights_path = '/home/maini/main/aiEngine/aiworker/weights/best_22.pt'# working

        self.visualize = False
        self.individual_thres = {}#best_22.pt

        self.rename_labels = {} 
        ## avoid labels with in the given co-ordinates
        self.avoid_labels_cords = [{'xmin':0,'ymin':0,'xmax':1280,'ymax':720},{'xmin':0,'ymin':6,'xmax':569,'ymax':548}]
        self.avoid_required_labels = ['person'] # ['person','cell phone']
        ##
        self.detector_predictions = None 
       




