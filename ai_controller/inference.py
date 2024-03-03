from concurrent.futures import thread
import torch
import os
import glob
import cv2
import numpy as np


class Predictor():
	def __init__(self):
		self.model_dir = r""
		self.weights_path =r''
		self.image_size = 640
		self.common_confidence = 0.2
		self.common_iou = 0.45
		self.line_thickness = None
		self.defects = ["crack"]

		self.features = []
		self.ind_thresh = {}

		self.rename_labels = {} # {'person':'manju'}
		## avoid labels with in the given co-ordinates
		self.avoid_labels_cords = [] #[{'xmin':0,'ymin':0,'xmax':1280,'ymax':720},{'xmin':0,'ymin':6,'xmax':569,'ymax':548}]
		self.avoid_required_labels = [] # ['person'] # ['person','cell phone']

		##
		self.detector_predictions = None # This will update from the predictions

	def load_model(self):
		model = torch.hub.load(self.model_dir,'custom',path=self.weights_path,source = 'local',force_reload=True)
		model.conf = self.common_confidence
		model.iou = self.common_iou
		return model

	def run_inference_hub(self,model, image):
		results = model(image,size=self.image_size)
		labels = results.pandas().xyxy[0]
		labels = list(labels['name'])
		result_dict = results.pandas().xyxy[0].to_dict()
		labels_ = []
		coordinates = []
		for i in range(len(labels)):
			xmin = list(result_dict.get('xmin').values())[i]
			ymin = list(result_dict.get('ymin').values())[i]
			xmax = list(result_dict.get('xmax').values())[i]
			ymax = list(result_dict.get('ymax').values())[i]
			c = list(result_dict.get('class').values())[i]
			name = list(result_dict.get('name').values())[i]
			confidence = list(result_dict.get('confidence').values())[i]
	
			## avoid labels with the given co ordinates
			skip = None
			if self.avoid_labels_cords:
				if bool(self.avoid_required_labels):
					for label in self.avoid_required_labels:
						if label == name:
							for crd in self.avoid_labels_cords:
								if round(xmin) >= crd['xmin'] and round(ymin) >= crd['ymin'] and round(xmax) <= crd['xmax'] and round(ymax) <= crd['ymax']:
									skip = True
				else:
					for crd in self.avoid_labels_cords:
						if round(xmin) >= crd['xmin'] and round(ymin) >= crd['ymin'] and round(xmax) <= crd['xmax'] and round(ymax) <= crd['ymax']:
							skip = True
			if skip :
				continue


			
			## line width
			line_width = self.line_thickness or max(round(sum(image.shape) / 2 * 0.003), 2)

			## Checking individual threshold for wach label 
			if name in self.ind_thresh:
				try:
					if self.ind_thresh.get(name) <= confidence:

						p1, p2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
						
			
						if name:
							namer = self.rename_labels.get(name)
							if namer is None:
								name = name
							else:
								name = namer			
							
							## Bounding color   
							if name in self.defects:
								color = (0,0,255) # Red color bounding box 
							else:
								color = (0,128,0) # Green color bounding box 


							cv2.rectangle(image, p1, p2, color, thickness=line_width, lineType=cv2.LINE_AA)
							
							tf = max(line_width - 1, 1)  # font thickness
							

							w, h = cv2.getTextSize(name, 0, fontScale=line_width / 3, thickness=tf)[0]  # text width, height
							outside = p1[1] - h - 3 >= 0  # label fits outside box
							p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
							cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
							coordinates.append({name:[int(xmin),int(ymin),int(xmax),int(ymax)]})
							
							cv2.putText(image, name, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, line_width / 3, (255,255,255),
										thickness=tf, lineType=cv2.LINE_AA)
								
							labels_.append(name)

							
				except:
					pass
			
			## If not individual threshold
			else:
				# line_width or max(round(sum(im.shape) / 2 * 0.003), 2)
				p1, p2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))	

				
			
				if name:
					namer = self.rename_labels.get(name)
					if namer is None:
						name = name
					else:
						name = namer
					
					## Bounding color   
					if name in self.defects:
						color = (0,0,255) # Red color bounding box 
					else:
						color = (0,128,0) # Green color bounding box
					
					cv2.rectangle(image, p1, p2, color, thickness=line_width, lineType=cv2.LINE_AA)

					
					tf = max(line_width - 1, 1)  # font thickness
					# tf = self.line_thickness
					w, h = cv2.getTextSize(name, 0, fontScale=line_width / 3, thickness=tf)[0]  # text width, height
					outside = p1[1] - h - 3 >= 0  # label fits outside box
					p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
					cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
					coordinates.append({name:[int(xmin),int(ymin),int(xmax),int(ymax)]})

					
					cv2.putText(image, name, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, line_width / 3, (255,255,255),
								thickness=tf, lineType=cv2.LINE_AA)
					labels_.append(name)			
					

		self.detector_predictions = labels_
		for img in results.ims:
			return img, labels_, coordinates

	def check_kanban(self):
		defect_list = []
		feature_list = []
		response = {}
		for i in self.detector_predictions:
			if i in self.defects:
				defect_list.append(i)
		
		for feature in self.features:
			if not feature in self.detector_predictions:
				feature_list.append(feature)

		if bool(defect_list) or bool(feature_list):
			is_accepted = "Rejected"
		else:
			is_accepted = "Accepted"
		response['status'] = is_accepted
		response['defects'] = defect_list
		response['features'] = feature_list
		return response


if __name__ == '__main__':
	predictor = Predictor()
	model = predictor.load_model()

