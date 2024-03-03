import os
import sys
from PIL import Image
# import imagehash
import cv2
import argparse
import shutil
import redis
from pymongo import MongoClient
# from bson import ObjectId
import json
from ai_controller.ai_settings import *
import ai_controller.ai_settings as settings
import datetime
import csv
import os

def cv2_pil(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil



def insert_inspection_col(collection_name ,mydict ):
    myclient = MongoClient("mongodb://localhost:27017/")
    mydb = myclient["Indo_trial"]
    mycol = mydb[str(collection_name)]
    # mydict = { "name": "John", "address": "Highway 37" }
    print(mycol ,mydict )
    _x = mycol.insert_one(mydict)
    return _x




def create_temp_folder1(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)  # delete existing lotmark folder
    os.makedirs(directory)        # creating new lotmark folder

def create_folder(directory):
    if os.path.exists(directory):
        pass
    else:
        os.makedirs(directory) 
    print(f'{directory} is created!!')


#Key Builder
def read_json_file(json_file):
    with open(json_file,'r') as f:
        data = json.load(f)
        f.close()
        return data





def store_input_image(frame , status ,opt , img_name ,stage):
    x = datetime.datetime.now()
    date = x.strftime("%d_%m_%Y")
    root_dir = os.path.join(opt.input_image_path ,str(stage))
    storage_dir = os.path.join(root_dir ,date)
    if status :
        folder = os.path.join(storage_dir  , "accepted")
    else:
        folder = os.path.join(storage_dir , "rejected")
    create_folder(directory = folder)
    img_path = folder + "/"+ str(img_name)+".jpg"

    cv2.imwrite(img_path , frame) 
    return img_path
        
def store_output_image(frame , status ,opt , img_name ,stage):
    x = datetime.datetime.now()
    date = x.strftime("%d_%m_%Y")
    # storage_dir = os.path.join(opt.output_image_path,date)
    root_dir = os.path.join(opt.output_image_path ,str(stage))
    storage_dir = os.path.join(root_dir ,date)
    if status :
        folder = os.path.join(storage_dir , "accepted")
    else:
        folder = os.path.join(storage_dir  , "rejected")
    create_folder(directory = folder)
    img_path = folder + "/"+ str(img_name)+".jpg"
    cv2.imwrite(img_path , frame) 
    return img_path

txts1 = ['CARVE','CARVI','CARV','CARVEL']
txts2 = ['EMO', 'LEMON', 'MONI', 'MOND', 'EMONDE', 'EMOND', 'EMONE', 'MON', 'HMON', 'UMO', 'MOSSBROS', 'MOSSBRO','BROS']
def refined_txt(txt):
    if txt in txts1:
        return 'CARVELA'
    if txt in txts2:
        return 'MOSS BROS'

def crop_image(image,cord):
    if list(cord.keys())[0] == 'SHOP_NAME':
        xmin,ymin,xmax,ymax = list(cord.values())[0][0],list(cord.values())[0][1],list(cord.values())[0][2],list(cord.values())[0][3]
        cropped_image = image[ymin:ymax, xmin:xmax]
        return cropped_image
    
def save_ocr(result):
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    return txts


def ocr_prediction(cropped_img,ocr):
    result = ocr.ocr(cropped_img)
    for result in result:
        result = result
        txts = save_ocr( result)
    if len(txts)>0:
        return txts[0]
    else:
        return ""


def write_to_csv(dict1,filename):
    keys = list(dict1.keys())
    values = list(dict1.values())
    start_column = [value[0] for value in values]
    end_column = [value[1] for value in values]
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['', 'start', 'end'])
        for key, start, end in zip(keys, start_column, end_column):
            writer.writerow([key, start, end])


