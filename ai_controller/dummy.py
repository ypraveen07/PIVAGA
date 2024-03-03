# from worker import *


# x = controller.read_holding_register(9)
# print(x)




# import socket
# import time
# s = socket.socket()
# s.connect(('192.168.100.101',9004))
# s.send('LON\r'.encode())
# s.settimeout(2)
# # time.sleep(1)
# for i in range(2):
#     try:
#         data = s.recv(1024).decode().split('\r')[0]
#         print(data)
#     except socket.timeout as e:
#         print('Timeout Part is rejected')
#     except Exception as e:
#         print(e)


# ip = '192.168.0.100'
# import plc_controller
# import sys
# controller = plc_controller.ModbusController()
# status = controller.connect(ip,mode='TCP')
# # if not status:
# 	sys.exit(0)
# x = controller.read_holding_register(12)
# print(12,x)
# y = controller.read_holding_register(13)
# print(13,y)

# controller.write_holding_register(13,63)
# controller.write_coil(40,1)
# controller.write_holding_register(40,1)
# print(controller.read_holding_register(30))


features = ['a','b','d']

pred = {'top': {0: []}, 'left': {0: []}, 'right': {0: []}, 'front': {0: []}, 'back': {0: []}, 'screw': {0: []}, 'sticker': {0: []}, 'sticker_aesthatic': {0: []},'missing_label':{0: ['label1','label2'],1: ['label1','label2']},}


# for i in mp_data:
#     print(i.get('kanban').get('features'))
#     print('***********')


def check_features(feature_list,pred_list):
    missing_features = []
    for i in feature_list:
        if not i in pred_list:
            missing_features.append(i)
    return missing_features

for i,j in pred.items():
    if i == 'missing_label':
        print(i,j) 
        missing_label = []
        
        for k,v in j.items():
            print(k,v)
            missing_label.extend(v)
        print(missing_label)

        missing_features = check_features(['label1','a'],missing_label)
        print(missing_features)





print('**********************************')
from common_utils import MongoHelper

mp = MongoHelper().getCollection('parts')
mp_data = mp.find_one({'part_number':'variant2'})
features = mp_data.get('kanban').get('features')
print(features)






