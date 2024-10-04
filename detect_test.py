from detect_poker import detect_poker 
import cv2
import torch
from models.common import DetectMultiBackend
import numpy as np
import time



def judge(y,yr,y_thres):
    if (abs(y-yr)<yr*y_thres):
        return True
    return False
def check(l1,l2):
    if(l1[1]!=l2[1]):
        if (l1[0]>l2[0]):
            return int(l1[1])
    return int(l2[1])

weights='/best_striped.pt'  # model path or triton URL
data = './data/coco.yaml'
path = './testyellow.jpg'
conf_thres=0.5  # confidence threshold
device=1  # cuda device, i.e. 0 or 0,1,2,3 or cpu
model_type = 1 #0 for yolov9, 1 for gelan
im0 = cv2.imread(path)
#im0 = cv2.resize(im0,(640,480))
print(im0.shape)
x_center = im0.shape[1]/2
half = False #float16/float32
dnn=False # use OpenCV DNN for ONNX inference
# Load model


#print(im0.shape)
device = torch.device('cuda:0')
model = DetectMultiBackend('./best.pt', device=device, dnn=dnn, data=data, fp16=half)
start_time = time.time()
result,_ = detect_poker(model, im0, conf_thres, model_type, x_center)
names=  np.array(['AS', '2S', '3S', '4S', '5S', '6S', '7S', '8S', '9S', '10S', 'JS', 'QS', 'KS', 'AH', '2H', '3H', '4H', '5H', '6H', '7H', '8H', '9H', '10H', 'JH', 'QH', 'KH', 'AD', '2D', '3D', '4D', '5D', '6D', '7D', '8D', '9D', '10D', 'JD', 'QD', 'KD', 'AC', '2C', '3C', '4C', '5C', '6C', '7C', '8C', '9C', '10C', 'JC', 'QC', 'KC', 'XX'])

print(result)
result2 = []
for idx in result:
    if idx==255:
        x = None
    else:
        x = names[idx]
    result2.append(x)
print(f'Player:{result2[:3]}')
print(f'Banker:{result2[3:]}')
end_time = time.time()
execution_time = end_time - start_time
print(f"處理時間: {execution_time} 秒")
