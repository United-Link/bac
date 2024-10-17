import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
import torch
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.augmentations import letterbox

from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
import cv2

def judge(y,yr,width):
    if (abs(y-yr)<width):
        return True
    return False
def check(l1,l2):
    if(l1[1]!=l2[1]):
        if (l1[0]>l2[0]):
            return int(l1[1])
    return int(l2[1])
def detect_poker(model, im0, conf_thres, model_type, x_center=None):
    #weights='/best.pt'  # model path or triton URL
    #data='/data/coco.yaml'  # dataset.yaml path
    imgsz=(640, 640)  # inference size (height, width)
    #conf_thres=0.55  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=1000  # maximum detections per image
    #device='cuda:0'  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    augment=False  # augmented inference
    agnostic_nms=False  # class-agnostic NMS
    classes = None
    visualize=False  # visualize features
    yellow = False #yellow card



    # Load model
    #model_type = 0 #0 for yolov9, 1 for gelan
    
    stride, names, pt = model.stride, model.names, model.pt
    im = letterbox(im0, imgsz, stride)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(model.device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None] 
    model.warmup(imgsz=(1, 3, *(640,640)))  # warmup
    pred_0 = model(im, augment=augment, visualize=visualize)
    #print(pred)
    if(model_type==0):
        pred_0 = pred_0[0]
    pred_0 = non_max_suppression(pred_0[0], conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    pred_0 = pred_0[0]
    #print(pred)

    pred_0[:, :4] = scale_boxes(im.shape[2:], pred_0[:, :4], im0.shape).round()
    pred_1, index_1 = torch.sort(pred_0[:,0],dim=0,stable=True)
    

    
    #print(len(pred))
    card_list = [255,255,255,255,255,255]
    #print(pred.tolist())
    pred = pred_0
    if (pred_0.tolist() != []):
        if (pred_0[index_1[0],5]==52):
            pred = torch.cat((pred_0[:index_1[0]], pred_0[index_1[0]+1:]), dim=0)
            #print(pred)
            yellow = True
            
    if (pred.tolist() == []):
        return card_list, yellow
    pred_num = len(pred)
    original_card = [255,255,255,255]
    add_card = [[],[]]
    classes = [[]]
    y_thres=0.15
    width_sum=0
    width_n=0
    for x,y,x2,y2 in pred[:,:4]:
        if(x2-x<y2-y):
            width_sum+=x2-x
            width_n +=1
    if width_n==0:
        return card_list, yellow
    avg_width = (width_sum/width_n)*2
    #print(f'width{avg_width}')
    if (x_center is None):
        x_center = im0.shape[1]/2
    for x in pred[:,0]:
        abs_x = torch.abs(x_center-x)
        #print(abs_x)
        if(abs_x <avg_width):
            if x_center-x <0:
                x_center =x -avg_width-1
            else:
                x_center =x +avg_width+1
    #find y line
    pred_2, index_2 = torch.sort(pred[:,1],dim=0,stable=True)
    #print(pred_2)
    #print(index_2)
    line_y = [pred_2[0]]
    j=0
    for i in range(pred_num-1):
        if((pred_2[i+1]-line_y[j])>avg_width*1.5):
            j=j+1
            line_y.append(pred_2[i+1])
        else:
            line_y[j] = (line_y[j]+pred_2[i+1])/2
    #print('line')
    #print(line_y)
    j=0
    for y, idx in zip(pred_2,index_2):
        #print(idx)
        #print(avg_width)
        if(judge(y,line_y[j],1.5*avg_width)):
            classes[j].append([y,idx])
        else:
            classes.append([])
            j=j+1
            classes[j].append([y,idx])
    # for i in range(len(classes)):
    #     print(len(classes[i]))
    #print('###')
    #print(classes)
    #print(len(classes[-1]))
    #print(len(classes[-2]))

    
    #print(f'x_center:{x_center}')
    #x_center = pred[]
    if ((len(classes)>2 and (len(classes[-1])>2 or len(classes[-2])>2))):
        #additional card
        for i in range(len(classes)-2):
            p1 = classes[i]
            #print(p1)
            for j in range(len(p1)):
                idx = p1[j][1].tolist()
                #print(idx)
                l_r =0
                if(pred[idx,0]>x_center):
                    l_r =1
                add_card[l_r].append(pred[idx,4:6].tolist())
    if len(classes)==2 and len(pred[:, 5].unique())>4:
        p1 = classes[0]
        #print(p1)
        for j in range(len(p1)):
            idx = p1[j][1].tolist()
            #print(idx)
            l_r =0
            if(pred[idx,0]>x_center):
                l_r =1
            add_card[l_r].append(pred[idx,4:6].tolist())

    #print(add_card)



    #original_card
    original_idxs1 = [[-1,255],[-1,255],[-1,255],[-1,255]]
    #
    #print(classes[-1])
    for e in classes[-1]:
        idx = e[1].tolist()
        dis = pred[idx,0]-x_center
        #print(dis)
        if(dis>2*avg_width):
            if(dis>5*avg_width):
                original_idxs1[3] = pred[idx,4:6].tolist()
            else:
                original_idxs1[2] = pred[idx,4:6].tolist()
        if(dis<-avg_width):
            if(dis<-4*avg_width):
                original_idxs1[0] = pred[idx,4:6].tolist()
            else:
                original_idxs1[1] = pred[idx,4:6].tolist()
    #print(original_idxs1)


    #for i in range(4):
        #print(original_idxs1[i][0])
    #print(len(classes))
    #print(classes)
    if len(classes)>1:
        process_num = [-x for x in range(2,len(classes)+1)]
        #print(process_num)
        add_or_not = False
        for n in process_num:
            if(add_or_not):
                break
            if not (original_idxs1[0] == [-1,255] or original_idxs1[1] == [-1,255] or original_idxs1[2] == [-1,255] or original_idxs1[3] == [-1,255]):
                break
            for e in classes[n]:
                
                #print(e)
                #print(e[0].tolist())
                
                idx = e[1].tolist()
                score = pred[idx,4]
                #print(f'score{score}')
                #print(add_card)
                if len(add_card[0])>0:
                    if score==add_card[0][0][0]:
                        break
                if len(add_card[1])>0:
                    if score==add_card[1][0][0]:
                        break
                #print(idx)
                dis = pred[idx,0]-x_center
                #print(dis)
                #print(original_idxs1)
                #print(pred[idx,4].tolist())
                if(dis>avg_width):
                    if (dis>4*avg_width):
                        if (original_idxs1[3][0]) >0:
                            add_or_not = True
                        if(pred[idx,4]>original_idxs1[3][0]):
                            original_idxs1[3] = pred[idx,4:6].tolist()
                    else:
                        if (original_idxs1[2][0]) >0:
                            add_or_not = True 
                        if(pred[idx,4]>original_idxs1[2][0]):
                            original_idxs1[2] = pred[idx,4:6].tolist()
                if(dis<-2*avg_width):
                    if(dis<-5*avg_width):
                        if (original_idxs1[0][0]) >0:
                            add_or_not = True
                        if(pred[idx,4]>original_idxs1[0][0]):
                            original_idxs1[0] = pred[idx,4:6].tolist()
                    else:
                        if (original_idxs1[1][0]) >0:
                            add_or_not = True
                        if(pred[idx,4]>original_idxs1[1][0]):
                            original_idxs1[1] = pred[idx,4:6].tolist()
                if(add_or_not):
                    if(original_idxs1[0][0]<0 or original_idxs1[1][0]<0):
                        add_or_not = False
                    if(original_idxs1[2][0]<0 or original_idxs1[3][0]<0):
                        add_or_not = False

    #print(original_idxs1)
    for i, idx in enumerate(original_idxs1):
        if idx != []:
            original_card[i] = int(idx[1])
    
    #print(original_card)
    #print(card_list)
    card_list[:2] = original_card[0:2]
    card_list[3:5] = original_card[2:4]
    #print(add_card)
    for i,labels in enumerate(add_card):
        if(labels == []):
            continue
        #print(f'labels:{labels}')
        if (len(labels)==1):
                card_list[2+i*3] = int(labels[0][1])
        else:
            card_list[2+i*3] = check(labels[0],labels[1])
    #print(f'card_list:{card_list}')
    return card_list, yellow






        
