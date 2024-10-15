import os
import time
import queue
import multiprocessing as mp
from enum import Enum
from datetime import datetime
from pathlib import Path
import shutil
import requests
import asyncio

import numpy as np
import cv2
import torch
from pydantic import BaseModel
from typing import Optional, Union
from loguru import logger
from ultralytics import YOLO

from utils.bac_setting import BacSetting, CardName
from utils.img_process import get_rank
from detect_poker import detect_poker
from models.common import DetectMultiBackend

class APIStatus(Enum):
    ok = "ok"
    earlydraw = 'EARLY DRAW'

class BacResult(BaseModel):
    Player1: Optional[int] = 255
    Player2: Optional[int] = 255
    Player3: Optional[int] = 255
    Banker1: Optional[int] = 255
    Banker2: Optional[int] = 255
    Banker3: Optional[int] = 255

class BacReturn(BaseModel):
    ShoeNo: Optional[str] = None
    RoundNo: Optional[str] = None
    TableNo: Optional[str] = None
    Status: Optional[str] = None
    Result: Optional[Union[BacResult, dict]] = BacResult().model_dump()
    Yellow: Optional[int] = 0

class YoloResult(BaseModel):
    label: Optional[int] = None
    conf: Optional[float] = None
    xyxy: Optional[list] = None
    name: Optional[str] = None
    cnt: Optional[list] = None

logger.catch()
def monitor_folder(target_path: Path, keep: int=1000):
    if not target_path.is_dir():
        logger.error(f'{target_path} not found')
    else:
        try:
            files = sorted(list(target_path.glob('*')), key=os.path.getctime, reverse=True)
            num_of_file = len(files)
            logger.info(f'number of folder in {target_path}: {num_of_file}')
        except Exception as e:
            num_of_file = 0
            logger.error(f'sort file error: {e}')

        if num_of_file > keep:
            folders_to_del = files[keep:]
            for folder_to_del in folders_to_del:
                try:
                    shutil.rmtree(folder_to_del)
                    logger.info(f'remove folder: {folder_to_del}')
                except Exception as e:
                    logger.info(f'remove folder exception: {e}')

logger.catch()
async def move_file(save_root: str, TableNo: str, ShoeNo: str, RoundNo: str, Status, keep: int=1000, tg_bot = None):
    tg_bot.send_message(text=f'Wrong\nTable: {TableNo}, Shoe: {ShoeNo}, Round: {RoundNo}, Status: {Status}')

    await asyncio.sleep(5)
    
    save_root = Path(save_root)
    source_path = save_root / f'{TableNo:>04s}' / f'{ShoeNo:>04s}_{RoundNo:>04s}'
    target_path = save_root / 'wrong' / f'{TableNo:>04s}'

    if not source_path.is_dir():
        logger.error(f'source pat: {source_path} not exist.')
    else:
        target_path.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), target_path)
        logger.info(f'move folder from {source_path} to {target_path}')
        monitor_folder(target_path, keep)

logger.catch()
def get_max_contour(crop_img, lower, upper):
    hsv_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    mask =  cv2.inRange(hsv_img, lower, upper)
    cnt, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnt) != 0:
        max_contour = max(cnt, key=cv2.contourArea)
    else:
        max_contour = None
    return max_contour

@logger.catch()
def save_result(result_dict: dict, current_info: BacReturn, save_root: str):
    # save image and pred result
    decoded_image = result_dict['img']
    
    save_root = Path(save_root)
    ori_img_file = save_root / f"{current_info.TableNo:>04s}" / f'{current_info.ShoeNo:>04s}_{current_info.RoundNo:>04s}' / 'ori'
    pred_img_file = save_root / f"{current_info.TableNo:>04s}" / f'{current_info.ShoeNo:>04s}_{current_info.RoundNo:>04s}' / 'pred'
    txt_file = save_root / f"{current_info.TableNo:>04s}" / f'{current_info.ShoeNo:>04s}_{current_info.RoundNo:>04s}' / 'txt'
    
    ori_img_file.mkdir(parents=True, exist_ok=True)
    pred_img_file.mkdir(parents=True, exist_ok=True)
    txt_file.mkdir(parents=True, exist_ok=True)

    num_of_img = len(list(ori_img_file.glob('*.jpg')))
    
    ori_img_path = ori_img_file / f'{current_info.ShoeNo:>04s}_{current_info.RoundNo:>04s}_{num_of_img:04d}.jpg'
    pred_img_path = pred_img_file / f'{current_info.ShoeNo:>04s}_{current_info.RoundNo:>04s}_{num_of_img:04d}.jpg'
    txt_path = txt_file / f'{current_info.ShoeNo:>04s}_{current_info.RoundNo:>04s}_{num_of_img:04d}.txt'

    logger.info(f'save result at:{ori_img_path}')
    
    cv2.imwrite(str(ori_img_path), decoded_image)

    cnt_list = []

    with open(txt_path, 'w') as f:
        for card in result_dict['result']:
            x1, y1, x2, y2 = card.xyxy
            cnt_list.append(card.cnt)
            cv2.rectangle(decoded_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f'{card.name}, {card.conf:.4f}'
            cv2.putText(decoded_image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            line = f'{card.label} {x1} {y1} {x2} {y2}\n'
            f.write(line)
    
    cv2.drawContours(decoded_image, cnt_list, -1, (0, 0, 255), 2)
    cv2.imwrite(str(pred_img_path), decoded_image)

@logger.catch()
async def get_frame_from_queue(global_dict, request):
    while True:
        
        if await request.is_disconnected():
            logger.info('request is disconnected.')
            break

        result_dict = global_dict['result_queue'].get()
        frame = result_dict['img']

        
        for box in global_dict['boxes']:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        _, decoded_image = cv2.imencode('.jpg', frame)
        
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + decoded_image.tobytes() + b'\r\n')
        
    logger.info('calibration end.')

@logger.catch()
async def get_card_cnt(global_dict, request):
    p1_loc = global_dict['setting'].calibration_p1

    while True:
        if await request.is_disconnected():
            logger.info('calibration card request is disconnected.')
            break
        
        result_dict = global_dict['result_queue'].get()
        decoded_image = result_dict['img']
        crop_img=decoded_image[p1_loc[1]:p1_loc[3], p1_loc[0]:p1_loc[2], :]
        max_contour = get_max_contour(crop_img, global_dict['setting'].green_lower, global_dict['setting'].green_upper)

        if max_contour is not None:
            approx = cv2.convexHull(max_contour)
            cv2.drawContours(crop_img, [approx], -1, (0, 0, 255), 3)
            card_area = cv2.contourArea(approx)
        else:
            card_area = float('inf')

        
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + decoded_image.tobytes() + b'\r\n')
    
    global_dict['card_area'] = card_area
    logger.info(f'calibration card end. and set card area: {card_area}')

@logger.catch()
def get_newest_weight(folder_path: Path):
    if folder_path.is_dir():
        all_weights = [w for w in folder_path.glob('*.pt')]

        if not all_weights:
            logger.error(f"Can't found any weight.")
            return None

        newest_weight = max(all_weights, key=os.path.getmtime)
        logger.info(f'get the newest weight: {newest_weight}')

        return newest_weight
    
    else:
        logger.error('folder does not exist.')
        return None

@logger.catch()
def pred_img_from_queue(img_queue: mp.Queue, result_queue: mp.Queue, close_event: mp.Event, model_folder: str, conf=0.5, max_det=20):
    """
    img_queue: encoded image
    result_queue: put result in this queue
    model: yolov9-seg model
    """
    
    weight_path = './best.pt'
    data = './data/coco.yaml'
    #assert weight_path, "YOLO weight not Found"
    device=0  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    model_type = 0 #0 for yolov9, 1 for gelan
    half = False #float16/float32
    dnn=False # use OpenCV DNN for ONNX inference
    conf_res = 0.5
    logger.info('init YOLO model.')
    device = torch.device('cuda:0')
    model = DetectMultiBackend(weight_path, device=device, dnn=dnn, data=data, fp16=half)
    logger.info(f'init YOLO model{model_type}.')
    
    #model = YOLO(weight_path)

    while not close_event.is_set():
        start_time = time.time()
        decoded_image = img_queue.get()
        result, yellow = detect_poker(model, decoded_image, conf_res, model_type, None)
        #logger.info(f'inference time:{time.time()-start_time}')
        
        
        #results = model.predict(decoded_image, conf=conf, max_det=max_det, retina_masks=True, verbose=False)[0]
        #results = {'names':[],'boxes':[],'masks':[]}
        #names = results.names

        # parser bbox and mask
        result_list = []
        result_dict = {}

        #for i, bbox in enumerate(results.boxes):
            #temp_result = YoloResult()

            #temp_result.label = int(bbox.cls.item())
            #temp_result.conf = bbox.conf.item()
            #temp_result.xyxy = bbox.xyxy.cpu().numpy().astype(np.int32).tolist()[0]
            #temp_result.name = names[temp_result.label]
            #temp_result.cnt = results.masks[i].xy[0].astype(np.int32)

            #result_list.append(temp_result)

        result_dict['img'] = decoded_image
        for x in result:
            result_list.append(x)
        result_dict['result'] = result_list
        result_dict['yellow'] = yellow
        

        # if (result_dict['result'] != []):
        #     logger.info(f'init YOLO model{result}.')

        try:
            if result_queue.full():
                result_queue.get_nowait()
            result_queue.put_nowait(result_dict)
        except queue.Empty:
            pass
        except queue.Full:
            pass
        except Exception as e:
            logger.error(f'other error in pred: {e}')

    logger.info('close pred process.')
        


@logger.catch()
def pred_img_from_queue2(img_queue: mp.Queue, result_queue: mp.Queue, close_event: mp.Event, model_folder: str, conf=0.5, max_det=20):
    """
    img_queue: encoded image
    result_queue: put result in this queue
    model: yolov9-seg model
    """
    
    weight_path = './best_striped.pt'
    data = './data/coco.yaml'
    #assert weight_path, "YOLO weight not Found"
    device=0  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    model_type = 1 #0 for yolov9, 1 for gelan
    half = False #float16/float32
    dnn=False # use OpenCV DNN for ONNX inference
    conf_res = 0.5
    logger.info('init YOLO model.')
    device = torch.device('cuda:0')
    model = DetectMultiBackend(weight_path, device=device, dnn=dnn, data=data, fp16=half)
    logger.info(f'init YOLO model{model_type}.')
    
    #model = YOLO(weight_path)

    while not close_event.is_set():

        decoded_image = img_queue.get()
        start_time = time.time()
        result, yellow = detect_poker(model, decoded_image, conf_res, model_type, None)
        result_list = []
        result_dict = {}

        result_dict['img'] = decoded_image
        for x in result:
            result_list.append(x)
        result_dict['result'] = result_list
        result_dict['yellow'] = yellow
        
        # if (result_dict['result'] != []):
        #     logger.info(f'init YOLO model{result}.')
        #logger.info(f"img_queue size before get: {img_queue.qsize()}")
        #logger.info(f"result_queue size before put: {result_queue.qsize()}")
        try:
            if result_queue.full():
                result_queue.get_nowait()
            result_queue.put_nowait(result_dict)
        except queue.Empty:
            pass
        except queue.Full:
            pass
        except Exception as e:
            logger.error(f'other error in pred: {e}')

    logger.info('close pred process.')

@logger.catch()
def monitor_early_draw(result_queue: mp.Queue, current_info: BacReturn, bettime: int, setting: BacSetting, monitor_flag: list):

    start_time = time.time()
    start_time_fmt = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S.%f')
    end_time = time.time() + (bettime - 1)
    end_time_fmt = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S.%f')

    logger.info(f'start to monitor Early Draw. start time: {start_time_fmt}, end time: {end_time_fmt}')

    # set Yellow Card init.
    current_info.Yellow = 0
    current_info.Status = APIStatus.ok.value

    # get pred result from result queue.
    while time.time() < end_time and monitor_flag[0]:

        logger.debug('monitor Early Draw.')
        result_dict = result_queue.get()
        
        for card in result_dict['result']:
                
            if card != 255:
                current_info.Status = APIStatus.earlydraw.value
                logger.info(f'Get Early Draw')
                break
        
        if current_info.Status == APIStatus.earlydraw.value:
            break

        time.sleep(0.5)

    logger.info('Monitor Early Draw Finish.')
    monitor_flag[0] = False

    if current_info.Status == APIStatus.earlydraw.value:
        save_result(result_dict, current_info, setting.image_folder)

logger.catch()
def find_points_inside_boxes(points: np.ndarray, boxes: np.ndarray):
    """
    points: (n, 2)
    boxes: (m, 4)
    """
    
    if len(points) == 0:
        return points
    else:
        x_inside = np.logical_and(points[:, 0][:, np.newaxis] >= boxes[:, 0], points[:, 0][:, np.newaxis] <= boxes[:, 2])
        y_inside = np.logical_and(points[:, 1][:, np.newaxis] >= boxes[:, 1], points[:, 1][:, np.newaxis] <= boxes[:, 3])
    
        result = np.logical_and(x_inside, y_inside)
    
        point_to_box_indices = np.argmax(result, axis=1)
        point_to_box_indices[~np.any(result, axis=1)] = -1
    
        return point_to_box_indices

@logger.catch()
def check_overlap(mask_img: np.ndarray, cnt: list):
    
    cnt_temp = np.zeros_like(mask_img)
    cv2.drawContours(cnt_temp, [cnt], -1, 255, -1)
    
    cnt_area = np.sum(cnt_temp.astype(np.bool_))
    merge_area = np.sum(cv2.bitwise_and(mask_img, cnt_temp).astype(np.bool_))

    overlap_ratio = merge_area / cnt_area

    return overlap_ratio

@logger.catch()
def process_result(result_dict: dict, current_info: BacReturn, same: bool):

    temp_result = BacResult().model_dump()
        
    if (same):
        for i, label in enumerate(result_dict['result']):
            match i:
                case 0:
                    temp_result[CardName.Player1.name] = label
                case 1:
                    temp_result[CardName.Player2.name] = label
                case 2:
                    temp_result[CardName.Player3.name] = label
                case 3:
                    temp_result[CardName.Banker1.name] = label
                case 4:
                    temp_result[CardName.Banker2.name] = label
                case 5:
                    temp_result[CardName.Banker3.name] = label
    
    current_info.Result = temp_result

logger.catch()
def exam_result(current_info: BacReturn):

    # check p1/p2/b1/b2 is None
    if current_info.Result[CardName.Banker1.name] == 255 or current_info.Result[CardName.Player1.name] == 255:
        current_info.Result = BacResult().model_dump()

    if current_info.Result[CardName.Banker2.name] == 255 or current_info.Result[CardName.Player2.name] == 255:
        current_info.Result[CardName.Banker2.name] = 255
        current_info.Result[CardName.Player2.name] = 255
        current_info.Result[CardName.Banker3.name] = 255
        current_info.Result[CardName.Player3.name] = 255
        
    logger.info(f'check p1/p2/b1/b2 is 255.')
    logger.info(f'current result: {current_info.Result}')

logger.catch()
def detect_qrcode_and_post(img_queue: mp.Queue, url: str):
    '''
    get frame from image queue and detect QR code and send it to dealer PC
    '''

    det = cv2.QRCodeDetector()
    last_dealer_id = None
    frame_count = 0

    while True:
        start_time = time.time()
    
        decoded_image  = img_queue.get()
        h, w, _ = decoded_image.shape

        for start_h in range(0, h // 2, 100):
            for start_w in range(0, w // 2, 100):
                try:
                    crop_img = decoded_image[start_h: start_h + 300, start_w: start_w + 300, :]
                    info, _, _ = det.detectAndDecode(crop_img)
                except:
                    continue
                
                if info:
                    if last_dealer_id != info:
                        logger.info(f'get QR code: {info} and send')
                        try:
                            requests.post(url, data=info, timeout=5)
                            last_dealer_id = info
                            logger.info(f'set current dealer id: {info}')
                            break
                        except  Exception as e:
                            logger.error(f'post error: {e}')
            if info:
                break
        
        if not info:
            frame_count += 1
        
        end_time = time.time()
        time.sleep(max(0, 1 - (end_time - start_time)))
        if frame_count > 5:
            frame_count = 0
            last_dealer_id = None

