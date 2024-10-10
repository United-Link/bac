import os
import signal
import multiprocessing as mp
import threading
from pathlib import Path
import time
import queue
from contextlib import asynccontextmanager

import numpy as np
import cv2
from loguru import logger
from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
# for Python 3.10+
from typing import Annotated
from collections import Counter

from utils.bac_setting import BacSetting
from utils.telegram_bot import TgBot
from utils.camera_utils import open_cam
from utils.api_utils import monitor_early_draw, process_result, exam_result, save_result, move_file
from utils.api_utils import get_frame_from_queue, pred_img_from_queue, pred_img_from_queue2, monitor_folder, detect_qrcode_and_post, get_card_cnt
from utils.api_utils import APIStatus, BacReturn, BacResult

# Ensure the spawn method is set before FastAPI or any other processes are started
def set_multiprocessing_start_method():
    try:
        mp.set_start_method('spawn', force=True)  # 'force=True' ensures it's set even if already initialized.
    except RuntimeError:
        # The start method has already been set. This is expected if it was initialized earlier.
        pass

def empty_queue(queue):
    result = []
    print(f'Initial queue size: {queue.qsize()}')
    for i in range(10):
        try:
            item = queue.get(timeout=1.0) 
            result.append(item)
            print(f'Retrieved item: {item}. Remaining size: {queue.qsize()}')
        except queue.Empty:  
            print("Queue was empty or timed out, stopping.")
            break
    print(f'Result size: {len(result)}')
    return result




# Call the function early to ensure it's set properly
set_multiprocessing_start_method()
target = os.getenv('TARGET', 'bac')

logger.add(
    sink='./logs/detection.{time:YYYY.MM.DD}.log',
    level="INFO",
    rotation="00:00",
    retention="1 months",
    backtrace=True,
    enqueue=True)

# global value dict
global_dict = {}
current_info = BacReturn()
status_lock = threading.Lock()
early_monitor_flag = [False]
error_count = {
    'total': 0,
    'EarlyDraw': 0,
    'Timeout': 0,
    'Finial Match': 0,
    'Recognition Match': 0,
    'Others': 0,
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # init bac setting
    global_dict['setting'] = BacSetting()
    logger.info(f"init bac setting: {global_dict['setting'].model_dump()}")
    
    global_dict['boxes'] = np.array([
        global_dict['setting'].calibration_p1,
        global_dict['setting'].calibration_p2,
        global_dict['setting'].calibration_p3,
        global_dict['setting'].calibration_b1,
        global_dict['setting'].calibration_b2,
        global_dict['setting'].calibration_b3,],
        dtype=np.uint32)
    
    # init others var
    global_dict['img_queue'] = mp.Queue(maxsize=10)
    global_dict['result_queue'] = mp.Queue(maxsize=10)
    global_dict['result_queue2'] = mp.Queue(maxsize=10)
    global_dict['close_event'] = mp.Event()
    global_dict['init_cam_event'] = mp.Event()


    global_dict['tg_bot'] = TgBot(global_dict['setting'].tg_token, global_dict['setting'].tg_chat)
    logger.info('init TG bot')
    global_dict['tg_bot'].send_message(text=f"{global_dict['setting'].server_name} start.")

    logger.info('init streaming.')
    logger.info('start to open video source.')
    cap_prop = (cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000)

    global_dict['cam_process'] = mp.Process(
        target=open_cam,
        daemon=True,
        args=(
            global_dict['setting'].streaming,
            global_dict['img_queue'],
            global_dict['close_event'],
            None,
            cap_prop,
            global_dict['tg_bot']
        )
    )

    global_dict['cam_process'].start()
    logger.info('wait camera ready.')
    global_dict['img_queue'].get()
    logger.info('Warm Up Finish.')

    global_dict['pred_process'] = mp.Process(
        target=pred_img_from_queue,
        daemon=False,
        args=(
            global_dict['img_queue'],
            global_dict['result_queue'],
            global_dict['close_event'],
            global_dict['setting'].weight_folder,
            global_dict['setting'].model_threshold,
            global_dict['setting'].max_det
        )
    )
    global_dict['pred_process2'] = mp.Process(
        target=pred_img_from_queue2,
        daemon=False,
        args=(
            global_dict['img_queue'],
            global_dict['result_queue2'],
            global_dict['close_event'],
            global_dict['setting'].weight_folder,
            global_dict['setting'].model_threshold,
            global_dict['setting'].max_det
        )
    )

    global_dict['pred_process'].start()
    global_dict['pred_process2'].start()
    logger.info('wait pred process ready.')
    
    global_dict['result_queue'].get()
    global_dict['result_queue2'].get()
    logger.info('pred process start.')
    #temp =global_dict['result_queue']
    #logger.info(f'result:{temp}')
    # find card area
    # encoded_image = global_dict['img_queue'].get()
    # buffer = np.frombuffer(encoded_image, dtype=np.uint8)
    # decoded_image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    # h, w, _ = decoded_image.shape
    # crop_img = decoded_image[global_dict['setting'].calibration_p1[1]:global_dict['setting'].calibration_p1[3], global_dict['setting'].calibration_p1[0]:global_dict['setting'].calibration_p1[2], :]
    # hsv_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    # mask =  cv2.inRange(hsv_img, global_dict['setting'].green_lower, global_dict['setting'].green_upper)
    # cnt, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # if len(cnt) == 0:
    #     global_dict['tg_bot'].send_message(text=f"{global_dict['setting'].server_name} check card calibration.")
    #     max_contour_area = float('inf')
    # else:
    #     max_contour = max(cnt, key=cv2.contourArea)
    #     max_contour_area = cv2.contourArea(max_contour)    
    # global_dict['card_area'] = max_contour_area
    # global_dict['img_h'] = h
    # global_dict['img_w'] = w
    # logger.info(f'original image size: H:{h}, w:{w}')
    #logger.info(f'get card area: {max_contour_area}')

    logger.info('start to Detect QRcode process')
    dealer_pc_url = f"http://{global_dict['setting'].dealer_pc}:{global_dict['setting'].dealer_pc_prot}/{target}_readcard"
    
    global_dict['qrcode_process'] = mp.Process(
        target=detect_qrcode_and_post,
        daemon=True,
        args=(
            global_dict['img_queue'],
            dealer_pc_url
        )
    )
    global_dict['qrcode_process'].start()

    yield
    # end API service
    global_dict['close_event'].set()
    logger.info('close read camera process.')
    logger.info('wait cam process close')
    global_dict['cam_process'].kill()
    logger.info('close process ok.')
    global_dict['pred_process'].kill()
    global_dict['pred_process2'].kill()
    global_dict['qrcode_process'].kill()
    logger.info('End API service.')

app = FastAPI(lifespan=lifespan)

@app.get('/calibration')
async def calibration(request: Request):
    return StreamingResponse(get_frame_from_queue(global_dict, request), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get('/calibration_card')
async def calibration_card(request: Request):
    return StreamingResponse(get_card_cnt(global_dict, request), media_type='multipart/x-mixed-replace; boundary=frame')

@app.post(f'/{target}_status')
async def game_status(
    ShoeNo: Annotated[str, Form()],
    RoundNo: Annotated[str, Form()],
    TableNo: Annotated[str, Form()],
    BetTime: Annotated[int, Form()],
    background_task: BackgroundTasks,
):
    global current_info
    global status_lock
    global early_monitor_flag

    if early_monitor_flag[0]:
        logger.info('monitor early draw process is running.')

    with status_lock:
        if not early_monitor_flag[0]:
            early_monitor_flag[0] = True
            current_info = BacReturn()
            current_info.ShoeNo = ShoeNo
            current_info.RoundNo = RoundNo
            current_info.TableNo = TableNo
            background_task.add_task(monitor_early_draw, global_dict['result_queue'], current_info, BetTime, global_dict['setting'], early_monitor_flag)

    return APIStatus.ok.value

@app.get(f'/{target}_status_end')
async def game_status_end():
    global early_monitor_flag

    early_monitor_flag[0] = False

    return APIStatus.ok.value

@app.post(f'/{target}_streaming', response_class=JSONResponse)
async def detection(
    ShoeNo: Annotated[str, Form()],
    RoundNo: Annotated[str, Form()],
    TableNo: Annotated[str, Form()],
    background_task: BackgroundTasks,
):
    global current_info

    current_info.ShoeNo = ShoeNo
    current_info.RoundNo = RoundNo
    current_info.TableNo = TableNo
    
    logger.info(f'get streaming API, Table: {TableNo}, Shoe: {ShoeNo}, Round: {RoundNo}.')
    
    # get frame from queue
    try:
        result_dict = empty_queue(global_dict['result_queue'])
        result_dict2 = empty_queue(global_dict['result_queue2'])
        print(f'size:{len(result_dict)}')
        if (result_dict==[] or result_dict2==[]):
            return current_info
        temp1 = result_dict[0]['result']
        temp2 = result_dict2[0]['result']
        logger.info(f'model1:{temp1}')
        logger.info(f'model2:{temp2}')
        same = True
        if temp1==temp2:
            logger.info(f'same!!!')
        else:
            logger.info(f'not the same!!!')
            same = False
        #print(result_dict)
    except queue.Empty:
        logger.error('get result timeout.')
        return current_info
    if (result_dict[0]['yellow'] or result_dict2[0]['yellow']):
        current_info.Yellow = 1
    else:
        current_info.Yellow = 0
    # save information
    #background_task.add_task(save_result, result_dict, current_info, global_dict['setting'].image_folder)
    
    process_result(result_dict[0], current_info, same)
    #exam_result(current_info)

    logger.info(f'return result: {current_info}')

    return current_info

@app.post(f'/{target}_wrong')
async def record_wrong(
    ShoeNo: Annotated[str, Form()],
    RoundNo: Annotated[str, Form()],
    TableNo: Annotated[str, Form()],
    Status: Annotated[str, Form()],
    background_task: BackgroundTasks
):
    Status = Status.lower()
    match Status:
        case value if "finial" in value:
            error_count['Finial Match'] += 1
        case value if "timeout" in value:
            error_count['Timeout'] += 1
        case value if "recognition" in value:
            error_count['Recognition Match'] += 1
        case value if "early" in value:
            error_count['EarlyDraw'] += 1
        case _:
            error_count['Others'] += 1

    try:
        background_task.add_task(
            move_file,
            global_dict['setting'].image_folder,
            TableNo,
            ShoeNo,
            RoundNo,
            Status,
            global_dict['setting'].folder_keep,
            global_dict['tg_bot'])
    except Exception as e:
        pass
    return 'ok'

@app.post(f'/{target}_end')
async def end(
    ShoeNo: Annotated[str, Form()],
    RoundNo: Annotated[str, Form()],
    TableNo: Annotated[str, Form()],
    background_task: BackgroundTasks,
):
    
    images_folder = Path(global_dict['setting'].image_folder)
    images_folder = images_folder / f'{TableNo:>04s}'

    background_task.add_task(monitor_folder, images_folder, global_dict['setting'].folder_keep)
    error_count['total'] += 1

    return APIStatus.ok.value

@app.get(f'/{target}_info')
async def get_info():
    logger.debug(global_dict['card_area'])
    return error_count

@app.get('/restart')
async def shutdown():
    os.kill(os.getpid(), signal.SIGTERM)
    return 'ok'
