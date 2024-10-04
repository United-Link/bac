import time
import threading
import queue
import multiprocessing as mp

import cv2
from loguru import logger

logger.catch()
def get_cam(source, cap_api, cap_prop):
    if cap_api:
        cap = cv2.VideoCapture(source, cap_api, cap_prop)
    else:
        cap = cv2.VideoCapture(source)
    return cap

@logger.catch()
def open_cam(source, img_queue: mp.Queue, close_event: mp.Event, cap_api: int=None, cap_prop: tuple=None, telegram_bot=None):
    logger.info('start to connect to camera.')
    cap = get_cam(source, cap_api, cap_prop)
    logger.info('open cam finish')

    retry = 0

    while not close_event.is_set():
        try:
            ret, frame = cap.read()
        except:
            logger.error(f'Opencv VideoCapture read API error.')
            break
        if not ret:
            retry += 1
            logger.error(f"can't opencv camera from {source}. Wait 3 second to reconnected. retry: {retry}")
            if retry % 5 == 0 and telegram_bot:
                telegram_bot.send_message(text=f"can't opencv camera from {source}. Wait 3 second to reconnected. retry: {retry}")
            cap.release()
            time.sleep(3)
            logger.info('Start reconnecting the camera.')
            cap = get_cam(source, cap_api, cap_prop)
            continue
            
        retry = 0
        # encoded image and put in queue

        try:
            if img_queue.full():
                img_queue.get_nowait()
            img_queue.put_nowait(frame)
            if img_queue.full():
                img_queue.get_nowait()
            img_queue.put_nowait(frame)
        except queue.Empty:
            pass
        except queue.Full:
            continue
        except Exception as e:
            logger.error(f'others error: {e}')
            break

    cap.release()
    logger.info('get frame process closed.')


@logger.catch()
def open_cam_backup(source, img_queue: mp.Queue, close_event: mp.Event, cap_api: int=None, cap_prop: tuple=None, telegram_bot=None):
    logger.info('start to connect to camera.')
    cap = get_cam(source, cap_api, cap_prop)
    logger.info('open cam finish')

    retry = 0

    while not close_event.is_set():
        try:
            ret, frame = cap.read()
        except:
            logger.error(f'Opencv VideoCapture read API error.')
            break
        if not ret:
            retry += 1
            logger.error(f"can't opencv camera from {source}. Wait 3 second to reconnected. retry: {retry}")
            if retry % 5 == 0 and telegram_bot:
                telegram_bot.send_message(text=f"can't opencv camera from {source}. Wait 3 second to reconnected. retry: {retry}")
            cap.release()
            time.sleep(3)
            logger.info('Start reconnecting the camera.')
            cap = get_cam(source, cap_api, cap_prop)
            continue
            
        retry = 0
        # encoded image and put in queue
        _, encoded_image = cv2.imencode('.jpg', frame)
        encoded_image = encoded_image.tobytes()

        try:
            if img_queue.full():
                img_queue.get_nowait()
            img_queue.put_nowait(encoded_image)
        except queue.Empty:
            pass
        except queue.Full:
            continue
        except Exception as e:
            logger.error(f'others error: {e}')
            break

    cap.release()
    logger.info('get frame process closed.')