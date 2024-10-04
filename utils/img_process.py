import numpy as np
import cv2
from pathlib import Path
from loguru import logger

logger.catch()
def order_points(pts: np.ndarray, axis: int=2):
    """
    return sort bounding box point(left top, right top, right down, left down)
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    s = np.sum(pts, axis=axis)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=axis)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

logger.catch()
def get_card_mask(img: np.ndarray, color_lower: np.ndarray, color_upper: np.ndarray):
    """
    img: crop card image.
    """
    img = cv2.GaussianBlur(img, (5, 5), 0)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, color_lower, color_upper)

    hsv_img[:,:,0] = cv2.equalizeHist(hsv_img[:,:,0])
    
    return mask

logger.catch()
def get_trans_card(img: np.ndarray, mask: np.ndarray, card_width: int, card_height: int):
    
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return None
    
    # get max contour
    max_card_cnt = sorted(cnts, key=lambda i: cv2.contourArea(i), reverse=True)[0]

    # fit min Rectangle
    rect = cv2.minAreaRect(max_card_cnt)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    
    approx_cnt = order_points(box, axis=1)
    _, _, w, h = cv2.boundingRect(approx_cnt)
    if w > h:
        approx_cnt = np.concatenate([approx_cnt[1:], approx_cnt[0:1]], axis=0)

    dst = np.array([[0, 0], [card_width - 1, 0], [card_width - 1, card_height - 1], [0, card_height - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(approx_cnt, dst)
    trans_card = cv2.warpPerspective(img, M, (card_width, card_height))

    return trans_card

@logger.catch()
def get_fit_rectangle_and_rescale(img: np.ndarray, target_w: int, target_h: int):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(cnts) == 0:
        return None
    
    cnt = sorted(cnts, key=lambda i: cv2.contourArea(i), reverse=True)[0]
    x, y, w, h = cv2.boundingRect(cnt)
    
    mask = mask[y:y+h, x:x+w]
    mask = cv2.resize(mask, (target_w, target_h), cv2.INTER_NEAREST)
    
    return mask

logger.catch()
def get_img_simility(mask_img, standard_img):

    bitwise_xor = cv2.bitwise_xor(mask_img, standard_img)
    bitwixe_not_xor = cv2.bitwise_not(bitwise_xor)
    h, w = bitwixe_not_xor.shape
    simility = (np.sum(bitwixe_not_xor / 255) / (h * w))

    return simility

@logger.catch()
def get_max_simility_result(mapping_dict, standard_img_dict, img):

    max_simility = 0
    max_idx = -1

    for i in range(13):
        simility = get_img_simility(img, standard_img_dict[mapping_dict[i]])
        if max_simility < simility:
            max_simility = simility
            max_idx = i
    
    return max_idx

logger.catch()
def get_rank(
        img: np.ndarray, 
        box: list, 
        color_lower: np.ndarray, 
        color_upper: np.ndarray, 
        card_width: int, 
        card_height: int,
        rank_size: list,
        rank_target: list,
        mapping_dict: dict,
        standard_img_dict: dict):
    """
    img: original image.
    box: bounding box [x1, y1, x2, y2] from yolo pred.
    color_lower, color_upper: create mask color range.
    card_width, card_height: transform card w/h.
    rank_size: crop rank image, [rank w, rank h].
    rank_target: rank target size [w, h].
    """
    h, w, _ = img.shape
    x1, y1, x2, y2 = box
    new_x1 = max(0, x1-5)
    new_y1 = max(0, y1-5)
    new_x2 = min(w-1, x2+5)
    new_y2 = min(h-1, y2+5)

    try:
        card = img[new_y1:new_y2, new_x1:new_x2, :]
        card_mask = get_card_mask(card, color_lower, color_upper)
        trans_card = get_trans_card(card, card_mask, card_width, card_height)

        if trans_card is None:
            logger.info("can't find Card mask")
            return None
    
        rank_img = trans_card[:rank_size[1], :rank_size[0]]
        resize_rank = get_fit_rectangle_and_rescale(rank_img, rank_target[0], rank_target[1])

        # check rank/suit not None
        if resize_rank is None:
            logger.info('rank is None')
            return None
        
        rank_idx = get_max_simility_result(mapping_dict, standard_img_dict, resize_rank)

        return rank_idx
    
    except Exception as e:
        logger.error(f'{e}')
        return None