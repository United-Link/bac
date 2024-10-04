from pathlib import Path
import numpy as np
import cv2
from pydantic_settings import BaseSettings
from pydantic import field_validator, model_validator 
from typing import List, Union
from enum import Enum, unique

mapping_name = {
    0: 'A',
    1: '2',
    2: '3',
    3: '4',
    4: '5',
    5: '6',
    6: '7',
    7: '8',
    8: '9',
    9: '10',
    10: 'J',
    11: 'Q',
    12: 'K',
}

def get_standard_img(img_path:Path, mapping_dict: dict):
    img_dict = {}
    for key, value in mapping_dict.items():
        img_p = img_path / f'{value}.png'
        img_dict[value] = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
    return img_dict

class BacSetting(BaseSettings):
    server_name: str
    dealer_pc: str
    dealer_pc_prot: str
    log_folder: str
    image_folder: str
    tg_token: str
    tg_chat: int
    streaming: str
    weight_folder: str
    model_threshold: float
    area_threshold: float
    overlap_threshold: float
    max_det: int
    folder_keep: int
    green_lower: Union[List, np.ndarray]
    green_upper: Union[List, np.ndarray]
    calibration_p1: List[int]
    calibration_p2: List[int]
    calibration_p3: List[int]
    calibration_b1: List[int]
    calibration_b2: List[int]
    calibration_b3: List[int]
    white_lower: Union[List, np.ndarray]
    white_upper: Union[List, np.ndarray]
    card_width: int
    card_height: int
    rank_width: int
    rank_height: int
    rank_width_target: int
    rank_height_target: int
    rank_img_path: str
    mapping_dict: dict = mapping_name
    standard_img_dict: dict = {}
    
    @field_validator('green_lower', 'green_upper', 'white_lower', 'white_upper')
    def lsit_to_array(cls, v):
        return np.array(v)
    
    @field_validator('standard_img_dict')
    def read_standard_img(cls, v, info):
        img_path = Path(info.data['rank_img_path'])
        return get_standard_img(img_path, info.data['mapping_dict'])

    class Config:
        protected_namespaces = ('settings_',)
        env_file = './bac_config/bac_config.ini'

@unique
class CardName(Enum):
    Player1: int = 0
    Player2: int = 1
    Player3: int = 2
    Banker1: int = 3
    Banker2: int = 4
    Banker3: int = 5
    
    @classmethod
    def get_name(cls, value):
        try:
            return cls(value).name
        except ValueError:
            return None