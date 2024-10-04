'''
將指定資料夾中ori資料夾下的最後一張圖片取出

example: Bac/images/wrong/tableNo/*/ori/{last image}
'''
import os
from pathlib import Path
from time import strftime
from argparse import ArgumentParser
import shutil

from loguru import logger

from utils.bac_setting import BacSetting

def get_newest_img(folder: Path):
    '''
    folder example: **/tableNo/shoe_round/ori
    '''
    if folder.is_dir():
        all_imgs = [img for img in folder.glob('*.jpg')]

        if not all_imgs:
            logger.error("Can't found any img.")
            return None
        
        newest_img_path = max(all_imgs, key=os.path.getmtime)
        logger.info(f'get the newest img: {newest_img_path}')

        return newest_img_path
    
    else:
        logger.error('folder does not exist.')


parser = ArgumentParser()
parser.add_argument('-p', '--path', help='source folder path, ex: ./images/wrong/0C01', required=True)
args = parser.parse_args()

bac_setting = BacSetting()
today_date = strftime('%Y%m%d')
folder_name = f'{bac_setting.server_name}_{today_date}'

target_folder = Path(f'./{folder_name}')
target_folder.mkdir(parents=True, exist_ok=True)

source_folder = Path(args.path)

for p in source_folder.glob('*'):
    shoe_round = p.parts[-1]
    talbe = p.parts[-2]
    imgs_folder = p / 'ori'
    img_path = get_newest_img(imgs_folder)
    target_name = f'{talbe}_{shoe_round}.jpg'
    target_path = target_folder / target_name

    logger.info(f'copy image. from {img_path} to {target_path}')
    shutil.copy(img_path, target_path)

