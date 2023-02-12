import os
from datetime import datetime
import json
import cv2
from tqdm import tqdm

from spoof_detect import *


PATH_IN = 'data/in/'
PATH_OUT = 'data/out/'

DETECT_MEAN_LIMIT = 10


def cls():
    os.system('cls' if os.name=='nt' else 'clear')


if __name__ == '__main__':
    figs = os.listdir(PATH_IN)

    log = {
        'datetime': datetime.now().__str__(), #.strftime('%d/%m/%Y %H:%M'),
        'data': {}
    }

    cls()
    for threshold in tqdm(range(200, 256)):
        log['data'][threshold] = {}

        path = f'{PATH_OUT}{threshold}/'
        os.mkdir(path)

        for fig_name in tqdm(figs):
            img = cv2.imread(f'{PATH_IN}{fig_name}')
            x, imgs = reflection_detect(img, threshold=threshold)

            log['data'][threshold][fig_name] = x
            save_fig(imgs, f'{path}{fig_name}')

        cls()
    
    with open(f'{PATH_OUT}log.json', 'w') as file:
        file.write(json.dumps(log))
