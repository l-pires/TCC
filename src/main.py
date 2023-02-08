import os
from datetime import datetime
import cv2
from spoof_detect import *


PATH_IN = 'data/in/'
PATH_OUT = 'data/out/'

DETECT_MEAN_LIMIT = 10


if __name__ == '__main__':
    with open(PATH_OUT + 'log.txt', 'w') as file:
        figs = os.listdir(PATH_IN)

        file.write(datetime.now().strftime('%d/%m/%Y %H:%M\n'))

        for fig_name in figs:
            img = cv2.imread(f'{PATH_IN}{fig_name}')
            x, imgs = reflection_detect(img)
            file.write(f'"{fig_name}": {x:02.4f}\n')
            save_fig(imgs, f'{PATH_OUT}{fig_name}')
