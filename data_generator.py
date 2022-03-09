import numpy as np
import cv2

import settings

from numpy import random


distances = [-50.000000, -49.000000, -48.001440, -47.004320, -46.008640, -45.014400, -44.021600, -43.030240, -42.040320,
             -41.051840, -40.064800, -39.079200, -38.095040, -37.112320, -36.131040, -35.151200, -34.172800, -33.195840,
             -32.220320, -31.246240, -30.273600, -29.302400, -28.332640, -27.364320, -26.397440, -25.432000, -24.468000,
             -23.505440, -22.544320, -21.584640, -20.626400, -19.669600, -18.714240, -17.760320, -16.807840, -15.856800,
             -14.907200, -13.959040, -13.012320, -12.067040, -11.123200, -10.180800, -9.239840, -8.300320, -7.362240,
             -6.425600, -5.490400, -4.556640, -3.624320, -2.693440, -1.764000, -0.836000, 0.090560, 1.015680, 1.939360,
             2.861600, 3.782400, 4.701760, 5.619680, 6.536160, 7.451200, 8.364800, 9.276960, 10.187680, 11.096960,
             12.004800, 12.911200, 13.816160, 14.719680, 15.621760, 16.522400, 17.421600, 18.319360, 19.215680,
             20.110560, 21.004000, 21.896000, 22.786560, 23.675680, 24.563360, 25.449600, 26.334400, 27.217760,
             28.099680, 28.980160, 29.859200, 30.736800, 31.612960, 32.487680, 33.360960, 34.232800, 35.103200,
             35.972160, 36.839680, 37.705760, 38.570400, 39.433600, 40.295360, 41.155680, 42.014560]

window = settings.IMAGE_SIZE[0]


def data_gen(data_info: list, batch_size: int, num_roi: int, pre_processing) -> (np.ndarray, np.array):
    cnt = 0
    while True:
        x = np.zeros((batch_size * num_roi,) + settings.IMAGE_SIZE, dtype=np.float)
        y = np.zeros((batch_size * num_roi, 1), dtype=np.float)

        for n in range(0, batch_size):
            if cnt >= len(data_info):
                break

            file_path, distance_z = data_info[cnt]
            img = cv2.imread(file_path, 1)

            cnt += 1

            for k in range(0, num_roi):
                x_r = random.randint(1024 - window)
                y_r = random.randint(1024 - window)
                img_n = img[y_r:y_r + window, x_r:x_r + window]
                x[n * batch_size + k] = pre_processing(img_n)
                y[n * batch_size + k] = distances[distance_z]

        if cnt >= len(data_info):
            cnt = 0

        yield x, y
