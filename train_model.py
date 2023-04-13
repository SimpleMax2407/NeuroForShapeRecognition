import cv2
from datetime import datetime
import glob
import numpy as np
import os
import random

from neuro.networkGym import NeuroNetwork, NetworkGym

shapes = [1, 0, 0, 0]
noises = [3, 1, 1, 1]

size = 6
percents_for_train = 0.7

path_theta = r'neuro/theta'


def download_set(path_l, list_l, output_result, noised=0, duplications=1, inverse=True):

    for path in os.listdir(dir_path + path_l):
        img = cv2.imread(dir_path + path_l + "/" + path, cv2.IMREAD_UNCHANGED)

        if inverse:
            img = cv2.bitwise_not(img)

        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        img = img[x:x + w, y:y + h]
        img = cv2.resize(img, (size, size))
        img = img.flatten() / 255

        for _ in range(duplications):
            list_l.append({'result': output_result, 'arguments': img})

        for _ in range(0, noised):
            i_img = img.copy()

            for ind, i in enumerate(i_img):
                i_img[ind] = ((i + random.uniform(-0.1, 0.1) * 10) % 10) / 10.0

            list_l.append({'result': output_result, 'arguments': i_img})


def set_distribution(tr_set, t_set, list_l):
    indexes = np.sort(random.sample(range(0, len(list_l)), round(percents_for_train * len(list_l))))

    for i in range(0, len(list_l)):
        if indexes.any() and i == indexes[0]:
            tr_set.append(list_l[i])
            indexes = indexes[1:]
        else:
            t_set.append(list_l[i])
    return


circles = []
squares = []
stars = []
triangles = []
corrections = []

# folder with dataset path
dir_path = 'dataset'
path_save_true = r'/correction/true'
path_save_false = r'/correction/false'

print('Loading datasets:')

start_exe = datetime.now()
print('Circles...')
download_set(r'/shapes/circle', circles, shapes[0], noises[0])

print('Squares...')
download_set(r'/shapes/square', squares, shapes[1], noises[1])

print('Stars...')
download_set(r'/shapes/star', stars, shapes[2], noises[2])

print('Triangles...')
download_set(r'/shapes/triangle', triangles, shapes[3], noises[3])

print('Additional dataset...')
download_set(path_save_true, corrections, 1, duplications=2, inverse=False)
download_set(path_save_false, corrections, 0, duplications=1, inverse=False)

print('Loading complete.\n')

train_set = []
test_set = []

print('Filling datasets for neural networks...')
set_distribution(train_set, test_set, circles)
set_distribution(train_set, test_set, squares)
set_distribution(train_set, test_set, stars)
set_distribution(train_set, test_set, triangles)
set_distribution(train_set, test_set, corrections)


n = NeuroNetwork(size**2, 1, [size])

ng = NetworkGym(n, train_set, test_set, lamda=0.1)

start_train = datetime.now()
while True:
    print('\nFitting...')
    ng.train(alpha=12, number_of_iterations=125, write_log=True, speed_up=True)

    a, tt, ff = ng.test(0.7)

    print(f'\nAccuracy: {a:.1%}\n\n'
          f'Correct:    True - {tt:.1%}  False - {ff:.1%}\n')

    if tt > 0.9 and ff > 0.9:
        break

stop_train = datetime.now()

files = glob.glob(r'neuro\theta\*.npy')
for f in files:
    os.remove(f)

if not n.hidden:
    with open(path_theta + "\\theta.npy", 'wb') as f:
        np.save(f, n.theta)
else:
    for i, theta in enumerate(n.theta):
        with open(path_theta + "\\theta{0}.npy".format(i), 'wb') as f:
            np.save(f, theta)

stop_exe = datetime.now()

print(f'\nTraining time: {stop_train - start_train}\nExecution time: {stop_exe - start_exe}')

