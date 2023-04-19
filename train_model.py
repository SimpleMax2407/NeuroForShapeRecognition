import cv2
from datetime import datetime
import glob
import json
import numpy as np
import os
import random

from neuro.networkGym import NeuroNetwork, NetworkGym

shapes = [1, 2, 3, 4]
shapes_names = ["Circles", "Squares", "Stars", "Triangles"]

number_of_noised_samples = 3700

size_x = 8
size_y = 8

max_number_of_iterations = 150

percents_for_train = 0.7

path_theta = r'neuro/theta'


def download_set(path_l, list_l, output_result, duplications=1, inverse=True):

    for path in os.listdir(dir_path + path_l):
        img = cv2.imread(dir_path + path_l + "/" + path, cv2.IMREAD_UNCHANGED)

        if inverse:
            img = cv2.bitwise_not(img)

        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        img = img[x:x + w, y:y + h]
        img = cv2.resize(img, (size_x, size_y))
        img = img.flatten() / 255.0

        for _ in range(duplications):
            list_l.append({'result': output_result, 'arguments': img})


def set_distribution(tr_set, t_set, list_l):
    indexes = np.sort(random.sample(range(0, len(list_l)), round(percents_for_train * len(list_l))))

    for itr in range(0, len(list_l)):
        if indexes.any() and itr == indexes[0]:
            tr_set.append(list_l[itr])
            indexes = indexes[1:]
        else:
            t_set.append(list_l[itr])
    return


def check_boolean_function(lst, possibility):
    for element in lst:
        if not element >= possibility:
            return False
    return True


circles = []
squares = []
stars = []
triangles = []
noised_samples = []

# folder with dataset path
dir_path = 'dataset'
path_save_true = r'/correction/true'
path_save_false = r'/correction/false'

print('Loading datasets:')

# start timer for measurement execution time
start_exe = datetime.now()

# load dataset
print('Circles...')
download_set(r'/shapes/circle', circles, shapes[0])

print('Squares...')
download_set(r'/shapes/square', squares, shapes[1])

print('Stars...')
download_set(r'/shapes/star', stars, shapes[2])

print('Triangles...')
download_set(r'/shapes/triangle', triangles, shapes[3])

print('Noised samples...')
for _ in range(number_of_noised_samples):
    noised_samples.append({'result': 0, 'arguments': list(round(random.uniform(0, 1), 2)
                                                          for _ in range(size_x * size_y))})

print('Loading complete.\n')

train_set = []
test_set = []

# filling datasets for learning and testing
print('Filling datasets for neural networks...')
set_distribution(train_set, test_set, circles)
set_distribution(train_set, test_set, squares)
set_distribution(train_set, test_set, stars)
set_distribution(train_set, test_set, triangles)
set_distribution(train_set, test_set, noised_samples)

# creation of NN
n = NeuroNetwork(size_x * size_y, max(shapes), [size_x + size_y])

# creating entity for learning and testing NN
ng = NetworkGym(n, train_set, test_set)

old_stat = [None] * (max(shapes) + 1)

# start timer for measurement learning time
start_train = datetime.now()

iterations = 0
for _ in range(max_number_of_iterations):

    iterations += 1

    # learning NN
    print('\nFitting...')
    ng.train(alpha=4, lamda=0.1, number_of_iterations=20, write_log=False)   # , write_log=True, speed_up=True

    # testing NN
    a, stat, af = ng.test(0.7)

    print(f'\nAccuracy: {a:.1%}\n')
    print(f'Statistics:')

    # print testing results
    for i in range(0, len(stat)):
        # if there are no undefined testing samples then don't print statistics about undefined samples recognition
        if not af:
            continue

        ch = ""
        if old_stat[0] is not None:
            ch = ' ({}%)'.format(round((stat[i] - old_stat[i])*100, 1))

        print(f'{shapes_names[i - 1] if i > 0 else "Undefined"}: {stat[i]:.1%}{ch}')

    if check_boolean_function(stat if af else stat[1:], 0.9):
        break

    old_stat = stat

stop_train = datetime.now()

files = glob.glob(r'neuro\theta\*.npy')

for f in files:
    os.remove(f)

with open(path_theta + "\\settings.json", 'w') as f:
    json.dump({"size_x": size_x, "size_y": size_y, "shapes_names": shapes_names}, f)

if not n.hidden:
    with open(path_theta + "\\theta.npy", 'wb') as f:
        np.save(f, n.theta)
else:
    for i, theta in enumerate(n.theta):
        with open(path_theta + "\\theta{0}.npy".format(i), 'wb') as f:
            np.save(f, theta)

stop_exe = datetime.now()

print(f'\nTraining time: {stop_train - start_train}\nExecution time: {stop_exe - start_exe}')
print(f'Number of iterations: {iterations}')
