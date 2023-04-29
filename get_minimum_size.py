import csv
import cv2
import numpy as np
import os
import random

from neuro.networkGym import NeuroNetwork, NetworkGym

shapes = [1, 2, 3, 4]
shapes_names = ["White noises", "Circles", "Squares", "Stars", "Triangles"]

number_of_noised_samples = 3700

min_size = 3
max_size = 12

max_number_of_iterations = 50
number_of_attempts = 3

percents_for_train = 0.67

path_theta = r'neuro/theta'

# folder with dataset path
dir_path = 'dataset'
path_save_true = r'/correction/true'
path_save_false = r'/correction/false'


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
        img = cv2.resize(img, (size, size))
        img = img.flatten() / 255.0

        img = [1 if pix > 0.5 else 0 for pix in img]

        for _ in range(duplications):
            list_l.append({'result': output_result, 'arguments': img})


def set_distribution(tr_set, t_set, list_l):
    indexes = np.sort(random.sample(range(0, len(list_l)), round(percents_for_train * len(list_l))))

    for it in range(0, len(list_l)):
        if indexes.any() and it == indexes[0]:
            tr_set.append(list_l[it])
            indexes = indexes[1:]
        else:
            t_set.append(list_l[it])
    return


def check_boolean_function(lst, possibility):
    for element in lst:
        if not element >= possibility:
            return False
    return True


statistics = []

for size in range(min_size, max_size + 1):

    circles = []
    squares = []
    stars = []
    triangles = []
    noised_samples = []

    print(f'\n\n -------------------------------------------------------- \n Learning NN with size {size} pixels\n')
    print('Loading datasets:')

    # load datasets
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
                                                              for _ in range(size ** 2))})

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

    s = {}

    for att in range(number_of_attempts):

        # creation of NN
        n = NeuroNetwork(size ** 2, max(shapes), [36])

        # creating entity for learning and testing NN
        ng = NetworkGym(n, train_set, test_set)

        old_accuracy = 0.0
        af = False
        old_stat = [None] * (max(shapes) + 1)

        print(f"\nAttempt #{att + 1}")
        for itr in range(max_number_of_iterations):

            # learning NN
            print(f'\nFitting (iteration #{itr + 1})...')
            ng.train(alpha=4, lamda=0.1, number_of_iterations=50, write_log=False)

            # testing NN
            accuracy, stat, af = ng.test(0.7)

            print(f'\nAccuracy: {accuracy:.1%} ({accuracy - old_accuracy:.1%})\n')
            print(f'Statistics (size {size} px):')

            # print testing results
            for i in range(0, len(stat)):
                # if there are no undefined testing samples then don't print statistics about undefined samples
                if not af:
                    continue

                ch = ""
                if old_stat[0] is not None:
                    ch = ' ({}%)'.format(round((stat[i] - old_stat[i])*100, 1))

                print(f'{shapes_names[i]}: {stat[i]:.1%}{ch}')

            old_stat = stat
            old_accuracy = accuracy

        if att == 0 or old_accuracy > s['Accuracy']:
            s = {
                'Number of pixels': size,
                'Accuracy': old_accuracy,
            }

            for i in range(0, len(old_stat)):
                s[shapes_names[i]] = old_stat[i]

    statistics.append(s)

with open('Test.csv', 'w', newline='') as csvfile:

    writer = csv.DictWriter(csvfile, fieldnames=statistics[0].keys(), delimiter=';')
    writer.writeheader()
    for s in statistics:
        writer.writerow(s)


with open('Test.csv', 'r') as csvfile:
    csv_content = csvfile.read()

csv_content = csv_content.replace('.', ',')

with open('Test.csv', 'w', newline='') as csvfile:
    csvfile.write(csv_content)