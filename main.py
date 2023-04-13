import cv2
import numpy as np
import os

from neuro.neuroNetwork import NeuroNetwork
from process_image import process_image


path_img = r'C:/Users/Maxim/Desktop/image5.jpg'
path_theta = r'neuro/theta'
path_save_true = r'dataset/correction/true'
path_save_false = r'dataset/correction/false'

cap = cv2.VideoCapture(0)


def nothing():
    pass


def init_cam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()


def read_cam():
    ret, frame = cap.read()
    if not ret:
        print("Cannot read image from camera")
        exit()

    return frame


def read_img():
    return cv2.imread(path_img)


def save_img(img, value):
    print(f"Saving as {value}")
    path_f = path_save_true if value else path_save_false
    cv2.imwrite("{0}/{1}.png".format(path_f, len(os.listdir(path_f))), img)


n = NeuroNetwork()

theta = []

for path in os.listdir(path_theta):
    if path == 'theta.npy':
        with open(path_theta + f'\\theta.npy', 'rb') as f:
            theta = np.load(f)
            break
    else:
        with open(path_theta + '\\' + path, 'rb') as f:
            theta.append(np.load(f))

n.get_theta(theta)
size = round(np.sqrt(n.inputs))

process_image(n, size, init_cam, read_cam, save_img)
