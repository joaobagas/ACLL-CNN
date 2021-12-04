import os
from random import random, uniform
from numpy import log, exp
from PIL import Image, ImageEnhance


def create_dataset(inputs, output_dir):
    count = 0
    csv = ""
    for i in inputs:
        count += 1
        img = Image.open(i)
        br, co, sa, sh = uniform(-1.6, 1.6), uniform(-1.6, 1.6), uniform(-1.6, 1.6), uniform(-1.6, 1.6)
        img = img.resize((640, 480))
        img = change_brightness(img, exp(br))
        img = change_contrast(img, exp(co))
        img = change_saturation(img, exp(sa))
        img = change_sharpness(img, exp(sh))
        csv += '{},{},{},{},{}\n'.format(count, -br, -co, -sa, -sh)
        img.save(output_dir + '/{}.jpg'.format(count))
    f = open(output_dir + "/labels.csv", "w+")
    f.write(csv)


def change_brightness(img, value):
    image_enhancer = ImageEnhance.Brightness(img)
    return image_enhancer.enhance(value)


def change_contrast(img, value):
    image_enhancer = ImageEnhance.Contrast(img)
    return image_enhancer.enhance(value)


def change_saturation(img, value):
    image_enhancer = ImageEnhance.Color(img)
    return image_enhancer.enhance(value)


def change_sharpness(img, value):
    image_enhancer = ImageEnhance.Sharpness(img)
    return image_enhancer.enhance(value)


def load_labels(folder):
    labels = []
    with open(folder + "/labels.csv", newline='') as csv_file:
        for row in csv_file:
            split_row = row.split(",")
            labels.append([float(split_row[1]), float(split_row[2]), float(split_row[3]), float(split_row[4].split("\n")[0])])
    return labels

paths = os.listdir("/home/bernardo/Desktop/NordicDenmark")
test = []
for path in paths:
    test.append("/home/bernardo/Desktop/NordicDenmark/" + path)
create_dataset(test, "/home/bernardo/Desktop/Images/No animals/Dataset1")