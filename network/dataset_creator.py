import os
from random import random, uniform
from numpy import log, exp
from PIL import Image, ImageEnhance


def create_dataset(input_dir, output_dir):
    names = os.listdir(input_dir)
    paths = []
    for name in names:
        paths.append(input_dir + "/" + name)

    count = 0
    csv = ""
    for a in range(3):
        for i in paths:
            count += 1
            img = Image.open(i)
            br, co, sa = uniform(-1.6, 1.6), uniform(-1.6, 1.6), uniform(-1.6, 1.6)
            img = img.resize((200, 200))
            img = change_brightness(img, exp(br))
            img = change_contrast(img, exp(co))
            img = change_saturation(img, exp(sa))
            csv += '{},{},{},{}\n'.format(count, -br, -co, -sa)
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
            labels.append([float(split_row[1]), float(split_row[2]), float(split_row[3].split("\n")[0])])
    return labels