from random import random, uniform

from PIL import Image, ImageEnhance


def create_dataset(inputs, output_dir):
    count = 0
    csv = ""
    for i in inputs:
        count += 1
        img = Image.open(i)
        br, co, sa, sh = uniform(0.2, 5), uniform(0.2, 5), uniform(0.2, 5), uniform(0.2, 5)
        img = change_brightness(img, br)
        img = change_contrast(img, co)
        img = change_saturation(img, sa)
        img = change_sharpness(img, sh)
        csv += '{},{},{},{},{}\n'.format(count, 1/br, 1/co, 1/sa, 1/sh)
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
            labels.append([split_row[1], split_row[2], split_row[3], split_row[4]])
    return labels


create_dataset(["/home/bernardo/Desktop/Images/No animals/1-10/landscape-photography-15.jpg"], "/home/bernardo/Desktop/Images/No animals/Dataset1")