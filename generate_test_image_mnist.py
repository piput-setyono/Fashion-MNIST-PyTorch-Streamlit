# -*- coding: utf-8 -*-
"""
Created on Wed May 29 15:53:32 2024

@author: Piput Setyono

This code to generate images from Fashion MNIST Dataset that will be used in application
"""
import torchvision

data_test = torchvision.datasets.FashionMNIST(root="data", train=False, download=True)

labels_map={
    0: 'T-shirt',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle Boot',
}

labels_count={
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
}

if not os.path.isdir("./mnist data test/"):
    os.makedirs("./mnist data test/")

for img, label in data_test:
    labels_count[label] += 1
    print(labels_map[label], labels_count[label], img.size, label)
    img.save("./mnist data test/" + labels_map[label] + '_' + str(labels_count[label]) + '.png')