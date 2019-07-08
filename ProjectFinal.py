from __future__ import print_function
from skimage.feature import hog
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import label2rgb
import cv2
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data, exposure
from matplotlib import pyplot
from sklearn.cluster import KMeans
import sys
import os
import argparse
import json
import _pickle as pickle

from time import time
import logging

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ======= read file ========
palm_path = '/Volumes/GoogleDrive/My Drive/#6 Spring 2019 Junior/CSC 391 Comp Vision/CSC391Projects/Final Project Dataset/Palms/'
non_palm_path = '/Volumes/GoogleDrive/My Drive/#6 Spring 2019 Junior/CSC 391 Comp Vision/CSC391Projects/Final Project Dataset/Non Palms/'
forest_path = '/Volumes/GoogleDrive/My Drive/#6 Spring 2019 Junior/CSC 391 Comp Vision/CSC391Projects/Final Project Dataset/Forest/'

def readFile(foldername):
    images = []
    if foldername == 'p':
        # for filename in os.listdir(palm_path):
        for i in range(0, 800):
            images.append(cv2.imread(palm_path + 'palm_%s.jpg' % i))
    if foldername == 'np':
        # for filename in os.listdir(non_palm_path):
        for i in range(0, 3613):
            images.append(cv2.imread(non_palm_path + 'nonpalm_%s.jpg' % i))
    if foldername == 'f':
        for filename in os.listdir(forest_path):
            images.append(cv2.imread(forest_path + filename))
    return images


image_array = readFile('p') + readFile('np')

image_forest = readFile('f')

image_forest_0 = image_forest[0] # FIXME: !!!!!! CHANGE IMAGE HERE !!!!!!!

# resize
# image_array_gray = cv2.resize(image_array_gray, (200, 200), interpolation=cv2.INTER_AREA)

image_name_array = []
image_label_array = []
for i in range(len(image_array)):
    if i < 800:
        image_name_array.append("palm_"+str(i))
        image_label_array.append("palm")
    else:
        image_name_array.append("nonpalm_"+str(i-800))
        image_label_array.append("nonpalm")


# ----

hog_array = []  # am hog array

for i in range(len(image_array)):

    # image = cv2.cvtColor(image_array[i], cv2.COLOR_RGB2GRAY)
    # image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_AREA)

    # ------------- HOG -------------
    # image_resized = cv2.resize(image_array[i], (100, 100), interpolation=cv2.INTER_AREA)
    fd, hog_image = hog(image_array[i], orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True,
                        multichannel=True)
    hog_array.append(fd)

# ============ SVM ================
# splitting into training and testing sets
testing_image_array = []
training_image_array = []
testing_label_array = []
training_label_array = []
testing_hog_array = []
training_hog_array = []
predicted_label_array = []


# ----------- FIXME: USING RAND GENERATOR TO DIVIDE INTO TRAINING & TESTING
# np.random.seed(10)
# rand = np.random.choice(4412, 1200, replace=False)  # generating 55 numbers from 0 - 219 to be testing
# for k in rand:
#     testing_image_array.append(image_array[k])
#     testing_label_array.append(image_label_array[k])
#     # testing_lbp_array.append(lbp_array[k])
#     testing_hog_array.append(hog_array[k])
# for i in range(len(image_array)):
#     if i not in rand:
#         training_image_array.append(image_array[i])
#         training_label_array.append(image_label_array[i])
#         # training_lbp_array.append(lbp_array[i])
#         training_hog_array.append(hog_array[i])

# --------- FIXME: ALL IMAGE IN TRAINING + ONE CROPPED PART AS TESTING
training_image_array = image_array
training_label_array = image_label_array
training_hog_array = hog_array


hgt, wdt = image_forest_0.shape[:2]
start_row, start_col = int(0), int(0)
delta_height = int(100)
delta_width = int(100)
height_range = int(hgt / 100)
width_range = int(wdt / 100)
for i in range(0, height_range):
    for j in range(0, width_range):
        start_row, start_col = i * delta_height, j * delta_width
        end_row, end_col = (i + 1) * delta_height, (j + 1) * delta_width
        cropped = image_forest_0[start_row:end_row, start_col:end_col]
        fd, hog_image = hog(cropped, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True,
                            multichannel=True)
        testing_hog_array.append(fd)

# --------------------------------------

testing_image_array = np.array(testing_image_array)
training_image_array = np.array(training_image_array)
# testing_lbp_array = np.array(testing_lbp_array)
# training_lbp_array = np.array(training_lbp_array)
testing_label_array = np.array(testing_label_array)
training_label_array = np.array(training_label_array)
testing_hog_array = np.array(testing_hog_array)
training_hog_array = np.array(training_hog_array)

# print(training_lbp_array.shape)
# print(training_hog_array.shape)
# print(len(testing_label_array))


# Train a SVM classification model / HOG
clf_hog = SVC(kernel='rbf', gamma='scale') #'scale'
clf_hog = clf_hog.fit(training_hog_array, training_label_array)
y_fit_hog = clf_hog.predict(testing_hog_array)
predicted_label_array = y_fit_hog

print("label added in array") # FIXME

# accuracy
# print("---- HOG ----")
# print(classification_report(testing_label_array, y_fit_hog))
# print(confusion_matrix(testing_label_array, y_fit_hog))
# cm = confusion_matrix(testing_label_array, y_fit_hog)


# misclassified
# for i in range(55):
#     # if testing_label_array[i] != y_fit[i]:
#     #     print('Missclassified SVM LBP: ', rand[i])
#     if testing_label_array[i] != y_fit_hog[i]:
#         print('Missclassified SVM HOG: ', rand[i])


# ---------- FIXME: FANCY CONFUSION MATRIX ---------
# Only use the labels that appear in the data
# classes = ["non_palm", "palm"]
# cmap=plt.cm.Blues
# title=None
# normalize=False
# fig, ax = plt.subplots()
# im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
# ax.figure.colorbar(im, ax=ax)
# # We want to show all ticks...
# ax.set(xticks=np.arange(cm.shape[1]),
#        yticks=np.arange(cm.shape[0]),
#        # ... and label them with the respective list entries
#        xticklabels=classes, yticklabels=classes,
#        title=title,
#        ylabel='True label',
#        xlabel='Predicted label')
#
# # Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#          rotation_mode="anchor")
#
# # Loop over data dimensions and create text annotations.
# fmt = '.2f' if normalize else 'd'
# thresh = cm.max() / 2.
# for i in range(cm.shape[0]):
#     for j in range(cm.shape[1]):
#         ax.text(j, i, format(cm[i, j], fmt),
#                 ha="center", va="center",
#                 color="white" if cm[i, j] > thresh else "black")
# fig.tight_layout()
# fig.savefig('cm_hog_svm_testing_1.png')


# ============= FIXME: CREATE ITERATE THRU MATRIX =============
hgt, wdt = image_forest_0.shape[:2]

print(image_forest_0.shape)  # FIXME: PREVIOUSLY OUTPUT MATRIX SHAPE WAS (10, 100, 100, 100, 3)


# =========== FIXME: CROP =============
hgt, wdt = image_forest_0.shape[:2]
start_row, start_col = int(0), int(0)
delta_height = int(100)
delta_width = int(100)
height_range = int(hgt / 100)
width_range = int(wdt / 100)
# FIXME: CREATE A (HEIGHT, WIDTH, 3) DIMENSIONAL OUTPUT MATRIX
cropped_image_matrix = np.zeros(shape=(hgt,wdt,3))
k = 0
for i in range(0, height_range):
    for j in range(0, width_range):
        start_row, start_col = i * delta_height, j * delta_width
        end_row, end_col = (i + 1) * delta_height, (j + 1) * delta_width
        cropped = image_forest_0[start_row:end_row, start_col:end_col]

        if predicted_label_array[k] == "nonpalm":
            gray_patch = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
            cropped_image_matrix[start_row:end_row, start_col:end_col] = cv2.cvtColor(gray_patch, cv2.COLOR_GRAY2RGB)
        else:
            cropped_image_matrix[start_row:end_row, start_col:end_col] = cropped
        k = k+1


print(cropped_image_matrix.shape)
cv2.imwrite("forest_highlighted.jpg", cropped_image_matrix)


#plt.show()
