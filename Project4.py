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

# ========== descriptorLBP_new.py ==========


plt.rcParams['font.size'] = 9

# settings for LBP, for more info see
# http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_local_binary_pattern.html
#
radius = 3
n_points = 8 * radius  # FIXME: PLAY WITH PARAMETERS
METHOD = 'uniform'


# lpb is the local binary pattern computed for each pixel in the image
def hist(ax, lbp):  # FIXME: PLAY WITH PARAMETERS
    n_bins = int(lbp.max() + 1)
    return ax.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')

# The Kullback-Leibler divergence is a measure of how one probability distribution
# is different from a second, reference probability distribution.
# These probability distributions are the histograms computed from the LBP
# KL(p,q) = 0 means p and q distributions are identical.
def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))

# refs is an array reference LB patterns for various classes (brick, grass, wall)
# img is an input image
# match() gives the best match by comparing the KL divergence between the histogram
# of the img LBP and the histograms of the refs LBPs.
def match(refs, img):
    best_score = 10
    best_name = None
    lbp = local_binary_pattern(img, n_points, radius, METHOD)
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    for name, ref in refs.items():
        ref_hist, _ = np.histogram(ref, density=True, bins=n_bins,
                                   range=(0, n_bins))
        score = kullback_leibler_divergence(hist, ref_hist)
        if score < best_score:
            best_score = score
            best_name = name
    return best_name


#brick = data.load('brick.png')

# palm_1 = cv2.imread('palm_1.png')
# palm_2 = cv2.imread('palm_2.png')
# palm_3 = cv2.imread('palm_3.png')
# palm_4 = cv2.imread('palm_4.png')
# palm_5 = cv2.imread('palm_5.png')
# palm_6 = cv2.imread('palm_6.png')
# palm_7 = cv2.imread('palm_7.png')
# palm_8 = cv2.imread('palm_8.png')
# palm_9 = cv2.imread('palm_9.png')
# palm_10 = cv2.imread('palm_10.png')
# non_palm_1 = cv2.imread("non_palm_1.png")
# non_palm_2 = cv2.imread("non_palm_2.png")
# non_palm_3 = cv2.imread("non_palm_3.png")
# non_palm_4 = cv2.imread("non_palm_4.png")
# non_palm_5 = cv2.imread("non_palm_5.png")
# non_palm_6 = cv2.imread("non_palm_6.png")
# non_palm_7 = cv2.imread("non_palm_7.png")
# non_palm_8 = cv2.imread("non_palm_8.png")
# non_palm_9 = cv2.imread("non_palm_9.png")
# non_palm_10 = cv2.imread("non_palm_10.png")
#
image_array = []
# image_array.append(palm_1)
# image_array.append(palm_2)
# image_array.append(palm_3)
# image_array.append(palm_4)
# image_array.append(palm_5)
# image_array.append(palm_6)
# image_array.append(palm_7)
# image_array.append(palm_8)
# image_array.append(palm_8)
# image_array.append(palm_10)
# image_array.append(non_palm_1)
# image_array.append(non_palm_2)
# image_array.append(non_palm_3)
# image_array.append(non_palm_4)
# image_array.append(non_palm_5)
# image_array.append(non_palm_6)
# image_array.append(non_palm_7)
# image_array.append(non_palm_8)
# image_array.append(non_palm_9)
# image_array.append(non_palm_10)

def readFile(foldername):
    images = []
    if foldername == 'p':
        for i in range(0,110):
            images.append(cv2.imread('/Volumes/GoogleDrive/My Drive/#6 Spring 2019 Junior/CSC 391 Comp Vision/CSC391Projects/p/p_%s.png' % (i)))
    if foldername == 'np':
        for i in range(0, 110):
            images.append(cv2.imread('/Volumes/GoogleDrive/My Drive/#6 Spring 2019 Junior/CSC 391 Comp Vision/CSC391Projects/np/np_%s.png' % (i)))
    return images

# def readInPalm(n):
#     palm =[]
#
#     for i in range(0, n):
#         img = cv2.imread('/Users/xjq/PycharmProjects/FirstOpenCVProject/Project 3/data/p/p_%s.png' % (i))
#         img = cv2.resize(img,(100, 100),interpolation=cv2.INTER_AREA)
#         # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#         # cv2.imshow('img',img)
#         # cv2.waitKey(0)
#         palm.append(img)
#     return palm

image_array = readFile('p') + readFile('np')  # 0 - 109 palm; 110 - end non_palm

# converting to grey
image_array_gray = []
# for i in range(len(image_array)):
for img in image_array:
    image_array_gray.append(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))

# resize
# image_array_gray = cv2.resize(image_array_gray, (200, 200), interpolation=cv2.INTER_AREA)

image_name_array = []
image_label_array = []
for i in range(len(image_array)):
    if i < 110:
        image_name_array.append("palm_"+str(i))
        image_label_array.append("palm")
    else:
        image_name_array.append("non_palm"+str(i-110))
        image_label_array.append("non_palm")
# image_name_array = ["palm_1", "palm_2", "palm_3", "palm_4", "palm_5", "palm_6", "palm_7", "palm_8", "palm_9", "palm_10",
#                     "non_palm_1", "non_palm_2", "non_palm_3", "non_palm_4", "non_palm_5", "non_palm_6", "non_palm_7",
#                     "non_palm_8", "non_palm_9", "non_palm_10"]
# image_label_array = [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]  # 1 = palm, 0 = non palm

# the database
database = {
    'palm_1': local_binary_pattern(image_array_gray[0], n_points, radius, METHOD),
    #'palm_2': local_binary_pattern(image_array_gray[1], n_points, radius, METHOD),
    #'palm_3': local_binary_pattern(image_array_gray[2], n_points, radius, METHOD),
    #'palm_4': local_binary_pattern(image_array_gray[3], n_points, radius, METHOD),
    #'palm_5': local_binary_pattern(image_array_gray[4], n_points, radius, METHOD),
    #'non_palm_1': local_binary_pattern(image_array_gray[10], n_points, radius, METHOD),
    #'non_palm_2': local_binary_pattern(image_array_gray[11], n_points, radius, METHOD),
    #'non_palm_3': local_binary_pattern(image_array_gray[12], n_points, radius, METHOD),
    #'non_palm_4': local_binary_pattern(image_array_gray[13], n_points, radius, METHOD),
    'non_palm_5': local_binary_pattern(image_array_gray[115], n_points, radius, METHOD),
    #'non_palm_11': local_binary_pattern(image_array_gray[20], n_points, radius, METHOD),
    #'river_1': local_binary_pattern(image_array_gray[23], n_points, radius, METHOD),
    #'mining_1': local_binary_pattern(image_array_gray[21], n_points, radius, METHOD),
}

# classify rotated textures | MATCHING
print('Matching using LBP:')
print('testing: ' + image_name_array[5], 'match result: ',
      match(database, image_array_gray[5]))  # palm
print('testing: ' + image_name_array[6], 'match result: ',
      match(database, image_array_gray[6]))  # palm
print('testing: ' + image_name_array[7], 'match result: ',
      match(database, image_array_gray[7]))  # palm
print('testing: ' + image_name_array[8], 'match result: ',
      match(database, image_array_gray[8]))  # palm
print('testing: ' + image_name_array[9], 'match result: ',
      match(database, image_array_gray[9]))  # palm
print('testing: ' + image_name_array[110], 'match result: ',
      match(database, image_array_gray[110]))  # non palm
print('testing: ' + image_name_array[111], 'match result: ',
      match(database, image_array_gray[111]))  # non palm
print('testing: ' + image_name_array[112], 'match result: ',
      match(database, image_array_gray[112]))  # non palm
print('testing: ' + image_name_array[113], 'match result: ',
      match(database, image_array_gray[113]))  # non palm
print('testing: ' + image_name_array[114], 'match result: ',
      match(database, image_array_gray[114]))  # non palm


# for i in range(0,19,3):  # range
#
#     img_1 = cv2.cvtColor(image_array[i], cv2.COLOR_RGB2GRAY)
#     img_2 = cv2.cvtColor(image_array[i+1], cv2.COLOR_RGB2GRAY)
#     img_3 = cv2.cvtColor(image_array[i+2], cv2.COLOR_RGB2GRAY)
#
#
#     refs = {
#         'img_1': local_binary_pattern(img_1, n_points, radius, METHOD),
#         'img_2': local_binary_pattern(img_2, n_points, radius, METHOD),
#         'img_3': local_binary_pattern(img_3, n_points, radius, METHOD)
#     }
#
#
#     # plot histograms of LBP of textures
#     fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
#     plt.gray()
#
#     ax1.imshow(img_1)
#     ax1.axis('off')
#     hist(ax4, refs['img_1'])
#     ax4.set_ylabel('Percentage')
#     ax4.set_title(image_name_array[i])
#
#     ax2.imshow(img_2)
#     ax2.axis('off')
#     hist(ax5, refs['img_2'])
#     ax5.set_xlabel('Uniform LBP values')
#     ax5.set_title(image_name_array[i+1])
#
#
#     ax3.imshow(img_3)
#     ax3.axis('off')
#     hist(ax6, refs['img_3'])
#     ax6.set_title(image_name_array[i+2])
#
#     #pyplot.imsave("image_array_0_to_2.png", fig)
#     #cv2.imwrite("image_array_0_to_2.png", plt)
#     fig.savefig('image_array_'+str(i)+'_to_'+str(i+2)+'.png')

# =============================

# settings for LBP
radius = 3
n_points = 8 * radius


def overlay_labels(image, lbp, labels):
    mask = np.logical_or.reduce([lbp == each for each in labels])
    return label2rgb(mask, image=image, bg_label=0, alpha=0.5)


def highlight_bars(bars, indexes):
    for i in indexes:
        bars[i].set_facecolor('r')

def hist(ax, lbp):
    n_bins = int(lbp.max() + 1)
    return ax.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')

# ----


lbp_array = []  # am lbp array
hog_array = []  # am hog array


for i in range(len(image_array)):

    # ------------ LBP --------------
    image = cv2.cvtColor(image_array[i], cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_AREA)
    lbp = local_binary_pattern(image, n_points, radius, METHOD)
    # store in lbp array
    lbp_array.append(lbp)

    # plot histograms of LBP of textures
    fig2, (ax_img, ax_hist) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
    plt.gray()
    titles = ('edge ['+ image_name_array[i]+']', 'flat ['+ image_name_array[i]+']', 'corner ['+ image_name_array[i]+']')

    w = width = radius - 1
    edge_labels = range(n_points // 2 - w, n_points // 2 + w + 1)
    flat_labels = list(range(0, w + 1)) + list(range(n_points - w, n_points + 2))
    i_14 = n_points // 4            # 1/4th of the histogram
    i_34 = 3 * (n_points // 4)      # 3/4th of the histogram
    corner_labels = (list(range(i_14 - w, i_14 + w + 1)) +
                     list(range(i_34 - w, i_34 + w + 1)))

    label_sets = (edge_labels, flat_labels, corner_labels)

    for ax, labels in zip(ax_img, label_sets):
        ax.imshow(overlay_labels(image, lbp, labels))

    for ax, labels, name in zip(ax_hist, label_sets, titles):
        counts, _, bars = hist(ax, lbp)
        highlight_bars(bars, labels)
        ax.set_ylim(top=np.max(counts[:-1]))
        ax.set_xlim(right=n_points + 2)
        ax.set_title(name)

    ax_hist[0].set_ylabel('Percentage')
    for ax in ax_img:
        ax.axis('off')

    fig2.savefig('p4_image_array_'+str(i)+'.png') # FIXME: SAVE LBG IMAGES

    # ------------- HOG -------------
    image_resized = cv2.resize(image_array[i], (200, 200), interpolation=cv2.INTER_AREA)
    fd, hog_image = hog(image_resized, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True)
    hog_array.append(fd)

# ============ SVM ================
# splitting into training and testing sets
testing_image_array = []
training_image_array = []
testing_lbp_array = []
training_lbp_array = []
testing_label_array = []
training_label_array = []
testing_hog_array = []
training_hog_array = []

np.random.seed(10)
rand = np.random.choice(219, 55, replace=False)  # generating 55 numbers from 0 - 219 to be testing
for k in rand:
    testing_image_array.append(image_array_gray[k])
    testing_label_array.append(image_label_array[k])
    testing_lbp_array.append(lbp_array[k])
    testing_hog_array.append(hog_array[k])
for i in range(len(image_array)):
    if i not in rand:
        training_image_array.append(image_array_gray[i])
        training_label_array.append(image_label_array[i])
        training_lbp_array.append(lbp_array[i])
        training_hog_array.append(hog_array[i])


# ----------- FIXME: MATCH FUNCTION -----------
count = 0
database2 = {training_label_array[0]: training_lbp_array[0]}
for i in range(1,len(training_label_array)):
    database2[training_label_array[i]] = training_lbp_array[i]
    # if i < 110:
    #     database2[training_label_array[i] + str(i)] = training_lbp_array[i]
    # else:
    #     database2[training_label_array[i] + str(i-110)] = training_lbp_array[i]
for k in range(0,len(testing_label_array)):
    print('testing: ' + testing_label_array[k], 'match result: ', match(database2, testing_image_array[k]))
    if match(database2, testing_image_array[k]) == testing_label_array[k]:
        count = count +1
print('accuracy: ', count / len(testing_label_array))
# --------------------------------------


testing_image_array = np.array(testing_image_array)
training_image_array = np.array(training_image_array)
testing_lbp_array = np.array(testing_lbp_array)
training_lbp_array = np.array(training_lbp_array)
testing_label_array = np.array(testing_label_array)
training_label_array = np.array(training_label_array)
testing_hog_array = np.array(testing_hog_array)
training_hog_array = np.array(training_hog_array)

print(training_lbp_array.shape)
print(training_hog_array.shape)
print(len(testing_label_array))


# reshape lbp
num, x, y = training_lbp_array.shape
training_lbp_array = training_lbp_array.reshape((num, x*y))
num, x, y = testing_lbp_array.shape
testing_lbp_array = testing_lbp_array.reshape((num, x*y))


# Train a SVM classification model / LBP
clf = SVC(gamma='scale')
clf = clf.fit(training_lbp_array, training_label_array)
y_fit = clf.predict(testing_lbp_array)
print("---- LBP ----")
print(classification_report(testing_label_array, y_fit))
print(confusion_matrix(testing_label_array, y_fit))
cm = confusion_matrix(testing_label_array, y_fit)


# Train a SVM classification model / HOG
clf_hog = SVC(gamma='scale')
clf_hog = clf_hog.fit(training_hog_array, training_label_array)
y_fit_hog = clf_hog.predict(testing_hog_array)
print("---- HOG ----")
print(classification_report(testing_label_array, y_fit_hog))
print(confusion_matrix(testing_label_array, y_fit_hog))
#cm = confusion_matrix(testing_label_array, y_fit_hog)

for i in range(55):
    if testing_label_array[i] != y_fit[i]:
        print('Missclassified SVM LBP: ', rand[i])
    if testing_label_array[i] != y_fit_hog[i]:
        print('Missclassified SVM HOG: ', rand[i])

# ---------- FIXME: FANCY CONFUSION MATRIX ---------
# Only use the labels that appear in the data
classes = ["non_palm", "palm"]
cmap=plt.cm.Blues
title=None
normalize=False
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
ax.figure.colorbar(im, ax=ax)
# We want to show all ticks...
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       # ... and label them with the respective list entries
       xticklabels=classes, yticklabels=classes,
       title=title,
       ylabel='True label',
       xlabel='Predicted label')

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
fmt = '.2f' if normalize else 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()
fig.savefig('cm_lbp.png')


# ---------- FIXME: RANDOM FOREST -------------
# LBP
clf_rf = RandomForestClassifier()
clf_rf.fit(training_lbp_array, training_label_array)
preds = clf_rf.predict(testing_lbp_array)
print("RF LBP Accuracy:", accuracy_score(testing_label_array,preds))
# HOG
clf_rf_hog = RandomForestClassifier()
clf_rf_hog.fit(training_hog_array, training_label_array)
preds = clf_rf_hog.predict(testing_hog_array)
print("RF HOG Accuracy:", accuracy_score(testing_label_array,preds))

#plt.show()
