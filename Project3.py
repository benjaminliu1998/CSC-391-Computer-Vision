from skimage.feature import hog
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import label2rgb
import cv2

from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data, exposure

from matplotlib import pyplot





# image1 = data.astronaut()
# image = cv2.imread("DSC02216.JPG")
#
# fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
#                     cells_per_block=(1, 1), visualize=True, multichannel=True)
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
#
# ax1.axis('off')
# ax1.imshow(image, cmap=plt.cm.gray)
# ax1.set_title('Input image')
#
# # Rescale histogram for better display
# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
#
# ax2.axis('off')
# ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
# ax2.set_title('Histogram of Oriented Gradients')
# plt.show()

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
    return ax.hist(lbp.ravel(), normed=True, bins=n_bins, range=(0, n_bins),
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

palm_1 = cv2.imread('palm_1.png')
palm_2 = cv2.imread('palm_2.png')
palm_3 = cv2.imread('palm_3.png')
palm_4 = cv2.imread('palm_4.png')
palm_5 = cv2.imread('palm_5.png')
palm_6 = cv2.imread('palm_6.png')
palm_7 = cv2.imread('palm_7.png')
palm_8 = cv2.imread('palm_8.png')
palm_9 = cv2.imread('palm_9.png')
palm_10 = cv2.imread('palm_10.png')
non_palm_1 = cv2.imread("non_palm_1.png")
non_palm_2 = cv2.imread("non_palm_2.png")
non_palm_3 = cv2.imread("non_palm_3.png")
non_palm_4 = cv2.imread("non_palm_4.png")
non_palm_5 = cv2.imread("non_palm_5.png")
non_palm_6 = cv2.imread("non_palm_6.png")
non_palm_7 = cv2.imread("non_palm_7.png")
non_palm_8 = cv2.imread("non_palm_8.png")
non_palm_9 = cv2.imread("non_palm_9.png")
non_palm_10 = cv2.imread("non_palm_10.png")
non_palm_11 = cv2.imread("non_palm_11.png")
mining_1 = cv2.imread("mining_1.png")
mining_2 = cv2.imread("mining_2.png")
river_1 = cv2.imread("river_1.png")
river_2 = cv2.imread("river_2.png")
leaf_1 = cv2.imread("leaf_1.png")
leaf_2 = cv2.imread("leaf_2.png")
branch_1 = cv2.imread("branch_1.png")
branch_2 = cv2.imread("branch_2.png")


# index 0 - 9 palm; 10 - 20 non_palm; 21-22 mining; 23-24 river;
image_array = []
image_array.append(palm_1)
image_array.append(palm_2)
image_array.append(palm_3)
image_array.append(palm_4)
image_array.append(palm_5)
image_array.append(palm_6)
image_array.append(palm_7)
image_array.append(palm_8)
image_array.append(palm_8)
image_array.append(palm_10)
image_array.append(non_palm_1)
image_array.append(non_palm_2)
image_array.append(non_palm_3)
image_array.append(non_palm_4)
image_array.append(non_palm_5)
image_array.append(non_palm_6)
image_array.append(non_palm_7)
image_array.append(non_palm_8)
image_array.append(non_palm_9)
image_array.append(non_palm_10)
image_array.append(non_palm_11)
image_array.append(leaf_1)
image_array.append(leaf_2)
image_array.append(branch_1)
image_array.append(branch_2)


# converting to grey
image_array_gray = []
for i in range(len(image_array)):
    image_array_gray.append(cv2.cvtColor(image_array[i], cv2.COLOR_RGB2GRAY))

image_name_array = ["palm_1", "palm_2", "palm_3", "palm_4", "palm_5", "palm_6", "palm_7", "palm_8", "palm_9", "palm_10",
                    "non_palm_1", "non_palm_2", "non_palm_3", "non_palm_4", "non_palm_5", "non_palm_6", "non_palm_7",
                    "non_palm_8", "non_palm_9", "non_palm_10", "non_palm_11",
                    "leaf_1", "leaf_2", "branch_1", "branch_2"]

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
    'non_palm_5': local_binary_pattern(image_array_gray[14], n_points, radius, METHOD),
    #'non_palm_11': local_binary_pattern(image_array_gray[20], n_points, radius, METHOD),
    #'river_1': local_binary_pattern(image_array_gray[23], n_points, radius, METHOD),
    #'mining_1': local_binary_pattern(image_array_gray[21], n_points, radius, METHOD),
}

# classify rotated textures | MATCHING
print('Matching using LBP:')
# print('original: brick, rotated: 30deg, match result: ',
#       match(refs, rotate(img_1, angle=30, resize=False)))
# print('original: brick, rotated: 70deg, match result: ',
#       match(refs, rotate(img_1, angle=70, resize=False)))
# print('original: grass, rotated: 145deg, match result: ',
#       match(refs, rotate(img_2, angle=145, resize=False)))
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
print('testing: ' + image_name_array[15], 'match result: ',
      match(database, image_array_gray[15]))  # non palm
print('testing: ' + image_name_array[16], 'match result: ',
      match(database, image_array_gray[16]))  # non palm
print('testing: ' + image_name_array[17], 'match result: ',
      match(database, image_array_gray[17]))  # non palm
print('testing: ' + image_name_array[18], 'match result: ',
      match(database, image_array_gray[18]))  # non palm
print('testing: ' + image_name_array[19], 'match result: ',
      match(database, image_array_gray[19]))  # non palm

forest_1 = cv2.cvtColor(cv2.imread("DSC02216.JPG"), cv2.COLOR_RGB2GRAY)
forest_2 = cv2.cvtColor(cv2.imread("DSC06464.JPG"), cv2.COLOR_RGB2GRAY)
forest_3 = cv2.cvtColor(cv2.imread("DSC06494.JPG"), cv2.COLOR_RGB2GRAY)
forest_4 = cv2.cvtColor(cv2.imread("DSC08606.JPG"), cv2.COLOR_RGB2GRAY)
print('testing: large forest image match result: ',
      match(database, forest_1))
print('testing: large forest image match result: ',
      match(database, forest_2))
print('testing: large forest image match result: ',
      match(database, forest_3))
print('testing: large forest image match result: ',
      match(database, forest_4))


for i in range(0,21,3):

    img_1 = cv2.cvtColor(image_array[i], cv2.COLOR_RGB2GRAY)
    img_2 = cv2.cvtColor(image_array[i+1], cv2.COLOR_RGB2GRAY)
    img_3 = cv2.cvtColor(image_array[i+2], cv2.COLOR_RGB2GRAY)


    refs = {
        'img_1': local_binary_pattern(img_1, n_points, radius, METHOD),
        'img_2': local_binary_pattern(img_2, n_points, radius, METHOD),
        'img_3': local_binary_pattern(img_3, n_points, radius, METHOD)
    }


    # plot histograms of LBP of textures
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
    plt.gray()

    ax1.imshow(img_1)
    ax1.axis('off')
    hist(ax4, refs['img_1'])
    ax4.set_ylabel('Percentage')
    ax4.set_title(image_name_array[i])

    ax2.imshow(img_2)
    ax2.axis('off')
    hist(ax5, refs['img_2'])
    ax5.set_xlabel('Uniform LBP values')
    ax5.set_title(image_name_array[i+1])


    ax3.imshow(img_3)
    ax3.axis('off')
    hist(ax6, refs['img_3'])
    ax6.set_title(image_name_array[i+2])

    #pyplot.imsave("image_array_0_to_2.png", fig)
    #cv2.imwrite("image_array_0_to_2.png", plt)
    fig.savefig('image_array_'+str(i)+'_to_'+str(i+2)+'.png')

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

for i in range(0,25):

    image = cv2.cvtColor(image_array[i], cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(image, n_points, radius, METHOD)


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

    fig2.savefig('image_array_'+str(i)+'.png')

# ============ HOG ===============

for i in range(0, 25):

    image = image_array[i]
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True)

    print(fd.shape)

    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')

    plt.figure(2)
    plt.plot(fd)
    plt.title(image_name_array[i])  # GIVES TITLE
    plt.savefig('image_array_' + str(i) + '_hog_plot.jpg')  # SAVES IMAGE

    fig3.savefig('image_array_' + str(i) + '_hog.png')

plt.show()
