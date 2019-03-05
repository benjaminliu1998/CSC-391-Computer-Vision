import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import peak_local_max


def print_howto():
    print("""
        1. Part 1: Canny Edge Detection - press 'c'
        2. Part 1: Harris Corner Detection - press 'h'
        3. Part 2: SIFT Descriptors - press 's'
        4. Quit - press 'esc'
    """)


# =============== Part 1 ===============
# ---------- Edge Detection ------------
def canny_image(image, thresh_1, thresh_2):
    canny_image = cv2.Canny(image, thresh_1, thresh_2)  # FIXME Parameters
    return canny_image


# ---------- Corner Detection ------------
def harris_image(image, blockSize, kSize, k):
    """
    :param image: Input image, it should be grayscale and float32 type
    :param blockSize: It is the size of neighbourhood considered for corner detection
    :param kSize: Aperture parameter of Sobel derivative used
    :param k: Harris detector free parameter in the equation
    :return: Image with corners detected
    """
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = np.float32(image_gray)
    dst = cv2.cornerHarris(image_gray, blockSize, kSize, k)
    # result is dilated for marking the corners
    dst = cv2.dilate(dst, None)
    # threshold for an optimal value
    image[dst > 0.02 * dst.max()] = [0, 0, 255]  # my parameter = 0.02
    return image


# =============== Part 2 ===============
# ----------- SIFT on Image ------------
def sift_image(image, name_string, sift):
    """
    :param image: The image used to do SIFT on
    :param name_string: Name of the image, type string; used to write name of the outcome image
    :param nfeat:
    :param sig:
    """
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #sift = cv2.xfeatures2d.SIFT_create(nfeatures=30, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=sig)  # FIXME Cannot pass variables in
    kp = sift.detect(image_gray, None)
    # image = cv2.drawKeypoints(image_gray, kp, image)  # gives equally sized blobs
    image = cv2.drawKeypoints(image, kp, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('sift_keypoints_' + name_string + '.jpg', image)



# ----------- SIFT on Video ------------
def sift_video(image):
    '''
    extract sift keypoints on image; no modifying parameters
    :param image: image that needs to be applied SIFT on
    :return: image containing keypoints
    '''
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # sift = cv2.xfeatures2d.SIFT_create()
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=30, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)  # FIXME Try out diff parameters
    kp = sift.detect(image_gray, None)
    # image = cv2.drawKeypoints(image_gray, kp, image)
    image = cv2.drawKeypoints(image, kp, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return image


# =============== Part 3 ===============
# ----- Match with Harris Corner -------
def harris_match(img_harris_1, img_harris_2):
    '''
    matching using harris
    :param img_harris_1: one of the images
    :param img_harris_2: the other image
    :return:
    '''
    sift = cv2.xfeatures2d.SIFT_create()  # FIXME Try out diff parameters
    bfmatcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=True)

    img_harris_gray_1 = cv2.cvtColor(img_harris_1, cv2.COLOR_BGR2GRAY)
    # img_harris_gray_1 = np.float32(img_harris_gray_1)
    corners_1 = cv2.cornerHarris(img_harris_gray_1, 3, 3, 0.01)  # FIXME Parameters
    kpsCorners_1 = np.argwhere(corners_1 > 0.01 * corners_1.max())
    kpsCorners_1 = [cv2.KeyPoint(pt[1], pt[0], 3) for pt in kpsCorners_1]
    # grayWithCorner_1 = cv2.drawKeypoints(img_harris_gray_1, kpsCorners_1, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kpsCorners_1, dscCorners_1 = sift.compute(img_harris_gray_1, kpsCorners_1)

    img_harris_gray_2 = cv2.cvtColor(img_harris_2, cv2.COLOR_BGR2GRAY)
    # img_harris_gray_2 = np.float32(img_harris_gray_2)
    corners_2 = cv2.cornerHarris(img_harris_gray_2, 3, 3, 0.01)  # FIXME Parameters
    kpsCorners_2 = np.argwhere(corners_2 > 0.01 * corners_2.max())
    kpsCorners_2 = [cv2.KeyPoint(pt[1], pt[0], 3) for pt in kpsCorners_2]
    # grayWithCorner_2 = cv2.drawKeypoints(img_harris_gray_2, kpsCorners_2, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kpsCorners_2, dscCorners_2 = sift.compute(img_harris_gray_2, kpsCorners_2)

    matchesCorners = bfmatcher.match(dscCorners_1, dscCorners_2)
    matchesCorners = sorted(matchesCorners, key=lambda x: x.distance)
    match_comparison = cv2.drawMatches(img_harris_1, kpsCorners_1, img_harris_2, kpsCorners_2,
                                       matchesCorners[:15], None, flags=2)  # top 15 matched keypoints
    return match_comparison

# ----- Match with SIFT Keypoints ------
def sift_match(img_sift_1, img_sift_2):
    """
    matching using sift
    :param img_sift_1: one of the images
    :param img_sift_2: the other image
    :return: image containing 2 images with lines indicating matches
    """
    sift = cv2.xfeatures2d.SIFT_create()
    bfmatcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=True)

    img_sift_gray_1 = cv2.cvtColor(img_sift_1, cv2.COLOR_BGR2GRAY)
    # img_sift_gray = np.float32(img_sift_gray)
    kpsSift_1 = sift.detect(img_sift_gray_1, None)
    # grayWithSift = cv2.drawKeypoints(img_sift_gray_1, kpsSift_1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kpsSift_1, dscSift_1 = sift.compute(img_sift_gray_1, kpsSift_1)

    img_sift_gray_2 = cv2.cvtColor(img_sift_2, cv2.COLOR_BGR2GRAY)
    # img_sift_gray = np.float32(img_sift_gray)
    kpsSift_2 = sift.detect(img_sift_gray_2, None)
    # grayWithSift = cv2.drawKeypoints(img_sift_gray_1, kpsSift_1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kpsSift_2, dscSift_2 = sift.compute(img_sift_gray_2, kpsSift_2)

    matchesSift = bfmatcher.match(dscSift_1, dscSift_2)
    matchesSift = sorted(matchesSift, key=lambda x: x.distance)
    match_comparison = cv2.drawMatches(img_sift_1, kpsSift_1, img_sift_2, kpsSift_2,
                                       matchesSift[:15], None, flags=2)  # top 15 matched keypoints
    return match_comparison


def rotate(img):  # from textbook
    num_rows, num_cols = img.shape[:2]
    translation_matrix = np.float32([[1, 0, int(0.5 * num_cols)],
                                     [0, 1, int(0.5 * num_rows)]])
    rotation_matrix = cv2.getRotationMatrix2D((num_cols, num_rows), 30, 1)
    img_translation = cv2.warpAffine(img, translation_matrix, (2 * num_cols,
                                                               2 * num_rows))
    img_rotation = cv2.warpAffine(img_translation, rotation_matrix,
                                  (num_cols * 2, num_rows * 2))
    return img_rotation


def scale(img):
    img_scaled = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    return img_scaled


def blur(img):
    gaussian_blur = cv2.GaussianBlur(img, (9, 9), 0)  # FIXME Parameters
    return gaussian_blur


def translation(img):
    num_rows, num_cols = img.shape[:2]
    translation_matrix = np.float32([[1, 0, 70], [0, 1, 110]])
    img_translation = cv2.warpAffine(img, translation_matrix, (num_cols,
                                                               num_rows), cv2.INTER_LINEAR)
    return img_translation


# ====================================== Main ===================================================
cap = cv2.VideoCapture(0)
# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")
cur_mode = None


# ------ Part 2: SIFT Experiments on Images ------
# trying out different parameters by creating sift function with different parameters
# then pass the sift function in to sift_image function

# normal (no translation / rotation / scale on image)
loaded_image = cv2.imread("ESB.jpg")  # FIXME HAVE TO REPEAT THIS LINE BELOW
loaded_image_name_string = "normal"
sift_image(loaded_image, loaded_image_name_string + "_50_16", cv2.xfeatures2d.SIFT_create(nfeatures=50, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6))
sift_image(loaded_image, loaded_image_name_string + "_50_25", cv2.xfeatures2d.SIFT_create(nfeatures=50, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=2.5))
sift_image(loaded_image, loaded_image_name_string + "_100_16", cv2.xfeatures2d.SIFT_create(nfeatures=100, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6))
sift_image(loaded_image, loaded_image_name_string + "_100_25", cv2.xfeatures2d.SIFT_create(nfeatures=100, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=2.5))
sift_image(loaded_image, loaded_image_name_string + "_0_16", cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6))  # nfeature = 0, sig = 1.6 default
sift_image(loaded_image, loaded_image_name_string + "_0_25", cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=2.5))

# translation
loaded_image = cv2.imread("ESB.jpg")
translation_image = translation(loaded_image)
translation_image_name_string = "translation"
sift_image(translation_image, translation_image_name_string + "_50_16", cv2.xfeatures2d.SIFT_create(nfeatures=50, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6))
sift_image(translation_image, translation_image_name_string + "_50_25", cv2.xfeatures2d.SIFT_create(nfeatures=50, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=2.5))
sift_image(translation_image, translation_image_name_string + "_100_16", cv2.xfeatures2d.SIFT_create(nfeatures=100, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6))
sift_image(translation_image, translation_image_name_string + "_100_25", cv2.xfeatures2d.SIFT_create(nfeatures=100, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=2.5))
sift_image(translation_image, translation_image_name_string + "_0_16", cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6))  # nfeature = 0, sig = 1.6 default
sift_image(translation_image, translation_image_name_string + "_0_25", cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=2.5))

# rotation
loaded_image = cv2.imread("ESB.jpg")
rotation_image = rotate(loaded_image)
rotation_image_name_string = "rotation"
sift_image(rotation_image, rotation_image_name_string + "_50_16", cv2.xfeatures2d.SIFT_create(nfeatures=50, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6))
sift_image(rotation_image, rotation_image_name_string + "_50_25", cv2.xfeatures2d.SIFT_create(nfeatures=50, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=2.5))
sift_image(rotation_image, rotation_image_name_string + "_100_16", cv2.xfeatures2d.SIFT_create(nfeatures=100, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6))
sift_image(rotation_image, rotation_image_name_string + "_100_25", cv2.xfeatures2d.SIFT_create(nfeatures=100, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=2.5))
sift_image(rotation_image, rotation_image_name_string + "_0_16", cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6))  # nfeature = 0, sig = 1.6 default
sift_image(rotation_image, rotation_image_name_string + "_0_25", cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=2.5))

# scale
loaded_image = cv2.imread("ESB.jpg")
scale_image = scale(loaded_image)
scale_image_name_string = "scale"
# scale_image_sift = sift_image(scale_image, scale_image_name_string)
sift_image(scale_image, scale_image_name_string + "_50_16", cv2.xfeatures2d.SIFT_create(nfeatures=50, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6))
sift_image(scale_image, scale_image_name_string + "_50_25", cv2.xfeatures2d.SIFT_create(nfeatures=50, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=2.5))
sift_image(scale_image, scale_image_name_string + "_100_16", cv2.xfeatures2d.SIFT_create(nfeatures=100, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6))
sift_image(scale_image, scale_image_name_string + "_100_25", cv2.xfeatures2d.SIFT_create(nfeatures=100, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=2.5))
sift_image(scale_image, scale_image_name_string + "_0_16", cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6))  # nfeature = 0, sig = 1.6 default
sift_image(scale_image, scale_image_name_string + "_0_25", cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=2.5))

# blur
loaded_image = cv2.imread("ESB.jpg")
blur_image = blur(loaded_image)
blur_image_name_string = "blur"
# blur_image_sift = sift_image(blur_image, blur_image_name_string)
sift_image(blur_image, blur_image_name_string + "_50_16", cv2.xfeatures2d.SIFT_create(nfeatures=50, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6))
sift_image(blur_image, blur_image_name_string + "_50_25", cv2.xfeatures2d.SIFT_create(nfeatures=50, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=2.5))
sift_image(blur_image, blur_image_name_string + "_100_16", cv2.xfeatures2d.SIFT_create(nfeatures=100, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6))
sift_image(blur_image, blur_image_name_string + "_100_25", cv2.xfeatures2d.SIFT_create(nfeatures=100, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=2.5))
sift_image(blur_image, blur_image_name_string + "_0_16", cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6))  # nfeature = 0, sig = 1.6 default
sift_image(blur_image, blur_image_name_string + "_0_25", cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=2.5))


# ---- Part 3：Keypoints and Matching -----
# ---------- Harris Corner ----------------
# normal (no translation / rotation / scale on image)
image_1 = cv2.imread('ESB.jpg')
image_2 = cv2.imread('ESB.jpg')
cv2.imwrite('harris_match.JPG', harris_match(image_1, image_2))

# scale
image_large = cv2.imread('ESB.jpg')
image_small = scale(image_large)
cv2.imwrite('harris_match_scale.JPG', harris_match(image_large, image_small))


# rotation
image_upright = cv2.imread('ESB.jpg')
image_rotated = rotate(image_upright)
cv2.imwrite('harris_match_rotation.JPG', harris_match(image_upright, image_rotated))

# blur
image_clear = cv2.imread('ESB.jpg')
image_blur = blur(image_clear)
cv2.imwrite('harris_match_blur.JPG', harris_match(image_clear, image_blur))


# different angle
image_angle_1 = cv2.imread('MacBook_1.jpeg')
image_angle_2 = cv2.imread('MacBook_2.jpeg')
cv2.imwrite('harris_match_angle.JPG', harris_match(image_angle_1, image_angle_2))


# translation
image_orig = cv2.imread('ESB.jpg')
image_translated = translation(image_orig)
cv2.imwrite('harris_match_trans.JPG', harris_match(image_orig, image_translated))


# ---------- SIFT Keypoints ----------------
# normal (no translation / rotation / scale on image)
image_1 = cv2.imread('ESB.jpg')
image_2 = cv2.imread('ESB.jpg')
cv2.imwrite('sift_match.JPG', sift_match(image_1, image_2))

# scale
image_large = cv2.imread('ESB.jpg')
image_small = scale(image_large)
cv2.imwrite('sift_match_scale.JPG', sift_match(image_large, image_small))

# rotation
image_upright = cv2.imread('ESB.jpg')
image_rotated = rotate(image_upright)
cv2.imwrite('sift_match_rotation.JPG', sift_match(image_upright, image_rotated))

# blur
image_clear = cv2.imread('ESB.jpg')
image_blur = blur(image_clear)
cv2.imwrite('sift_match_blur.JPG', sift_match(image_clear, image_blur))

# different angle
image_angle_1 = cv2.imread('MacBook_1.jpeg')
image_angle_2 = cv2.imread('MacBook_2.jpeg')
cv2.imwrite('sift_match_angle.JPG', sift_match(image_angle_1, image_angle_2))


# translation
image_orig = cv2.imread('ESB.jpg')
image_translated = translation(image_orig)
cv2.imwrite('sift_match_trans.JPG', sift_match(image_orig, image_translated))


# ---------- Part 1: Experiments on Video ----------
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    c = cv2.waitKey(1)  # display a frame for 1 ms, after which display will be automatically closed
    if c == 27:  # 27 = esc
        break
    if c != -1 and c != 255 and c != cur_mode:  # -1 and 255 are illegal
        cur_mode = c
    # ----- Harris -----
    if cur_mode == ord('h'):
        cv2.imshow('Harris Corner Detection', harris_image(frame, 2, 3, 0.04))  # block size = 2 # DEFAULT
    if cur_mode == ord('g'):
        cv2.imshow('Harris Corner Detection', harris_image(frame, 4, 3, 0.04))  # block size = 4
    if cur_mode == ord('i'):
        cv2.imshow('Harris Corner Detection', harris_image(frame, 6, 3, 0.04))  # block size = 6
    if cur_mode == ord('j'):
        cv2.imshow('Harris Corner Detection', harris_image(frame, 8, 3, 0.04))  # block size = 8

    # ----- Canny -----
    elif cur_mode == ord('c'):
        cv2.imshow('Canny Edge Detection', canny_image(frame, 100, 200))  # thresh_1 = 100, thresh_2 = 200 # DEFAULT
    elif cur_mode == ord('d'):
        cv2.imshow('Canny Edge Detection', canny_image(frame, 250, 300))  # thresh_1 = 250, thresh_2 = 300
    elif cur_mode == ord('e'):
        cv2.imshow('Canny Edge Detection', canny_image(frame, 30, 50))  # thresh_1 = 30, thresh_2 = 50
    elif cur_mode == ord('a'):
        cv2.imshow('Canny Edge Detection', canny_image(frame, 50, 150))  # thresh_1 = 50, thresh_2 = 150
    elif cur_mode == ord('b'):
        cv2.imshow('Canny Edge Detection', canny_image(frame, 150, 250))  # thresh_1 = 150, thresh_2 = 250
    elif cur_mode == ord('f'):
        cv2.imshow('Canny Edge Detection', canny_image(frame, 250, 350))  # thresh_1 = 250, thresh_2 = 350

    # ----- Canny Blur -----
    elif cur_mode == ord('w'):
        cv2.imshow('Blur Camera', blur(frame))
    elif cur_mode == ord('x'):
        cv2.imshow('Canny Edge Detection', canny_image(blur(frame), 50, 100))  # thresh_1 = 50, thresh_2 = 150
    elif cur_mode == ord('y'):
        cv2.imshow('Canny Edge Detection', canny_image(blur(frame), 150, 250))  # thresh_1 = 150, thresh_2 = 250
    elif cur_mode == ord('z'):
        cv2.imshow('Canny Edge Detection', canny_image(blur(frame), 250, 350))  # thresh_1 = 250, thresh_2 = 350

    # ----- SIFT -----
    elif cur_mode == ord('s'):
        cv2.imshow('SIFT Descriptor on Video', sift_video(frame))
    elif cur_mode == ord('t'):
        cv2.imshow('SIFT Descriptor on Video', sift_video(blur(frame)))
    else:
        cv2.imshow('Camera', frame)

cap.release()
key = cv2.waitKey(0)
cv2.destroyAllWindows()


