import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from mpl_toolkits.mplot3d import Axes3D

# ---------- 3.1 applies a spatial filter to an image and displays the results ----------
# Define a square filter of any size k x k.
# Create a 2-D box filter of size 3 x 3 and scale so that sum adds up to 1
image = cv2.imread("DSC_9259.JPG")
windowLen = 27
box_filter = np.ones((windowLen, windowLen), np.float32)
box_filter = box_filter / box_filter.sum()

# Apply the filter
# filtered_image = np.zeros(image.shape, np.float64)  # array for filtered_image
# filtered_image[:, :, 0] = cv2.filter2D(image[:, :, 0], -1, box_filter)
# filtered_image[:, :, 1] = cv2.filter2D(image[:, :, 1], -1, box_filter)
# filtered_image[:, :, 2] = cv2.filter2D(image[:, :, 2], -1, box_filter)
filtered_image = cv2.filter2D(image, -1, box_filter)

# Display the original image, the filter, and filtered image.
cv2.imshow("original image", image)  # original

plt.matshow(box_filter), plt.title("filter")  # filter
plt.show()

cv2.imshow("filtered image", filtered_image)  # filtered

# Save the filtered image to JPG files.
cv2.imwrite('filtered_image.jpg', filtered_image)


# ---------- 3.2.1 Smoothing and denoising ----------
puppy_noisy_image = cv2.imread("DSC_9259-0.50.JPG")
# gaussian smoothing
gaussian_3 = cv2.GaussianBlur(puppy_noisy_image, (3, 3), 0)
gaussian_9 = cv2.GaussianBlur(puppy_noisy_image, (9, 9), 0)
gaussian_27 = cv2.GaussianBlur(puppy_noisy_image, (27, 27), 0)

cv2.imwrite('gaussian_3.jpg', gaussian_3)
cv2.imwrite('gaussian_9.jpg', gaussian_9)
cv2.imwrite('gaussian_27.jpg', gaussian_27)

# median smoothing
median_3 = cv2.medianBlur(puppy_noisy_image, 3)
median_9 = cv2.medianBlur(puppy_noisy_image, 9)
median_27 = cv2.medianBlur(puppy_noisy_image, 27)

cv2.imwrite('median_3.jpg', median_3)
cv2.imwrite('median_9.jpg', median_9)
cv2.imwrite('median_27.jpg', median_27)


# ---------- 3.2.2 Edge detection ----------
canny_original = cv2.Canny(image, 100, 200)
cv2.imwrite('canny_original.jpg', canny_original)

canny_noisy = cv2.Canny(puppy_noisy_image, 300, 350)
cv2.imwrite('canny_noisy.jpg', canny_noisy)

image_nature = cv2.imread("window-00-10.jpg")
canny_nature = cv2.Canny(image_nature, 150, 200)
cv2.imwrite('canny_nature.jpg', canny_nature)

# ---------- 4.1 2-D DFT ----------
# converting to greyscale | using image = DSC_9259.JPG.jpg
image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2-D DFT and plotting coefficients

# Take the 2-D DFT and plot the magnitude of the corresponding Fourier coefficients
F2_image_grey = np.fft.fft2(image_grey.astype(float))

Y = (np.linspace(-int(image_grey.shape[0]/2), int(image_grey.shape[0]/2)-1, image_grey.shape[0]))
X = (np.linspace(-int(image_grey.shape[1]/2), int(image_grey.shape[1]/2)-1, image_grey.shape[1]))
X, Y = np.meshgrid(X, Y)

# plot the magnitude
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, np.fft.fftshift(np.abs(F2_image_grey)), cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
plt.title("Magnitude of Fourier Coefficients"), plt.show()  # Plot the magnitude as 3-D image

# plot the magnitude w/ Log(magnitude + 1) plot: shrinks the range so that small differences are visible
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, np.fft.fftshift(np.log(np.abs(F2_image_grey)+1)), cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
plt.title("Magnitude of Fourier Coefficients with Log"), plt.show()  # Plot the magnitude as 3-D image

# Plot the magnitude & log(magnitude + 1) as 2-D images (view from top)
magnitudeImage = np.fft.fftshift(np.abs(F2_image_grey))
magnitudeImage = magnitudeImage / magnitudeImage.max()   # scale to [0, 1]
magnitudeImage = ski.img_as_ubyte(magnitudeImage)
cv2.imshow("2-D Fourier Coef Mag", magnitudeImage)  # Plot the magnitude as 2-D image

logMagnitudeImage = np.fft.fftshift(np.log(np.abs(F2_image_grey)+1))
logMagnitudeImage = logMagnitudeImage / logMagnitudeImage.max()  # scale to [0, 1]
logMagnitudeImage = ski.img_as_ubyte(logMagnitudeImage)
cv2.imshow("2-D Fourier Coef Mag w/ log", logMagnitudeImage)  # Plot the magnitude as 2-D image

# save as jpg
cv2.imwrite('mag_2D.jpg', magnitudeImage)
cv2.imwrite('mag_2D_log.jpg', logMagnitudeImage)


# ---------- 4.2 Frequency Analysis ----------
# analyzing image content and the decay of the magnitude of the Fourier coefficients

image_2 = cv2.imread("window-00-10.jpg")
image_3 = cv2.imread("DSC_9259-0.50.JPG")  # puppy_noisy_image

# 1-D Fourier
# F_image_2 = np.fft.fft(image_2.astype(float))
# F_image_3 = np.fft.fft(image_3.astype(float))

# image
col_image = int(image.shape[1]/2)
# Obtain the image data for this column
colData_image = image[0:image.shape[0], col_image, 0]
# 1-D Fourier
F_colData_image = np.fft.fft(colData_image.astype(float))


# image_2
col_image_2 = int(image_2.shape[1]/2)
# Obtain the image data for this column
colData_image_2 = image_2[0:image_2.shape[0], col_image_2, 0]
# 1-D Fourier
F_colData_image_2 = np.fft.fft(colData_image_2.astype(float))

# image_3
col_image_3 = int(image_3.shape[1]/2)
# Obtain the image data for this column
colData_image_3 = image_3[0:image_3.shape[0], col_image_3, 0]
# 1-D Fourier
F_colData_image_3 = np.fft.fft(colData_image_3.astype(float))


# plot
# image
xvalues = np.linspace(-int(len(colData_image)/2), int(len(colData_image)/2)-1, len(colData_image))
#xvalues = np.linspace(0, len(colData_image), len(colData_image))
markerline, stemlines, baseline = plt.stem(xvalues, np.fft.fftshift(np.abs(F_colData_image)), 'g')  # fftshift() to center the low frequency coefficients around 0
plt.setp(markerline, 'markerfacecolor', 'g')
plt.setp(baseline, 'color','r', 'linewidth', 0.5)
plt.title("1-D Fourier Coef Mag - image"), plt.show()

# image_2
xvalues = np.linspace(-int(len(colData_image_2)/2), int(len(colData_image_2)/2)-1, len(colData_image_2))
#xvalues = np.linspace(0, len(colData_image_2), len(colData_image_2))
markerline, stemlines, baseline = plt.stem(xvalues, np.fft.fftshift(np.abs(F_colData_image_2)), 'g')  # fftshift() to center the low frequency coefficients around 0
plt.setp(markerline, 'markerfacecolor', 'g')
plt.setp(baseline, 'color','r', 'linewidth', 0.5)
plt.title("1-D Fourier Coef Mag - image_2"), plt.show()

# image_3
xvalues = np.linspace(-int(len(colData_image_3)/2), int(len(colData_image_3)/2)-1, len(colData_image_3))
#xvalues = np.linspace(0, len(colData_image_3), len(colData_image_3))
markerline, stemlines, baseline = plt.stem(xvalues, np.fft.fftshift(np.abs(F_colData_image_3)), 'g')  # fftshift() to center the low frequency coeffic
plt.setp(markerline, 'markerfacecolor', 'g')
plt.setp(baseline, 'color', 'r', 'linewidth', 0.5)
plt.title("1-D Fourier Coef Mag - image_3"), plt.show()


# plot log
# image
xvalues = np.linspace(-int(len(colData_image)/2), int(len(colData_image)/2)-1, len(colData_image))
#xvalues = np.linspace(0, len(colData_image), len(colData_image))
markerline, stemlines, baseline = plt.stem(xvalues, np.fft.fftshift(np.log(np.abs(F_colData_image))), 'g')  # fftshift() to center the low frequency coefficients around 0
plt.setp(markerline, 'markerfacecolor', 'g')
plt.setp(baseline, 'color','r', 'linewidth', 0.5)
plt.title("1-D Fourier Coef Mag - image - log"), plt.show()

# image_2
xvalues = np.linspace(-int(len(colData_image_2)/2), int(len(colData_image_2)/2)-1, len(colData_image_2))
#xvalues = np.linspace(0, len(colData_image_2), len(colData_image_2))
markerline, stemlines, baseline = plt.stem(xvalues, np.fft.fftshift(np.log(np.abs(F_colData_image_2))), 'g')  # fftshift() to center the low frequency coefficients around 0
plt.setp(markerline, 'markerfacecolor', 'g')
plt.setp(baseline, 'color','r', 'linewidth', 0.5)
plt.title("1-D Fourier Coef Mag - image_2 - log"), plt.show()

# image_3
xvalues = np.linspace(-int(len(colData_image_3)/2), int(len(colData_image_3)/2)-1, len(colData_image_3))
#xvalues = np.linspace(0, len(colData_image_3), len(colData_image_3))
markerline, stemlines, baseline = plt.stem(xvalues, np.fft.fftshift(np.log(np.abs(F_colData_image_3))), 'g')  # fftshift() to center the low frequency coeffic
plt.setp(markerline, 'markerfacecolor', 'g')
plt.setp(baseline, 'color', 'r', 'linewidth', 0.5)
plt.title("1-D Fourier Coef Mag - image_3 - log"), plt.show()

# ---------- 5.1 ----------

# use image_grey

# Explore the Butterworth filter
# U and V are arrays that give all integer coordinates in the 2-D plane
# Use U and V to create 3-D functions over (U,V)
U = (np.linspace(-int(image_grey.shape[0]/2), int(image_grey.shape[0]/2)-1, image_grey.shape[0]))
V = (np.linspace(-int(image_grey.shape[1]/2), int(image_grey.shape[1]/2)-1, image_grey.shape[1]))
U, V = np.meshgrid(U, V)
# The function over (U,V) is distance between each point (u,v) to (0,0)
D = np.sqrt(X*X + Y*Y)
# create x-points for plotting
xval = np.linspace(-int(image_grey.shape[1]/2), int(image_grey.shape[1]/2)-1, image_grey.shape[1])
# Specify a frequency cutoff value as a function of D.max()
#D0 = 0.25 * D.max()
D0 = 0.1 * D.max()

# The ideal lowpass filter makes all D(u,v) where D(u,v) <= 0 equal to 1
# and all D(u,v) where D(u,v) > 0 equal to 0
idealLowPass = D <= D0

# Filter our small grayscale image with the ideal lowpass filter
# 1. 2-D DFT of image
print(image_grey.dtype)
FT_image_grey = np.fft.fft2(image_grey.astype(float))
# 2. Butterworth filter is already defined in Fourier space
# 3. Elementwise product in Fourier space (notice fftshift of the filter)
FT_image_grey_filtered = FT_image_grey * np.fft.fftshift(idealLowPass)
# 4. Inverse DFT to take filtered image back to the spatial domain
image_grey_filtered = np.abs(np.fft.ifft2(FT_image_grey_filtered))

# Save the filter and the filtered image (after scaling)
idealLowPass = ski.img_as_ubyte(idealLowPass / idealLowPass.max())
image_grey_filtered = ski.img_as_ubyte(image_grey_filtered / image_grey_filtered.max())
cv2.imwrite("ideal_LowPass.jpg", idealLowPass)
cv2.imwrite("image_grey_IdealLowpassFiltered.jpg", image_grey_filtered)

# save for 5.2
image_grey_ideal_low_pass = image_grey_filtered

# Plot the ideal filter and then create and plot Butterworth filters of order
# n = 1, 2, 3, 4
plt.plot(xval, idealLowPass[int(idealLowPass.shape[0]/2), :], 'c--', label='ideal')
colors='brgkmc'
for n in range(1, 5):
    # Create Butterworth filter of order n
    H = 1.0 / (1 + (np.sqrt(2) - 1)*np.power(D/D0, 2*n))
    # Apply the filter to the grayscaled image
    FT_image_grey_filtered = FT_image_grey * np.fft.fftshift(H)
    image_grey_filtered = np.abs(np.fft.ifft2(FT_image_grey_filtered))
    image_grey_filtered = ski.img_as_ubyte(image_grey_filtered / image_grey_filtered.max())
    cv2.imwrite("image_grey_Butterworth-n" + str(n) + ".jpg", image_grey_filtered)
    # cv2.imshow('H', H)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    H = ski.img_as_ubyte(H / H.max())
    cv2.imwrite("butter-n" + str(n) + ".jpg", H)
    # Get a slice through the center of the filter to plot in 2-D
    slice = H[int(H.shape[0]/2), :]
    plt.plot(xval, slice, colors[n-1], label='n='+str(n))
    plt.legend(loc='upper left')

    # saving image_grey_filtered_butterworth_n for each n
    if n == 1:
        image_grey_filtered_butterworth_1 = image_grey_filtered
    elif n == 2:
        image_grey_filtered_butterworth_2 = image_grey_filtered
    elif n == 3:
        image_grey_filtered_butterworth_3 = image_grey_filtered
    elif n == 4:
        image_grey_filtered_butterworth_4 = image_grey_filtered

# plt.show()
plt.savefig('butterworthFilters.jpg', bbox_inches='tight')


# ---------- 5.2 Low-pass and High-pass Filtering ----------

# high pass (using image_grey_ideal_low_pass)
image_grey_high_pass = image_grey - image_grey_ideal_low_pass
cv2.imshow("image_grey_high_pass", image_grey_high_pass)
cv2.imwrite("image_grey_high_pass.jpg", image_grey_high_pass)

# high pass (using image_grey_filtered_butterworth_1)
image_grey_high_pass = image_grey - image_grey_filtered_butterworth_1
cv2.imshow("image_grey_high_pass_n1", image_grey_high_pass)
cv2.imwrite("image_grey_high_pass_n1.jpg", image_grey_high_pass)

# high pass (using image_grey_filtered_butterworth_2)
image_grey_high_pass = image_grey - image_grey_filtered_butterworth_2
cv2.imshow("image_grey_high_pass_n2", image_grey_high_pass)
cv2.imwrite("image_grey_high_pass_n2.jpg", image_grey_high_pass)

# high pass (using image_grey_filtered_butterworth_3)
image_grey_high_pass = image_grey - image_grey_filtered_butterworth_3
cv2.imshow("image_grey_high_pass_n3", image_grey_high_pass)
cv2.imwrite("image_grey_high_pass_n3.jpg", image_grey_high_pass)

# high pass (using image_grey_filtered_butterworth_4)
image_grey_high_pass = image_grey - image_grey_filtered_butterworth_4
cv2.imshow("image_grey_high_pass_n4", image_grey_high_pass)
cv2.imwrite("image_grey_high_pass_n4.jpg", image_grey_high_pass)


cv2.waitKey(0)
cv2.destroyAllWindows()



