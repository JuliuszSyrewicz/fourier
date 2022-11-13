import numpy as np
from matplotlib import pyplot as plt
# apply a linear filter to a given neighbourhood
def applyFilter(kernel, image, x, y, mode):

    # set sign needed for kernel access
    if mode == "correlation":
        sign = 1
    elif mode == "convolution":
        sign = -1
    else:
        print("unknown operation")
        sign = 0

    # neighbourhood is of size (2a+1)x(2b+1)
    a = (kernel.shape[0] - 1) // 2
    b = (kernel.shape[1] - 1) // 2
    size = image.shape[0]
    sum = 0
    for i in range(-a, a+1, 1):
        for j in range(-b, b+1, 1):
            value = image[(x + sign * i) % size][(y + sign * j) % size]
            sum += kernel[i][j] * value
    return sum

# filter an entire image
def linearFilter(kernel, image, mode ="correlation"):
    filtered = np.copy(image)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            filtered[x][y] = applyFilter(kernel, image, x, y, mode)
    return filtered

# return square root of sum of squares for every pixel in two separate images
def rootOfSquares(img1, img2):
    assert img1.shape == img2.shape
    combined = np.copy(img1)
    for x in range(combined.shape[0]):
        for y in range(combined.shape[1]):
            combined[x][y] = np.sqrt(img1[x][y]**2 + img2[x][y]**2)
    return combined



def lowPassFilter(n, center=None, radius=None):
    if not center:  # use the middle of the image
        center = int(n / 2)
    if not radius:  # use the smallest distance between the center and image walls
        radius = min(center, n - center)

    Y, X = np.ogrid[:n, :n]
    dist_from_center = np.sqrt((X - center) ** 2 + (Y - center) ** 2)
    mask = dist_from_center <= radius
    return mask

def bandStopFilter(n, omega, r, r0):
    mask = np.zeros((n, n))
    for u in range(n):
        for v in range(n):
            mask[u][v] = 1 / (1 + (omega * r / np.sqrt(np.abs(u**2 + v**2 - r0**2)))**(2*n))
    return mask

# sinusoidal gratings useful for visualizing Fourier bases
def getSinusoidalGrating(frequency, theta = np.pi/4, limit = 5, density=1001):
    x = np.linspace(-limit, limit, density)
    X, Y = np.meshgrid(x, x)
    grating = np.sin(2 * np.pi * (X * np.cos(theta) + Y * np.sin(theta)) * frequency)
    return grating

# return 2D Fourier transform of the input image
def get2DFourierTransform(img, shift=True):
    # fourier transform with shift for easier plotting
    FT = img
    if shift:
        FT = np.fft.ifftshift(img)
    FT = np.fft.fft2(FT)
    FT = np.fft.fftshift(FT)
    return FT

def getInverse2DFourierTransform(ft):
    ifft = np.fft.ifft2(ft)
    return ifft

# return a magnitude spectrum of the input Fourier Transform
def getMagnitudeSpectrum(FT, log=False):
    if log:
        return np.log(abs(FT))
    return abs(FT)

# return a phase spectrum of the input Fourier Transform
def getPhaseSpectrum(FT, log=False):
    if log:
        return np.log(np.angle(FT))
    return np.angle(FT)

# plots an image grid from an n x m numpy array of images with an optional n x m grid of corresponding titles
def plotImageGrid(grid, plot_size = 6, padding = 0.3, title = None, titles = False, fontsize = 7, y_factor = 1, x_factor = 1, cmap='gray'):
    grid_x, grid_y = grid.shape[0], grid.shape[1]
    plot_size, padding = plot_size, padding
    fig, axs = plt.subplots(grid_x, grid_y, figsize=(x_factor * plot_size, y_factor * plot_size), squeeze=False)
    fig.subplots_adjust(wspace=padding)

    fig.tight_layout()
    if title:
        st = fig.suptitle(title, fontsize="x-large")
        fig.subplots_adjust(top=1-1/(4*grid_x))
        st.set_y(0.99)
    for i in range(grid_x):
        for j in range(grid_y):
            img = grid[i][j]
            axs[i, j].imshow(img, cmap=cmap)
            if titles:
                axs[i, j].set_title(titles[i][j], fontsize=fontsize)

