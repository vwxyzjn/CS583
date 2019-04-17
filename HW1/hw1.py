
"""
Imports we need.
Note: You may _NOT_ add any more imports than these.
"""
import argparse
import imageio
import logging
import numpy as np
from PIL import Image


def load_image(filename):
    """Loads the provided image file, and returns it as a numpy array."""
    im = Image.open(filename)
    return np.array(im)


def create_gaussian_kernel(size, sigma=1.0):
    """
    Creates a 2-dimensional, size x size gaussian kernel.
    It is normalized such that the sum over all values = 1. 

    Args:
        size (int):     The dimensionality of the kernel. It should be odd.
        sigma (float):  The sigma value to use 

    Returns:
        A size x size floating point ndarray whose values are sampled from the multivariate gaussian.

    See:
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution
        https://homepages.inf.ed.ac.uk/rbf/HIPR2/eqns/eqngaus2.gif
    """

    # Ensure the parameter passed is odd
    if size % 2 != 1:
        raise ValueError('The size of the kernel should not be even.')

    # TODO: Create a size by size ndarray of type float32
    h = np.zeros((size, size), dtype=np.float32)

    # TODO: Populate the values of the kernel. Note that the middle `pixel` should be x = 0 and y = 0.
    offset = size // 2
    for i in range(-offset, offset+1):
        for j in range(-offset, offset+1):
            h[i+offset, j+offset] = 1 / (2*np.pi*sigma**2) * np.exp(-(i**2 + j**2)/(2*sigma**2))

    
    # TODO:  Normalize the values such that the sum of the kernel = 1
    return (1 / h.sum()) * h


def convolve_pixel(img, kernel, i, j):
    """
    Convolves the provided kernel with the image at location i,j, and returns the result.
    If the kernel stretches beyond the border of the image, it returns the original pixel.

    Args:
        img:        A 2-dimensional ndarray input image.
        kernel:     A 2-dimensional kernel to convolve with the image.
        i (int):    The row location to do the convolution at.
        j (int):    The column location to process.

    Returns:
        The result of convolving the provided kernel with the image at location i, j.
    """

    # First let's validate the inputs are the shape we expect...
    if len(img.shape) != 2:
        raise ValueError(
            'Image argument to convolve_pixel should be one channel.')
    if len(kernel.shape) != 2:
        raise ValueError('The kernel should be two dimensional.')
    if kernel.shape[0] % 2 != 1 or kernel.shape[1] % 2 != 1:
        raise ValueError(
            'The size of the kernel should not be even, but got shape %s' % (str(kernel.shape)))

    # TODO: determine, using the kernel shape, the ith and jth locations to start at.
    offset = len(kernel) // 2
    
    # TODO: Check if the kernel stretches beyond the border of the image.
    if i-offset < 0 or j-offset < 0 or i+offset > len(img)-1 or j+offset > len(img[0])-1:
        return img[i, j]
        # TODO: if so, return the input pixel at that location.
    else:
        # TODO: perform the convolution.
        value = 0
        for u in range(-offset, offset+1):
            for v in range(-offset, offset+1):
                value += kernel[u+offset, v+offset] * img[i-u, j-v]
        return value


def convolve(img, kernel):
    """
    Convolves the provided kernel with the provided image and returns the results.

    Args:
        img:        A 2-dimensional ndarray input image.
        kernel:     A 2-dimensional kernel to convolve with the image.

    Returns:
        The result of convolving the provided kernel with the image at location i, j.
    """
    # TODO: Make a copy of the input image to save results
    result = np.zeros_like(img)

    # TODO: Populate each pixel in the input by calling convolve_pixel and return results.
    for i in range(len(result)):
        for j in range(len(result[0])):
            result[i,j] = convolve_pixel(img, kernel, i, j)
    return result


def split(img):
    """
    Splits a image (a height x width x 3 ndarray) into 3 ndarrays, 1 for each channel.

    Args:
        img:    A height x width x 3 channel ndarray.

    Returns:
        A 3-tuple of the r, g, and b channels.
    """
    if img.shape[2] != 3:
        raise ValueError('The split function requires a 3-channel input image')
    return img[:,:,0], img[:,:,1], img[:,:,2]
    # TODO: Implement me


def merge(r, g, b):
    """
    Merges three images (height x width ndarrays) into a 3-channel color image ndarrays.

    Args:
        r:    A height x width ndarray of red pixel values.
        g:    A height x width ndarray of green pixel values.
        b:    A height x width ndarray of blue pixel values.

    Returns:
        A height x width x 3 ndarray representing the color image.
    """
    # TODO: Implement me
    result = np.zeros(r.shape + (3,), dtype=np.uint8)
    result[:,:,0] = r
    result[:,:,1] = g
    result[:,:,2] = b
    return result


"""
The main function
"""
if __name__ == '__main__':
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Blurs an image using an isotropic Gaussian kernel.')
    parser.add_argument('input', type=str, help='The input image file to blur', default="example-input.jpg", nargs='?')
    parser.add_argument('output', type=str, help='Where to save the result', default="test.jpg", nargs='?')
    parser.add_argument('--sigma', type=float, default=2.0,
                        help='The standard deviation to use for the Guassian kernel')
    parser.add_argument('--k', type=int, default=5,
                        help='The size of the kernel.')

    args = parser.parse_args()

    # first load the input image
    logging.info('Loading input image %s' % (args.input))
    inputImage = load_image(args.input)

    # Split it into three channels
    logging.info('Splitting it into 3 channels')
    (r, g, b) = split(inputImage)

    # compute the gaussian kernel
    logging.info('Computing a gaussian kernel with size %d and sigma %f' %
                 (args.k, args.sigma))
    kernel = create_gaussian_kernel(args.k, args.sigma)

    # convolve it with each input channel
    logging.info('Convolving the first channel')
    r = convolve(r, kernel)
    logging.info('Convolving the second channel')
    g = convolve(g, kernel)
    logging.info('Convolving the third channel')
    b = convolve(b, kernel)

    # merge the channels back
    logging.info('Merging results')
    resultImage = merge(r, g, b)

    # save the result
    logging.info('Saving result to %s' % (args.output))
    imageio.imwrite(args.output, resultImage)
