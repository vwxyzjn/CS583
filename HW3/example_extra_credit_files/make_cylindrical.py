import sys

import numpy as np
from scipy.misc import *

def to_cylindrical(image, camera_params):
    F, k1, k2 = camera_params

    cyl_img_pts = np.mgrid[:image.shape[0], :image.shape[1]].transpose(1,2,0).astype(np.float32)
    x_cyl = cyl_img_pts[..., 1]
    y_cyl = cyl_img_pts[..., 0]

    # Convert from cylindrical image points x_cyl/y_cyl to cylindrical
    # coordinates theta/h.
    theta = (x_cyl - image.shape[1] / 2.) / F
    h = (y_cyl - image.shape[0] / 2.) / F

    # Convert from cylindrical coordinates theta/h to x/y/z coordinates on the
    # 3D cylinder.
    x_hat = np.sin(theta)
    y_hat = h
    z_hat = np.cos(theta)

    # Convert from cylinder x/y/z to normalized input image x/y coordinates.
    x = x_hat / z_hat
    y = y_hat / z_hat

    # Apply radial distortion correction to the normalized image x/y
    # coordinates.
    r2 = x*x + y*y
    df = 1 + k1*r2 + k2*r2**2.
    x *= df
    y *= df

    # Convert from normalized image x/y coordinates to actual x/y coordinates.
    xPrime = x * F + image.shape[1] / 2.
    yPrime = y * F + image.shape[0] / 2.

    # Look up pixels in the input image at the final coordinates (using
    # bilinear interpolation) to form the cylindrical image and return it.
    final_coords = np.dstack([yPrime[..., np.newaxis], xPrime[..., np.newaxis]])
    return bilinear_interp(image, final_coords)

if __name__ == "__main__":
    filenames = [l.strip().split()[0] for l in open(sys.argv[1]).readlines()]
    camera_params = np.loadtxt(sys.argv[2])

    for fn in filenames:
        ext = "." + fn.split(".")[-1]
        cyl_fn = fn.replace(ext, "_cylindrical" + ext)

        print("Processing '%s' -> '%s' ..." % (fn, cyl_fn))
        img = imread(fn).astype(np.float32) / 255.
        cyl = to_cylindrical(img, camera_params)
        imsave(cyl_fn, cyl)
