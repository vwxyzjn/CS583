import sys

import cv2
import numpy as np

def calibrate(images, board_dim = (5,8), debug=False):
    images = np.asarray(images)
    found = []
    image_corners = []
    for count, image in enumerate(images):
        print("Finding chessboard...")
        f, ic = cv2.findChessboardCorners(image, board_dim)
        found.append(f)
        image_corners.append(ic)

        if f:
            print("Found board in image %d." % count)
            if debug:
                cv2.drawChessboardCorners(image, board_dim, np.squeeze(ic), f)
                cv2.imshow("Detected Chessboard", image)
                cv2.waitKey()
        else:
            print("NO BOARD in image %d." % count)

    found = np.asarray(found, dtype=np.bool)
    image_corners = [np.squeeze(crn) for crn in np.array(image_corners)[found]]
    images = images[found]

    crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 30, 0.1)
    for img, corners, count in zip(images, image_corners, range(len(images))):
        cv2.cornerSubPix(img.mean(2).astype(np.uint8), corners, (5,5), (-1,-1), crit)
        print("Refined corners in image %d." % count)

    world_corners = np.zeros((np.prod(board_dim), 3), np.float32)
    world_corners[:,:2] = np.indices(board_dim).T.reshape(-1, 2)
    world_corners = [world_corners] * len(image_corners)

    print("Calibrating...")
    rms, K, distortion, rvecs, tvecs = cv2.calibrateCamera(world_corners, image_corners, images[0].shape[:2][::-1], None, None, flags=cv2.CALIB_ZERO_TANGENT_DIST)
    return K, np.squeeze(distortion), rms

if __name__ == "__main__":
    board_dim = (int(sys.argv[1]), int(sys.argv[2]))
    images = [cv2.imread(fn) for fn in sys.argv[3:]]

    K, dist, rms = calibrate(images, board_dim)

    print("Camera Matrix (K):")
    print(K)
    print("k1, k2:", dist[0], dist[1])
    print("RMS error:", rms)
