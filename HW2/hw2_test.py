import hw2
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


class Homework2Test(unittest.TestCase):

    def test_build_A(self):
        pts1 = np.array([0, 600, 0, 0, 874, 0, 900, 500]).reshape(4, 2)
        pts2 = np.array([100, 120, 90, 10, 340, 9, 480, 140]).reshape(4, 2)
        expected = np.array([[0., 600., 1., 0., 0., 0., 0., -60000., -100.],
                             [0., 0., 0., 0., 600., 1., 0., -72000., -120.],
                             [0., 0., 1., 0., 0., 0., 0., 0., -90.],
                             [0., 0., 0., 0., 0., 1., 0., 0., -10.],
                             [874., 0., 1., 0., 0., 0., -297160., 0., -340.],
                             [0., 0., 0., 874., 0., 1., -7866., 0., -9.],
                             [900., 500., 1., 0., 0., 0., -
                                 432000., -240000., -480.],
                             [0., 0., 0., 900., 500., 1., -126000., -70000., -140.]])
        A = hw2.build_A(pts1, pts2)
        difference = np.linalg.norm(A - expected)
        self.assertEqual(expected.shape, A.shape)
        self.assertAlmostEqual(
            difference, 0.0, places=4, msg='The matrix returned for A was not as expected.')

    def test_compute_H(self):
        pts1 = np.array([0, 600, 0, 0, 874, 0, 900, 500]).reshape(4, 2)
        pts2 = np.array([100, 120, 90, 10, 340, 9, 480, 140]).reshape(4, 2)
        expected = np.array([[0.00205, -0.00032,  0.99382],
                             [-0.00004,  0.00142,  0.11042],
                             [-0., -0.00001,  0.01104]])
        A = hw2.compute_H(pts1, pts2)
        difference = np.linalg.norm(A - expected)
        self.assertEqual(expected.shape, A.shape)
        self.assertAlmostEqual(
            difference, 0.0, places=4, msg='The matrix returned for H was not as expected')

    def test_bilinear_interp(self):
        img = np.array([[0, 4, 10], [8, 23, 4], [16, 9, 7]], np.uint8)
        self.assertAlmostEqual(hw2.bilinear_interp(img, (1.7, 1.05)), 9.595)
        self.assertAlmostEqual(hw2.bilinear_interp(img, (1.5, 2)), 9)

    def test_apply_homography(self):
        H = np.array([[0.00205, -0.00032,  0.99382],
                      [-0.00004,  0.00142,  0.11042],
                      [-0., -0.00001,  0.01104]])
        pts = np.array([30, 100, 35, 60, 492, -234, 73, 501]).reshape(4, 2)
        expected = np.array([[101.9243,  25.02191],
                             [100.22701,  18.60345],
                             [155.25411, -18.05232],
                             [163.04312, 135.80763]])
        res = hw2.apply_homography(H, pts)
        difference = np.linalg.norm(res - expected)
        self.assertEqual(expected.shape, res.shape)
        self.assertAlmostEqual(
            difference, 0.0, places=4, msg='The points returned aftering applying the homography were not as expected')


if __name__ == '__main__':
    unittest.main()
