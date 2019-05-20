# Panorama

In this project you will take an ordered sequence of images, taken by rotating the camera around the vertical axis, along with manually-defined initial rough alignments, and stitch them into a single cylindrical panoramic image. You will accompilsh this by following the high-level process below:
* Calibrate your camera (already done for the test images).
* Reproject the input images into cylindrical coordinates (already done for the test images).
* Use the pyramid-based iterative Lucas Kanade algorithm to compute alignments between each image and the next one in the sequence.
* Blend the edges of each image with its neighbors both before and after in the sequence.
* Place the images at their proper positions in the panorama.
* Extra Credit: Correct for any vertical drift caused by imperfect camera rotation.

Once you have generated a panorama, you can view it with included `example_viewer` in the skeleton kit.


# Getting Started

## Environment Setup

We'll be using Python 3 for this assignment. To test your code on tux, you'll need to run:

```
pip3 install --user imageio
```

## Skeleton Code

Skeleton code has been linked below to get you started. Though it is not necessary, you are free to add functions as you see fit to complete the assignment. You are _not_, however, allowed to import additional libraries or submit separate code files. Everything you will need has been included in the skeleton. DO NOT change how the program reads parameters or loads the files. failure to follow these rule will result in a large loss of credit.

## Example

I've generated an example output with the images included in the skeleton kit using the following command:

```
python3 hw3.py example_inputs/files.txt example_output.png 
```

The output of this command was:
```
Refining displacements with Pyramid Iterative Lucas Kanade...
Image 0: [-260.    0.] -> [-240.23285      2.6782336]    0.1444 -> 0.1089
Image 1: [-255.    0.] -> [-260.5952       2.0931993]    0.1204 -> 0.1039
Image 2: [-270.    0.] -> [-278.91953    2.28174]    0.1536 -> 0.1389
Image 3: [-276.    0.] -> [-280.18338     4.463224]    0.1477 -> 0.1350
Image 4: [-268.    0.] -> [-278.13458      3.2441852]    0.1302 -> 0.1179
Image 5: [-296.    0.] -> [-308.70874      5.1885405]    0.1478 -> 0.1342
Image 6: [-306.    0.] -> [-318.7632      -0.7127918]    0.1157 -> 0.1079
Image 7: [-335.    0.] -> [-341.7655       4.5057926]    0.0847 -> 0.0696
Image 8: [-300.    0.] -> [-307.3074       0.8345972]    0.0862 -> 0.0754
Image 9: [-297.    0.] -> [-303.13068     5.694725]    0.0714 -> 0.0653
Image 10: [-340.    0.] -> [-329.35242      5.6397634]    0.1019 -> 0.0901
Image 11: [-316.    0.] -> [-322.0941       2.3241656]    0.1174 -> 0.1083
Image 12: [-368.    0.] -> [-380.24768     5.327328]    0.1653 -> 0.1531
Image 13: [-300.    0.] -> [-312.0026      3.009793]    0.1670 -> 0.1476
Image 14: [-240.    0.] -> [-264.93216     4.865892]    0.1783 -> 0.1429
Saving displacements to final_displacements.pkl
```

I've included the displacements generated in `example_displacements.pkl`

## Testing Correctness

I've provided a unit test for _some_ of the skeleton code that should help guide you in verifying the correctness of your program. To run it, execute:

```
python3 hw3_test.py -v
```

For more information about unit tests, see the [Python unittest documentation](https://docs.python.org/3/library/unittest.html).


# Submission

All of your code should be in a single file called `hw3.py`. Be sure to try to write something for each function. If your program errors on run you will loose many, if not all, points.

In addition to your code, you _must_ include the following items:

* A  `ReadMe.pdf` with the following items with a description of your experiments (e.g., changing parameters and showing their effects on the results, include lots of pictures) and a short discussion of what you struggled with, if anything. If you didn't complete the assignment or there is a serious bug in your code, indicate it here. If you believe your code is awesome, just say so. If you did extra credit, discuss it in this file as well.

Call your python file `hw3.py` and zip it, along with `ReadMe.pdf`, into an archive called `DREXELID_hw_3.zip` where `DREXELID` is your `abc123` alias. DO NOT include _any_ other files.

# Grading
All submissions will be graded by a program run on the [tux cluster](https://www.cs.drexel.edu/Account/Account.html). The grader is similar to the provided unit test, but will use different input and outputs. It is your responsibility to ensure your submission can be run on the tux cluster (if you follow the above instructions, it will!).

To avoid runtime errors:
* do no import any libraries (this is a requirement)
* do not rely on environment variables
* do not hard code any paths into your program

The assignment is worth 50 points and will be graded as follows:
* [20 pts]: lucas_kanade function
* [5 pts]: iterative_lucas_kanade function
* [5 pts]: gaussian_pyramid function
* [10 pts]: pyramid_lucas_kanade function
* [5 pts]: Proper mosaic function
* [5 pts]:  Include the report PDF described above. 

# Extra Credit Options

There are a large number of options for extra credit on this project:
* [10 pts]: Implement the warp_panorama function to correct any vertical drift (hint: it's just a homography!). I've included a `example_warped.png` inside the viewer directory that shows what it should look like.
* [10 pts]: Take a series of your own pictures and make a panorama. To do this, first calibrate your camera by printing out [this image](http://www.emgu.com/wiki/images/OpenCV_Chessboard.png), taking a bunch of photos, and run the included `calibrate_camera.py` script. Then, reproject the files using the included `make_cylindrical.py` script. You'll need to make a new file with initial alignments for the input to your program.
* [10 pts]: Implement Laplacian pyramid blending instead of linear blending.
* [15 pts]: Automatically compute initial rough alignments for the image pairs.
* [? pts]: Do something really interesting.

You must show the results of your code for each extra credit portion.
