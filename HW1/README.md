# Image Processing
The goal of this project is to familiarize you with image processing in python with numpy and get experience implementing low-level image filtering techniques. This homework will have relatively less depth than the other assignments; the goal here is to familiarize you with python, image processing, and the assignment expectations of the course.

# Goals

For this assignment, you will write a program that applies Gaussian blur to image. Specifically, the program will:

1. Load an image file provided on the command line, and decompress it into a numpy array.
2. Split the input image into 3 channels (R, G, B)
3. Compute a two-dimensional isotropic Gaussian kernel.
4. Convolve the Gaussian kernel with each channel of the image.
5. Save the result.

# Getting Started

## Python
If you do not have experience programming with Python or numpy, I recommend you read the following tutorials:

* [A Byte of Python](https://python.swaroopch.com/) for learning python
* [Scipy's Quickstart](https://docs.scipy.org/doc/numpy/user/quickstart.html) for familiarizing yourself with numpy.

The Python IDE I recommend is [Visual Studio Code](https://code.visualstudio.com/). If you are comfortable with another, stick with it.

## Environment Setup

We'll be using Python 3 for this assignment. To test your code on tux, you'll need to run:

```
pip3 install --user imageio
```

## Skeleton Code

Skeleton code has been linked below to get you started. Though it is not necessary, you are free to add functions as you see fit to complete the assignment. You are _not_, however, allowed to import additional libraries or submit separate code files. Everything you will need has been included in the skeleton. DO NOT change how the program reads parameters or loads the files. failure to follow these rule will result in a large loss of credit.

## Example

I've generated an example output with the image included in the skeleton kit using the following command:

```
python3 hw1.py --k 5 --sigma 2 example-input.jpg example-output.jpg 
```

## Testing Correctness

I've provided a unit test for the skeleton code that should help guide you in verifying the correctness of your program. To run it, execute:

```
python3 hw1_test.py -v
```

You should see the following output:
```
test_convolve (__main__.Homework1Test) ... ok
test_convolve_pixel (__main__.Homework1Test) ... ok
test_create_gaussian_kernel (__main__.Homework1Test) ... ok
test_merge_image (__main__.Homework1Test) ... ok
test_split_image (__main__.Homework1Test) ... ok

----------------------------------------------------------------------
Ran 5 tests in 0.002s

OK
```

You may want to only test one of the function. You can do this by specifying the test name on the command line. For example, if you want to test the `convolve_pixel` function, run:

```
python3 hw1_test.py Homework1Test.test_convolve_pixel -v
```

For more information about unit tests, see the [Python unittest documentation](https://docs.python.org/3/library/unittest.html).


# Submission

All of your code should be in a single file called `hw1.py`. Be sure to try to write something for each function. If your program errors on run you will loose many, if not all, points.

In addition to your code, you _must_ include the following items:

* Two examples of your code running on images you took yourself. You should name the input and output `example_1.input.jpg` and  `example_1.output.jpg`, respectively, for the first image and change the number for subsequent images. 
* A  `ReadMe.pdf` with a description of your experiments (e.g., changing parameters and showing their effects on the results, include lots of pictures) and a short discussion of what you struggled with, if anything. If you didn't complete the assignment or there is a serious bug in your code, indicate it here. If you believe your code is awesome, just say so. If you did extra credit, discuss it in this file as well.

Call your python file `hw1.py` and zip it, along with your example images and `ReadMe.pdf`, into an archive called `DREXELID_hw_1.zip` where `DREXELID` is your `abc123` alias. DO NOT include _any_ other files.

# Grading
All submissions will be graded by a program run on the [tux cluster](https://www.cs.drexel.edu/Account/Account.html). The grader is similar to the provided unit test, but will use different input and outputs. It is your responsibility to ensure your submission can be run on the tux cluster (if you follow the above instructions, it will!).

To avoid runtime errors:
* do no import any libraries (this is a requirement)
* do not rely on environment variables
* do not hard code any paths into your program

The assignment is worth 50 points and will be graded as follows:
* 10 points for following submission directions.
* 10 points for correctly computing the gaussian kernel.
* 15 points for correctly implementing convolution.
* 10 points for correct split & merge functions.
* 5 points for including two examples on images you took yourself.

Extra Credit Options:
(Note: If you implement these, do so in a seperate file called `hw1_extra_credit.py` so the grading program doesn't get confused.)
* 5 points: make it run faster using seperable kernels and leveraging numpy's built in optimizations.
* 5 points: make it read and write images with any number of channels (i.e., 1 for grayscale and 4 for images with transparency)
