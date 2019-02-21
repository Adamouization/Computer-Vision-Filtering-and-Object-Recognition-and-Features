# Computer-Vision-Coursework

## 2019 CM30080 Coursework: Filtering, Object Recognition & Features

## Introduction

In this lab we implemented algorithms for matching images (similar to the principle of playing Dobble). This will be split into three tasks, 1. Implementing 2D Image convolution, 2. Intensity-based template matching: where we will use correlation for object recognition, and 3. Feature-based template matching using SIFT: where we will implement the method
proposed by Lowe [1- 3 ] and perform matching using Sum of Squared Differences (SSD). 

A Training dataset of images (icons) as shown in Figure 1 and a Testing dataset (various combinations) as shown in Figure 2 will be provided. The icons from the Training dataset have to be recognised and matched in Test dataset provided.

## Task 1: Image convolution

A function which takes an image and a kernel as inputs, and computes the two-dimensional convolution of the two.

**Output 1 :** The function should output an image of the same size as the original. Areas outside of the image are treated as zero. The results to the in-built conv2 function of MATLAB. Tested with a range of kernels: blurs of different sizes and edge detectors in different directions.

## Task 2: Intensity-based template matching (worth 9 pts)

A function which takes an image from Test dataset and outputs the following:

**Output 2.1:** Detect objects in Test images, recognize to which class (from the 50 Training image classes) it belongs, and identify their scales and orientations. For visual demonstration the function opens a box around each detected object and indicate its class number according to the order of the Training images. This box is scaled and rotated according to the object’s scale and orientation.

Algorithm evaluated on all Test images with the overall False Positive (FP), True Positive (TP) rates and the average runtime reported.

**Creating scaled & rotated templates:** A Gaussian pyramid of appropriate scale levels for each Training image is created. This task implements appropriate Gaussian blurs (using own convolution algorithm) and subsampling of the training images through a hierarchy. After creating scaled templates, this set is appended by creating Gaussian pyramids for an appropriate number of orientations per class. This creates the overall scaled & rotated templates.

Choice of parameters: Gaussian kernel specs, initial/final scales, number of scale levels i.e. how many octaves, number of rotations, etc.

**Pre-processing:** For each (scaled and rotated) template the background is set to zero and normalized it (the mean equal is set to zero, and the Euclidean norm of the template is set to 1. For the Test data all backgrounds are set to zero.

**Intensity-based matching (related to output 2.1):** Slide each template image 푇 over the given test
image 퐼 (across x, y directions) and report the similarity score based on their correlations

### 푐표푟(푥,푦)=-푇(푖,푗)퐼(푖−푥,푗−푦)

1 , 2
This formula should be familiar to you. What does it indicate? According to that propose an efficient
implementation.
Define appropriate thresholds on this similarity measure, a proper non-maxima suppression strategy
and return the **output 2.1** in specified format.

Throughout, choose appropriate parameters (and report them) in order to minimize the False
Positive and False Negative rates.


## Task 3: Feature-based template matching using SIFT (worth 13 pts)

In this part you will implement a simple form of the scale-invariant feature transform (SIFT) method
described by Lowe [1-3].

First of all, write a function which creates a DoG pyramid from an image. This function will upgrade
your Gaussian pyramid implementation in the last task– note that unlike the previous section, each
level i.e. octave of the pyramid, will have multiple (for example s=3) Gaussians applied before
resampling. Next, look for local extrema in this DoG space. Each interest point is then a point given
by (푥,휎), where 휎 represents the scale of the feature.
**Output 3.1 ( 4 pts):** Visualise your results on the original image as boxes where, each box is scaled by
the size of the feature it represents. Demonstrate your algorithm on a range of images from the
Training and Test datasets.

**Output 3.2 (4 pts):** SIFT interest points can be refined to improve low-quality results. For this part,
filter your results using the methods described in the references [1-3], including low contrast and
edge removal. Indicate details of these updates in your report. Also, using the technique described in
the papers (i.e. histogram of the weighted gradient directions), add orientations to each feature, so
each is now represented by (푥,휎,휃), where 휃 denotes the orientation. Add details about this step in
your report. Demonstrate your algorithm (now scaled and rotated boxes corresponding to each
feature) on a range of images from the Training and Test datasets.

**Output 3.2 ( 5 pts):** For this part, demonstrate the ability to match your features between images e.g.
lines connecting matches. Perform a feature-based template matching in order to detect and
recognize objects in Test dataset (for demonstration create labelled boxes around Test objects).
Evaluate your algorithm on all test images and report the overall False Positive, True Positive rates
and the average runtime. Show and explain cases where this scheme finds difficulty to perform
correctly. Give insight about complexity (memory and runtime) of your algorithm in terms of the
problem size, how does the complexity scale? Again, provide details of your approach, choice of
parameters/thresholds in your report.

Doing this requires two things: a feature descriptor, and a method for matching.

- A simple feature descriptor could just be a neighbourhood of pixel values around the
    interest point. For a more advanced feature descriptor, you could use SIFT descriptors.
- A simple method for matching is to check the sum squared distance (SSD) of the descriptors,
    and filter-out bad matches using the nearest neighbour ratio.
Finally find strong correspondences between a given Test image and each of the Training image
classes, and use this comparison for object detection and recognition.

## References

Lowe published two papers that describe SIFT. The one in 1999 gives basic version of the technique,
but also includes an application to object recognition. The later paper in 2004, gives more details
about the technique. A PowerPoint presentation from Lowe [3] is also added which highlights the
key steps. Learning to find the relevant section of the paper you are reading is an important skill!

[1] Lowe, David G. "Object recognition from local scale-invariant features." Computer vision, 1999.
The proceedings of the seventh IEEE international conference on. Vol. 2. IEEE, 1999.
[2] Lowe, David G. "Distinctive image features from scale-invariant keypoints." International journal
of computer vision, 60.2 (2004): 91-110.
[3] https://people.cs.pitt.edu/~kovashka/cs3710_sp15/features_yan.pdf


