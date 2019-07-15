# Computer-Vision-Coursework

## 1) Introduction

For this project, the programming language of choice was Python. It was chosen over MATLAB due to the availability of functions similar to MATLAB's functions through libraries, and due to the syntax and ﬂexibility of the language. With libraries such as OpenCV for image manipulations, Numpy and SciPy for more advanced mathematical functions including array manipulations, and MatPlotLib for plotting data, Python has all the tools for the task at hand.

## 2) Task 1: Image Convolution

For the image convolution task, a function to perform convolution on a grey image was written. This function takes an image to operate on and a kernel to apply on the image. In order to preserve image size, extra padding is added on the edges of the original image. The function can be found in Listing A.

Multiple ﬁlters are used to test the convolution function, including a Gaussian kernel, a sharpen kernel, a horizontal edge detector (Sobel ﬁlter) and an identity matrix. To conﬁrm that the results are correct, the resulting ﬁltered image compared to a library function. SciPy's "convolve" function was used to carry out this task as it is the equivalent of MATLAB's "conv2" function. The comparison is done by subtracting the function's result from the library's result and checking that all the values are equal to 0. See Listing B for the ﬁlters used and the test.

## 3) Task 2: Intensity-Based Template Matching

### 3.1) Pre-Processing

Multiple steps are followed to pre-process the training dataset. Firs, The background is set to 0 for each image in the training dataset. This is done by looking for white pixels (value 255) and replacing them with black pixels (value 0), as depicted in Listing C line 3. Next, each rotated and scaled template in the pyramid is then normalised using OpenCV's "normalize" function, which can be found in Listing C line 10. Finally, images are processed in RGB (3 channels) rather than being converting to grey scale (1 channel), therefore avoiding the loss of information and accuracy.

### 3.2) Gaussian Pyramid Generation

A Gaussian Pyramid is generated for each image in the training dataset. The scaling and rotation values for the pyramids were chosen by manually inspecting the training images to ﬁnd out what angles the images were rotated to and by how much the images were scaled down. Therefore, scales of 50%, 25%, 12.5% and 6.25% were chosen, along with rotations ranging from 0 ◦ to 330 ◦ with steps of 30 ◦ for each scale. A brief overview of the training function used to generate the templates can be found in Listing D.

The rotation of the templates was achieved through SciPy's "rotate" function (Listing D, line 22), while the scaling down was done manually with a custom subsampling function that recursively scales the image down by half by subsampling one pixel every two pixels (see Listing E). The subsampling blurring uses a Gaussian Filter with a size of 5x5 and a standard deviation σ = 15.

Each template is saved in separate binary ".dat" ﬁles for quicker I/O operations using Python's "pickle" library for object serialisation conversion in byte streams (see Listing D, line 26).

## 3.3) Output

#### 3.3.1) Testing

For the testing phase, each template previously stored in binary ﬁles is loaded in memory and is slid over one of the images from the testing dataset. All the templates for a single class are used to calculate the correlation score using Equation 1. The template with the highest score for a class is kept as the best match for that class, meaning there are 50 potential matching templates. Each template is then ﬁltered based on its correlation score: it is kept if it is higher than a threshold, set at 0.5. This threshold is set empirically by testing diﬀerent values until one that gave the most matches is found.

cor(x,y) = T(i,j) * I(i+x,j+y)

#### 3.3.2) Box Drawing

The ﬁnal step in producing the output is to draw a box around the detected objects that passed the threshold. The scaling and the rotation of the selected templates are used to draw the rectangle at the correct scale and at the correct rotation (see Listing G). The class name is drawn along with the box rather than class number to improve the output's clarity.

A rotated rectangle needs 4 pairs of coordinates to be drawn. Obtaining one pair will make it trivial to ﬁnd the others. So, the real objective is to ﬁnd the x and y shift from the one of the hypothetical straight square. Let us denote the x shift by ∆x. Starting from a system of equation deﬁning that the height/width of the original image is h = ∆x + ∆y, that sin(α) ∗ n = ∆y and that n 2 = ∆x 2 + ∆y 2 (where n is the length of the newly rotated and scaled square). So, resulting from the above, we can ﬁnd ∆x (as shown in Equation 2), thus all the corners of the rotated square: (sin(α) = k)

