# ECE 661 Computer Vision Course
This repository is created for the ECE 661 Computer Vision course.
These comprise of the homeworks that was done as part of the course.
All the codes are in python. And the reports are in LaTeX.

Below are some short descriptions of the homework with some sample images.

---
---

### Short Descriptions of Homeworks:

#### Homework 1:
Review of basic Homogeneous coordinates.

[hw1_problem_statement](homework1/PROBLEM_STATEMENT)

[hw1_report](homework1/REPORT)

---

#### Homework 2:
Estimate Homographies between images and use them to transform images and map images from one domain to a different domain.

[hw2_problem_statement](homework2/PROBLEM_STATEMENT)

[hw2_report](homework2/REPORT)

**Sample Input:**

![](homework2/extra_files/1_tiny.jpg)   ![](homework2/extra_files/3_tiny.jpg)

**Sample Output:**

![](homework2/extra_files/2_tiny.png)

---

#### Homework 3:
Eliminate the Affine and Projective distortions from images using point-to-point correspondence and Vanishing Line method.

[hw3_problem_statement](homework3/PROBLEM_STATEMENT)

[hw3_report](homework3/REPORT)

**Sample Input:**

![](homework3/extra_files/1_tiny.jpg)

**Sample Output:**

![](homework3/extra_files/2_tiny.png)

---

#### Homework 4:
Find the SIFT or SURF or Harris Corners interest points from two images of the same scene taken from two different viewpoints, and establish the correspondence between the interest points.

[hw4_problem_statement](homework4/PROBLEM_STATEMENT)

[hw4_report](homework4/REPORT)

**Sample Input:**

![](homework4/extra_files/1_tiny.jpg)   ![](homework4/extra_files/2_tiny.jpg)

**Sample Output:**

![](homework4/extra_files/3_tiny.png)

---

#### Homework 5:
Create an initial estimate of homography using Linear Least Squares Minimization and RANSAC algorithm. Then use this initial estimate to calculate Homography automatically by a Non-linear Least Squares method like Levenberg-Marquardt method. And finally use this to stitch a group of images together.

[hw5_problem_statement](homework5/PROBLEM_STATEMENT)

[hw5_report](homework5/REPORT)

**Sample Input:**

![](homework5/extra_files/1_tiny.jpg)   ![](homework5/extra_files/2_tiny.jpg)   ![](homework5/extra_files/3_tiny.jpg)   ![](homework5/extra_files/4_tiny.jpg)   ![](homework5/extra_files/5_tiny.jpg)

**Sample Output:**

![](homework5/extra_files/6_tiny.png)

---

#### Homework 6:
Implement Otsu's Algorithm and use RGB and Texture-based Segmention to separate out foreground objects from background of the given images. Also, extract the contours of the segmented foreground.

[hw6_problem_statement](homework6/PROBLEM_STATEMENT)

[hw6_report](homework6/REPORT)

**Sample Input:**

![](homework6/extra_files/1_tiny.jpg)

**Sample Output:**

![](homework6/extra_files/2_tiny.png)   ![](homework6/extra_files/3_tiny.png)

---

#### Homework 7:
Implement a classification algorithm using Local Binary Pattern features and K-Nearest Neighbor Classifier on 5 classes of images {Beach, Building, Car, Mountain, Tree}.

[hw7_problem_statement](homework7/PROBLEM_STATEMENT)

[hw7_report](homework7/REPORT)

**Sample Input:**

![](homework7/extra_files/beach_tiny.jpg)   ![](homework7/extra_files/building_tiny.jpg)   ![](homework7/extra_files/car_tiny.jpg)   ![](homework7/extra_files/mountain_tiny.jpg)   ![](homework7/extra_files/tree_tiny.jpg)

**Sample Output:**

![](homework7/extra_files/beach_pred_tiny.png)   ![](homework7/extra_files/building_pred_tiny.png)   ![](homework7/extra_files/car_pred_tiny.png)   ![](homework7/extra_files/mountain_pred_tiny.png)   ![](homework7/extra_files/tree_pred_tiny.png)

---

#### Homework 8:

Implement Zhang's Algorithm to find the Intrinsic and Extrinsic camera calibration parameters using checker-board patterns. Extract corners from the checker-board patterns, use them as to estimate the parameters and then map world points onto the pattern using those parameters.

[hw8_problem_statement](homework8/PROBLEM_STATEMENT)

[hw8_report](homework8/REPORT)

**Sample Input:**

![](homework8/extra_files/Pic_1_tiny.jpg)

**Sample Output:**

![](homework8/extra_files/Pic_1_lines_tiny.jpg)    ![](homework8/extra_files/Pic_1_corners_tiny.jpg)    ![](homework8/extra_files/Pic_1_mapped_tiny.jpg)

---

#### Homework 9:

The focus of this assignment is projective reconstruction of a scene using stereo images. What we obtain from projective reconstruction might appear distorted when compared to the actual scene.

[hw9_problem_statement](homework9/PROBLEM_STATEMENT)

[hw9_report](homework9/REPORT)

**Sample Input:**

![](homework9/extra_files/3.jpg)   ![](homework9/extra_files/4.jpg)

**Sample Output:**

![](homework9/extra_files/6.jpg)

![](homework9/extra_files/5_tiny.jpg)

---

#### Homework 10 Part 1:

The first part of the homework is about face recognition with PCA and LDA for dimensionality reduction and KNN for classification.

[hw10_problem_statement](homework10_part1/PROBLEM_STATEMENT)

[hw10_report](homework10_part1/REPORT)

**Sample Input:**

The actual images used for this homework is not uploaded as they were from a non-public dataset, but the feature vectors created from them are shared in this repository.

Total number of **Training** Samples:   **630** images, **30** classes, **21** image in each class. Size of each image is **128 x 128** pixels.

Total number of **Testing** Samples:   **630** images, **30** classes, **21** image in each class. Size of each image is **128 x 128** pixels.

The 30 classes were the faces of 30 different persons. Each class had 21 images of the front face of the person, taken under different lighting conditions.

**Sample Output:**

![](homework10_part1/extra_files/1_tiny.png)

---

#### Homework 10 Part 2:

The second part of the homework is about classifying images of cars from non-car images using HAAR cascade classifier created using ADABOOST algorithm.
The actual images used for this homework is not uploaded as they were from a non-public dataset, but the feature vectors created from them are shared in this repository.

[hw10_problem_statement](homework10_part2/PROBLEM_STATEMENT)

[hw10_report](homework10_part2/REPORT)

**Sample Input:**

The actual images used for this homework is not uploaded as they were from a non-public dataset, but the feature vectors created from them are shared in this repository.

Total number of **Training** Samples:   **710** positive and **1758** negative images, **2** classes (cars and non-cars). Size of each image is **20 x 40** pixels. 

Total number of **Testing** Samples:   **178** positive and **440** negative images, **2** classes (cars and non-cars). Size of each image is **20 x 40** pixels.

**Sample Output:**

![](homework10_part2/extra_files/1_tiny.png)    ![](homework10_part2/extra_files/2_tiny.png)

---





