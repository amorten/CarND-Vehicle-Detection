## Writeup


---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[single_img_features]: ./output_images/single_img_features.jpg
[simple_windowing]: ./output_images/simple_windowing.jpg
[find_cars]: ./output_images/find_cars.jpg
[find_cars_many_scales]: ./output_images/find_cars_many_scales.jpg
[thresholded]: ./output_images/thresholded.jpg
[testing_pipeline]: ./output_images/thresholded.jpg
[pipeline_no_history]: ./output_images/pipeline_no_history.jpg
[pipeline_history]: ./output_images/pipeline_history.jpg

[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4
[notebook]: ./vehicle_detection.ipynb

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

In cell #4 of the [notebook] I define a function `printlog()` that not only prints to the screen but also writes the same text to the file [vehicle_detection.log](vehicle_detection.log). I use this function whenever printing training results to the screen. This way I keep a log of every hyperparameter combo tried and the resulting validation accuracy.

Cell #5 of the [notebook] has code originally used to load the data provided by the class. However, because the images of cars in the GTI dataset were grouped by car (approximately 10 images of each car), random shuffling of the data produced training and validation datasets that were very similar. I therefore split the data manually into 80% training and 20% validation, and put the split data into `train` and `test` folders. 

Then in cell #6 of the [notebook] I load the split data from those folders. So far, when I say "load the data" I mean to say that I construct lists of image file names based on the dataset.  The images will only later be loaded one-by-one when needed later.

In cell #7 of the [notebook] I load image names from the additional optional Udacity dataset. There were some problems with this dataset.  The heading names in the .csv file for the dataset were wrong: the `xmax` data was stored in the `ymin` column, while the `ymin` data was stored in the `xmax` column. Also, some of the bounding boxes were defined with width zero, so those had to be ignored. The bounding boxes were not square, so I cropped them to produce the largest square possible positioned randomly within the original bounding box. In total, I added 4464 training vehicle images from the the new dataset.

In cell #9  of the [notebook] I split the new dataset into training and validation and then merge them with the training and validation subsets of the old dataset.

In cells #11-24 I define helper functions for cells #25 and beyond. These helper functions are slightly modified versions of the functions defined in the course. I will explain some of the helper functions below as needed.



### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.


In cell #25 I calculate features for a random car and noncar training image. The features are calculated using the function `single_img_features()` defined in cell #21. 

Three kinds of features are extracted: spatial, histogram, and histogram of oriented gradients (HOG). The spatial features are just the flattend pixel color data (single or all color channels). The histogram features are simply histograms of the values in each color channel. HOG features are basically (weighted) histograms of the angles of gradients within each "cell" of an image, where the size of each cell is selected to be roughly the smallest relevant feature in the image.


The function `single_img_features()` is just one example of a helper function that extracts all three types of features from image data. Other helper functions that perform the same taks in incrementally improved ways will be discussed below. What they all have in common is that they all take in the following inputs:  which colorspace to use and which channel(s) to use; spatial size to use for spatial features (resizing down if necessary); number of bins for histogram features; orientation, number of pixels per cell, and number of cells per block for HOG features; and booleans which decide whether or not to include each type of feature (spatial, histogram, or HOG) in the resulting flattened feature array.

Continuing with the discussion of cell #25, the example feature extraction performed in cell #25 uses the first channel of the RGB colorspace, 9 HOG orientations, 8 pixels per cell, 2 cells per block, a spatial feature size of 16x16, and 16 histogram bins. 

While I will later choose to use all three channels of the `HSV` colorspace, here I just try `RGB` as an example. I ultimately chose the `HSV` color space because it gave the highest validation accuracy. On the raw data set, `YCrCb` was slightly better, but after altering the data set as described earlier, `HSV` had better accuracy (and was able to detect the white car, unlike `YCrCb`). 
I chose 9 orientations because several papers recommend that as a rough upper limit for the number of orientations.  I selected `pixels_per_cell = 8` by trying different cell sizes and seeing which captured the most useful features of cars vs noncars. The value of `cells_per_block` was set to 2 so that there would be at least some regional normalization.  I tried various spatial feature sizes, selecting the smallest that did not significantly reduce the validation accuracy. The number of histogram bins was chosen similarly.

Here is an example of HOG features calculated with the above parameters on a random car and noncar image:

![image of HOG features of car and noncar][single_img_features]

In cell #27 of the [notebook] I extract features from the training and test data. Instead of using `single_img_features()` I use `extract_features()` defined in cell #15. Extract features calculates the three types of features for a list of images rather than a single image. It outputs an array of features arrays (one feature array per image). By feeding in a list of training images and then a list of validation images, the resulting outputs can be used for training a classifier. Note that here I use the `HSV` color channel with all three channels. This channel was chosen because it gave the best validation accuracy.



#### 2. Explain how you settled on your final choice of HOG parameters.

I varied each HOG parameter as described in the previous section.  I varied each one while keeping the others constant and chose the hyperparameter value that maximized the validation accuracy each time. If multiple hyperparameter values gave the same validation accuracy, then I chose the value that minimized the prediction running time (e.g. fewer spatial histograms didn't change the validation accuracy, so I used fewer spatial histograms).

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In cell #28 I train a linear support vector machine. 
I used all three feature types, because that choice maximized the validation accuracy (no surprise).
The features were normalized to have unit variance and zero mean. Unlike in class, I only used the training data to calculate the normaliziation, so that the information about the test data would not leak into the training process.

Each time I train, all of the feature-extraction hyperparameters described above are output to a [log file](vehicle_training.log) along with the SVM hyperparameters and validation accuracy. Then I can look through the history and see which hyperparameters worked best. The linear SVM has one parameter, `C`, which I varied by several orders of magnitude, finding it had little to no effect on the validation accuracy, so I left it at `C=1`, which minimized predition time.

In cell #29 I attempted to do a grid search of hyperparameters and to try both linear and rbf kernal functions.  However, it was very slow, even to do the linear kernal (because LinearSVC is apparently much faster than SVC with a linear kernel). I therefore relied on manually testing a few hyperparameters and focusing on other ways to improve the vehicle detection algorithm.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In cell #31 I check how well the predictions work on the six test images of highway driving. To define a set of windows that cover roughly the bottom half of the test image (where the road is), I use the function `slide_window()` defined in cell #19. I used `slide_window()` to create windows of size `(96,96)` that overlap 50% in both the x- and y-directions. The set of windows and fitted support vector machine are passed to the function `search_windows()`, which is defined in cell #22. The `search_windows()` function uses the SVM to predict whether or not there is a car in each window and outputs the windows where vehicles have been detected. The function `draw_boxes()` defined in cell #20 is then used to drow the vehicle-detected windows onto the original image. The following image shows the vehicle bounding boxes detected by this method for the six test images:

![Vehicles found by simply windowing method][simple_windowing]


In cell #32 I basically redo the previous cell using  the function `find_cars()`, which is significantly faster. The function `find_cars()` is defined in cell #16. Basically, `search_windows()` calculates HOG features for every single window within the image, while `find_cars()` calculates HOG features just once for each image. The multidimensional array of HOG features for the full image is used to obtain HOG features within each of the windows without significant additional computation. Also, the function `find_cars()` produces a heatmap, which is basically the sum of the number of windows that detect a vehicle at each pixel (which can be greater than one, because windows overlap).  Another difference is that instead of specifying a window size, `find_cars()` requires that we specify a window `scale` (with `scale=1` corresponding to a 64x64 window). The following image shows the vehicle bounding boxes and heatmaps calculated by `find_cars()` with `scale=1` and a window overlap of 75% for a test image:

![Vehicles found by find_cars() method][find_cars]


In cell #33 I redo the previous cell using the function `find_cars_many_scales()` defined in cell #18. Basically, the function `find_cars()` allows searching by a single window size (a single `scale`) whereas `find_cars_many_scales()` allows searching with multiple window sizes. It also outputs a heatmap.
The following image shows the vehicle bounding boxes and heatmaps calculated by `find_cars_many_scales()` with `scales=[1.2,1.5,1.8]` and a window overlap of 75% for the same test image:

![Vehicles found by find_cars_many_scales() method][find_cars_many_scales]


In cell #34 I use `find_cars_many_scales()` again, but instead of using the bounding boxes output by that function, I use the heatmap to produce higher-confidence bounding boxes. I select only pixels with heatmap values above a given theshold. In this way false positives can be reduced, since they often are detected by a single window. Contiguous regions of selected pixels are found using the `labels` function, which is used as an input to `draw_labeled_bboxes()` to draw rectangular bounding boxes around the labelled regions. The following image shows the vehicle bounding boxes and heatmaps calculated by `find_cars_many_scales()` for  the same test image with `scales=[1.2,1.5,1.8]`, a window overlap of 75%, and the heatmap threshold set to 1 (meaning a >1 heatmap value is needed):

![Vehicles found by thresholded heatmap][thresholded]

I tried different choices for `scales`. For the smallest scale, I tried 0.5, 0.8, 1, 1.2 and 1.5.  The smaller values would detect the cars farther away but also detect more false positives.  The larger values had few false positives, but could not detect small cars. As a compromise, I chose 1.2 as the smallest scale, because it could still detect the small white car in the distance. Scales larger than 1.8 did not get detected often. I choose three scales rather than two, in hopes that multiple detections of a car at multiple scales would counter false positives, which might be more specific to scale (particularly small scales). I chose to have the windows overlap 75% so that well-detected vehicls would be covered by at least 8 (or so) windows (assuming three window sizes).

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

In cell #35 I define the image processing pipeline that will be applied to the project video. The process in the previous cell - using a thresholded heatmap and multiple window scales -  is used by the pipeline. The pipeline takes the heatmap calculation one step further by adding up the heatmaps over the previous `n_history` video frames before applying a threshold. In this way, only positive vehicle detections that persist over multiple frames are counted as vehicle detections. This removes some of the false positives that do not stay in the same place on the road relative to the camera mounted on the car. Other cars driving at roughly the same speed as us will have summed heatmaps with much larger values.

In cell #36 I test the pipeline on the test images just to check if it is working properly. Here is the output:

![testing the pipeline][testing_pipeline]

Testing the pipeline on the test images above doesn't allow for any heatmap history to remove false positives, but it shows that everything else is working ok.


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

In the final cell I use the pipeline to produce the final output video!

Here's a [link to my video result](./video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. 

I then summed the heatmap over the previous `n_history` frames. The treshold was applied after the sum of heatmaps. For `n_history` I tried every number from 1 to 20, while also varying the threshold between 0 and 10.  I found that `n_history=15` worked well to reduce some of the false positives, although it fragmented some of the car detections into multiple, smaller bounding boxes. For that value of `n_history`, a threshold of 1.5 (average for each of the 10 frames) was a good compromise between detecting cars and removing false positives.

I also reduced false positives by modifing the original training set as described earlier in this writeup. When I divided the original data set so that the training and validation car images were truly independent, then the validation accuracy dropped from 0.999 to around 9.883. Even though the validation accuracy dropped, **the number of false positives in the driving video was reduced to practically zero**.

However, I could not detect the white car, so I added more car images from the optional Udacity data set.  when I did that, I was able to detect the white car, but validation accuracy dropped to 0.9571 and the number of false positives rose dramatically. I tried varying model hyperparameters again no little avail.


I then reduced some of the false positives by using hard negative mining. That is, I added all of the false positives in the test data set back into the training data set.  This helped a little.  The hard negative mining is implemented in cell #30 of the [notebook].

So, in the end I had to choose between a model with no false positives but no white car detection or a model with white car detection but many false positives. I chose the model that detected both of the cars and then tried to remove the false positives with heatmaps, which were somewhate successful. 

Here are some images that demonstrate the pipeline... 

The first set of images uses the pipeline with `n_history` set to zero. In other words, there is no heatmap history being kept. There are multiple false positives:


![pipeline with no history][pipeline_no_history]

The next set of images uses the pipeline with `n_history` set to 10. Thus, the heatmap is cumulative over multiple frames. 
There are fewer false positives:

![pipeline with history][pipeline_history]

**Note: when investigating the above images, it may seem that the white car is not detected. In fact, it is being detected by a single window, but the threshold was set at 0.7 in this case. That means that an average number of 0.7 detections over 10 frames was require. So, the heatmap averaging will not detect the white car until after about 7 frames. In fact, you can see it begins to be detected in the sixth frame. In a longer running video with prior history, the white car would be detected earlier.**

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

At first I trained the linear SVM using the YCrCb colorspace. I got many false positives despite a very high validation accuracy.  I was concerned about overfitting, so I manually split the training and validation data sets so that the same car did not appear in both sets (each car appeared in approximately 10 consecutive images in the original data set). Retraining the model resulted in lower validation error, but very few false positives. Unfortuneatly, the white car could not be detected.

So I added images from the Udacity data set. With the new images, validation accuracy dropped significantly, but the what car was detected. Unfortuneatly, this came with many false positives a well. 
So I took the false positives and added them back to the training data. It did help somewhat.


I think the problem with the Udacity data is that the car images are quite dark and the vast majority look quite different from the cars in our video. I converted all of the Udacity car bounding boxes to squares using cropping. Perhaps that removed too much of the car-like features from some of the photos.


Then I went back and tried HSV color space.  YCrCb and HSV had given similar validation accuracies on the original data set, but I noticed that on the newly split train/validation data the HSV color space gave higher validation accuracy of 0.996 after I removed the Udacity data set, which was causing problems with giving too many false positives.


My pipeline could fail badly in other types of driving conditions.  I looked through the training images provided with the project and noticed that the background images were all quite similar to the background in the video. The sample cars also appear cleanly (without obstruction) and with similar lighting bright as in the video.  So, it seems that the training data were specifically designed for our test video, which suggests the model will not generalize well. Also, my attempts to set heatmap history length and thresholds were very finetuned to the project video. The same parameters liekly won't work under different driving conditions.

Because of my use of a long heatmap history, cars with speeds very different from my own will not be detected well. That doesn't happen in the project video, but it does sometimes happen in reality, which is a situation that can be quite dangerous.

I could make the vehicle detection more robust in several ways. First, I definitely need more data with cars and vehicles in different kinds of lighting. Second, I should probably try other classifiers, perhaps combining predictions from multiple classifiers to make a final prediction (or just pick the best classifier and go with it). There may be other types of useful features to extract that I have yet learn.


