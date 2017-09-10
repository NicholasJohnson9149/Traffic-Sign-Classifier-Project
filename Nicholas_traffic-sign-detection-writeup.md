#Writeup#
**Traffic Sign Recognition**
**By Nicholas Johnson**
**Build a Traffic Sign Recognition Pipeline using TensorFlow**


[image1]: ./ouput-pictures/image1.png "histogram of label frequency before before data augmentation) "
[image2]: ./ouput-pictures/image2.png "Logo Title Text 2"
[image3]: ./ouput-pictures/image3.png "Logo Title Text 2"
[image4]: ./ouput-pictures/image4.png "Logo Title Text 2"
[image5]: ./ouput-pictures/image5.png "Logo Title Text 2"
[image6]: ./ouput-pictures/image6.png "Logo Title Text 2"
[image7]: ./ouput-pictures/image7.png "Logo Title Text 2"
[image8]: ./ouput-pictures/image8.png "Logo Title Text 2"
[image9]: ./ouput-pictures/image9.png "Logo Title Text 2"
[image10]: ./ouput-pictures/image10.png "Logo Title Text 2"

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


A perceptron is a state machine that basically deicides if something is on or off, the idea being that most problems in the world require more than one input of information to make an informed decision and based on the amount of information collected a decision might be more pronounced. This section of the class really focused on the behind the senses ideas that makes up machine learning, and gave deeper insight into how weights and biases impact a networks performance. This was helpful in modifying the LeNet network to recognize german traffic signs and create a model that achived high accuracy.

Artificial intelligence has become the hottest topic in tech the last few years with breakthroughs in hardware lowing computation cost of GPUs and software libraries that increase time to market for complex application. Amazon, Google and Apple have all heavily invested into handwriting and speech neural networks for added functionality in their products. Google, Tesla, and a wide range of startups are leveraging machine learning to train cars to drive without human interaction, and hence the importance of this course,  detecting signs in real time with higher accuracy than a human. All these technologies are the accumulation of decades of research into artificial inelegance.

Taking the basics of a Neural network pipelines, it's possible to create image recognition software that can detect objects once trained with cleaned data. To do this a set of steps must be made, and this assignment took a deep dive into each of this steps. The great part about this technology is it can be applied to a wide range of task. Topic of next section, once the model has been trained to detect objects we will retrain it for another task.


The LeNet Lab Solution was the starting part for the traffic detection pipeline I created. Using Tensor Flow, Google's machine learning library to speed up development time.  The Input was the german data set provided for free, the data set was already cleaned to only contain 32x32 pixel photos, saving the students from the many hours it takes to clean such a large data set. Data manipulation and conditioning can often be the most difficult part of creating a numeral network pipeline. While this project only required minor data to be loaded from a very clean data set, minor variation where added to go from the RGB color scale to a gray scale to better match the LeNet architecture.

The LeNet network was modified to gain accuracy for sign detection. The input of the network takes 32x32 photos from a data set and create hidden layers to analyze the image for features. Using convolution a mathematical principle to modify the pixels to add more layers and compare changes. Understanding that color images are not the best to analyze because each image has 3 layers, think (32X32)x3 equals 3,702 total pixel the script will have to process for every image.  By simplifying the data set to gray scale we can flatten in a senses the color data to a grayscale value between 0 and 255, if the pixel are 8 bits. This input becomes important to the feature map layer where we go from 3 layers of 32x32 to 6 layers of 28x28 pixels. Again we samples the data going form 6 layer to another 6 layers of 14X14 pixel.  We do a similar process two more times going to 16 layers at 10x10 and 16 layers going to 5x5 pixel. Ending with 3 passes of a connected layers to output the final answer output.

After running the model the first time with just the LeNet pipeline and achieving a ~91% accuracy it was time to play with hyperparameters to dial in the pipeline to achieve a better accuracy.   I personally choose  an epochs size of  50 because it seemed like a good number, and the number of times the model trains itself on different batches could only help with gaining accuracy. While there is a worry about overfitting I had a way to randomize the data sets to avoid this latter.
For the others two parameters I found that mu preformed best at about 0.001 and I couldn’t notice a difference when changing sigma. These values directly impact backpropagation and the way the model finds layer errors, so eh ideas to minimize offset and avoid local minimum in search of a global minimum.

Pipeline Implementation


One this I have to say about this Nano-degree is the side articles and resources are amazing, while working on my masters I pitched up a number of habits, and one I still love is reading cutting edge papers on different subjects. Two papers that helped me really grasp the subject and overcome some mental barrier I suffered with then I first worked on this assynmnet where “Dropout a Simple Way to Prevent Neural Networks from Overfitting” <a href= "http://www.jmlr.org/papers/volume15/srivastava14a.old/source/srivastava14a.pdf"> Source </a>

The other paper "Traffic Sign Recognition with Multi-Scale Convolutional Networks” <a href="http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf"> Source </a> is a great paper on doing exactly what this assignment is. Its network is based on the same underlying fundamentals to the LeNet network with some interesting variation on 2nd stage layers and pictured above. These different aurally reduce the number of hidden layers in the pipeline and also reduce the number of overall connected layers at the end. The results the paper achieved where also fascinating with this aprons and seem to be only slightly better with a 97.33% accuracy. I did try to remove a few of the hidden layers and she if the accuracy changes an defund that the LeNet2 model I created was not signifigantly better or worse missing 2 hidden layers at the classier stage.

Notes to Course, the stricture of the course was great, but took me off track as I dove a little to deep into stanford lectures on neural networks and the foundation, I have senses gone back to examine more the Tensor flow documentation to better understand the impact of working with libraries to avoid the low level implantation.

From what I have learned and please correct me if I miss something, the overall process of creating a neural network is to first develop and pipeline with some N number of hidden layers for future recognition that can accept a specific data type. Next is training the model so that the network creates it’s own weights and bias for the convolution and back propagation to find error and create understanding of the datas sets features.  Taking special care to avoid over and under fitting of the model you plan to uses un the next steps. Play with ethics’s model (AWS made this much faster) and see how twerking hyperparameters effects the output on different or randomized data sets.

Once you have a model with reasonable accuracy comes the next part, running real data through it and seeing how well it preforms, this requires cleaned real world data, thank you Germany for this, and away to teach check if the model is getting the correct answers. This is often left to the user, me, or on a larger scale real people telling the model it was right or wrong.

Three Main Steps
1.Create LeNet training Pipeline
2.Trying the Model
3.Evaluation of Pipeline

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is, 34799.
* The size of the validation set is, 12630.
* The size of test set is 32x32x3.
* The shape of a traffic sign image is square 32x32.
* The number of unique classes/labels in the data set is 43.

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]
histogram of label frequency (once again, before data augmentation)

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it helps isolate different visual attributes. Think about the lane line project, we converted to gray scale and then used Canny edge detection to find the gradient at different lines in the image. 

Conversion of images to grayscale using openCV to allow the model to train on traffic sign features
Image histogram equalization using the openCV CLAHE (Constrast-Limited Adaptive Histogram Equalization) algorithm to improve the illumination in the images. The poor brightness across some images can be seen in the data set visualization presented earlier in Section 2.2 Conversely, there are images included in the dataset that are over-exposed and also need to be normalized. This equalization is carried out on both the RGB images and grayscale images independently.

The (32,32,1) equalized grayscale images and (32,32,3) equalized RGB images are then merged to create a combined image of dimensions (32,32,4). This is done to provide the model with both the grayscale features as well as the color information embedded within the image which provides additional information for the classifier to train on.

Finally, the 4-channel images are normalized by subtracting the mean and dividing by the standard deviation of the entire training data split. It is important to note that the validation and test data splits are also normalized using the mean and std dev of the training data split to maintain consistency in pre-processing.

![alt text][image2]

![alt text][image3]

This final augmented & processed training data set is then used to train the LeNet Model. 

|####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I began by implementing the same architecture from the LeNet Lab, with no changes since because I converted the data set to grayscale. This model worked quite well obtaining 94% validation accuracy), but I also implemented the Sermanet/LeCun model from their traffic sign classifier paper and saw an immediate improvement. Although the paper doesn't go into detail describing exactly how the model is implemented (particularly the depth of the layers) I was able to make it work. 

My final model consisted of the following layers:

|Layer	| Description |
|:---------------------:|:---------------------------------------------:|
|Input	| 32x32x4 gray-RGB image|
|Convolution L1 5x5	| 1x1 stride, valid padding, outputs 28x28x6 |
|RELUActivation	| 	 | 
|Max pooling|	2x2 stride, outputs 14x14x6 |
|Convolution L2 5x5 |	1x1 stride, valid padding, outputs 10x10x16 |
|RELU Activation | 	  |
|Max pooling	2x2 | stride, valid padding, outputs 5x5x16 |
|Fully Connected Layer L3	| Input flattened 5x5x16 dims from previous max pooling layer, outputs 1x1x400 |
|RELU Activation | |
|Dropout	| |
|Flatten | Input = 5x5x16. Output = 400
|Fully Connected Layer L4 |	Input flattened 500 dims from previous max pooling layer, outputs 1x1x400  |
|RELU Activation |	|
|Dropout	| |
|Fully Connected Layer L5 | outputs 43 | 
|Flatten | Input = 5x5x16. Output = 400 |
|Flatten | Input = 1x1x400. Output = 400 |
|Concat | Input = 400 + 400. Output = 800 |
|Dropout	| |
|Fully Connected Layer4 | Input = 800. Output = 43 |

This  appeared  to work well, but I think I will need to go back and randomize the data better as it seems to be over-fitting with a accuracy of 98% after running teh model again. 

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an the training data provided and as explained above manipulated the data into grayscale and then applied  the Sermanet/LeCun model to get more accuracy in the training. I normalized the dataset to get a more even spreed of different signs so that signs that appear more don't have a higher probability of being mistaking for signs that appear less in the data set.

I used trial and error, so time consuming, every time I change the epoch's I has to weight for the model had to be re-trained. I ended up graphing the output to see how the epochs where effecting the training and test accuracy and found that I was over fitting. I need to go back and manipulate the data more to deal with the over fitting. I would probably flip some images and also distort others to help the model have more random images to train on. 

Two pentameters that have a large impact in training time and accuracy. 
BATCH_SIZE 100 
ECHOS 50 


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My first pass, after watching the video walk through in this assignment was 95% accuracy, I then played with the epochs and training data to get more accuracy until I achieved 99%. TI was not concerned at first with over-fitting. I have added two dropout layers to hopefully address the issues but have not seen huge gains in doing so. 


My final model results were:
* training set accuracy of 99%
* validation set accuracy of 98%
* test set accuracy of 99%

If an alliterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
	My first Architecture was LeNet used in to lesson, I then modified it based on ermanet/LeCun model.
* What were some problems with the initial architecture?
	It only achieved a 94% accuracy after training. 
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web, I ended up testing on 14 different signs, you can see the images of all the signs and the perditions next. 

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8] 

These images are taken from Google search and then processed to be 32x32x3 dimensions, then we run them through the classifier model and see what the probability is they are identified correctly. 

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image9]



The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.



| Probability         |     Prediction        |
|:---------------------:|:---------------------------------------------:|
| 1.0        | Stop sign   |
| 1.0     | U-turn |
| 1.5		| Yield|
| 1.4      | Bumpy Road |
| 1.1    | Slippery Road      |


the precision for a class is the number of true positives (i.e. the number of items correctly labeled as belonging to the positive class) divided by the total number of elements labeled as belonging to the positive class (i.e. the sum of true positives and false positives, which are items incorrectly labeled as belonging to the class). Recall in this context is defined as the number of true positives divided by the total number of elements that actually belong to the positive class (i.e. the sum of true positives and false negatives, which are items which were not labeled as belonging to the positive class but should have been).



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
