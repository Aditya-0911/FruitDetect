# FruitDetect
## FruitDetect: Machine Learning for Fruit Identification
![image](https://i.pinimg.com/originals/e8/03/2d/e8032db73ff332d3e3b1de0815738bef.gif)  

## Overview :clipboard:

The "FruitDetect" project harnesses the capabilities of neural networks to tackle the complex task of classifying 131 distinct fruit varieties based on their images. Identifying and categorizing fruits manually can be a daunting challenge due to the immense diversity in shape, size, color, and texture across different fruit types. The neural networks employed in this project possess the capability to learn intricate patterns and features from the input fruit images, enabling them to make highly accurate predictions regarding the fruit's class. Through the power of convolutional neural networks (CNNs) and deep learning techniques, we have fine-tuned our models to recognize not only the most common fruits but also rare and exotic varieties.  

## Data :mango:

The dataset used for this project is available on [Kaggle](https://www.kaggle.com/datasets/moltean/fruits) and can be imported using the Kaggle API. The dataset Fruits360 folder is organized into two main folders:

- **Train:** Contains subfolders labeled with the names of specific fruits or vegetables. Each subfolder contains images of the respective fruit or vegetable and will be used for model training.

- **Test:** Similar to the "Train" folder, the "Test" folder contains subfolders labeled with fruit or vegetable names. These subfolders contain images used for testing and evaluating our models.

## Models Used 🔍

All the models are created and used in `FruitDetect.ipynb` file which can also be used in google colab

* Model 1: Model 1 has a simple ANN architecture with 2 Dense layer one having 3 nodes of ReLU activation and other layer having 131 nodes of Softmax activation to classify the fruit images. Before the Dense layers I have used a Flatten layer to reshape 3D matrix to 1D matrix. The model uses Adam optimizer and runs for 10 epochs
  * Loss of Model1 on test data:  4.858613967895508
  * Accuracy of Model1 on test data:  0.014456981793045998
  * Precision of Model1 on test data:  0.0
  * Recall of Model1 on test data:  0.0
* Model 2: Model 2 has a simple CNN architecture with 1 Conv2D layer with 10 filters of 3X3 size with the activation function of ReLU, a Maxpooling layer with pool size 2 and valid padding, 2 Dense layer one having 3 nodes of ReLU activation and other layer having 131 nodes of Softmax activation to classify the fruit images. Before the Dense layers I have used a Flatten layer to reshape 3D matrix to 1D matrix. The model uses Adam optimizer and runs for 5 epochs
  * Loss of Model2 on test data:  4.856095314025879
  * Accuracy of Model2 on test data:  0.014456981793045998
  * Precision of Model2 on test data:  0.0
  * Recall of Model2 on test data:  0.0
* Model 3: Model 3 uses the concept of tranfer learning in which we use predefined weights from a large model trained on much larger and bigger dataset. For my model I will be using InceptionNet model to perform classificaiton on my dataset. I will be only using the Inception model and add one Dense layer of 131 nodes to classify the images.
  * Loss on test data:  0.804618775844574
  * Accuracy on test data:  0.7943847179412842
  * Precision on test data:  0.8339244723320007
  * Recall on test data:  0.7668811678886414
* Model 4: Model 4 uses the same concept of transfer learning but this time I will be using a different model, VGG19 model on my dataset and add one Dense layer of 131 nodes to classify the images.
  * Loss on test data:  1.8081496953964233
  * Accuracy on test data:  0.5828632116317749
  * Precision on test data:  0.9758908152580261
  * Recall on test data:  0.2836741805076599
* Model 5: Model 5 uses the  concept of transfer learning using the InceptionNet model but this time i will be adding Conv2D layer with 20 1X1 filteres with ReLU activation and Dense layer with 10 nodes of ReLU activation and a Dense layer to classify the images
  * Loss on test data:  1.5649464130401611
  * Accuracy on test data:  0.56633460521698
  * Precision on test data:  0.8666767477989197
  * Recall on test data:  0.2524241805076599
* Model 6: Model 6 uses the  concept of transfer learning using the VGG19 model but this time i will be adding Conv2D layer with 20 1X1 filteres with ReLU activation and Dense layer with 10 nodes of ReLU activation and a Dense layer to classify the images
  * Loss on test data:  1.9668771028518677
  * Accuracy on test data:  0.4842648208141327
  * Precision on test data:  0.8855530023574829
  * Recall on test data:  0.15210683643817902


