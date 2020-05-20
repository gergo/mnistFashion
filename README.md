## scripts 

```python/mnist_fashion_train.ipynb``` has a workbook to:
 - load the [mnist fashion dataset](https://github.com/zalandoresearch/fashion-mnist)
 - set up and train a convolutional neural network to classify images to 10 fashion categories
 - save the trained image
 
 ```python/categorize_images.py``` is a command-line executable python script that
 - gets an image as an argument
 - loads the pre-trained model
 - classifies the image
 
## Enhancement ideas
- use augmentation to make the model more robust
- use a pre-trained model for a more complex (probably more accurate) model