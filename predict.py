#!/usr/bin/env python
# coding: utf-8

# # Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.
# 
# Please make sure if you are running this notebook in the workspace that you have chosen GPU rather than CPU mode.

# In[50]:


# Imports here
import numpy as np                   # Import numpy library
import torch                         # Import pytorch
from torch import nn                 # Required for classifier
from torchvision import transforms   # Required to transform the data
from collections import OrderedDict  # Required for classifier
from torchvision import models       # Required to get the pre-Trained Model
from PIL import Image                #Required to open the image
import torch.nn.functional as F      #Required to use F.softmax function  
import matplotlib.pyplot as plt      #Required to plot the image in Sanity check
import seaborn as sns                #Required to plot the image in Sanity check


# In[2]:


#Global constants and variables
#Create some constants so to reuse them. in C++/.Net we usually use uppercase for the constants
TRAIN = "train"
VALID = "valid"
TEST  = "test"

# Command Line arguments
import argparse
ap = argparse.ArgumentParser(description='Parser for predict.py')
ap.add_argument('data_dir', nargs='*', action="store", default="./flowers")
#Set directory to save checkpoints:
ap.add_argument('--save_dir', dest="save_directory", action="store", default="./VikasClassifier.pth")
ap.add_argument('--arch', dest="arch", action="store", default="densenet121", type = str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('input_img', default='flowers/test/1/image_06743.jpg', nargs='*', action="store", type = str)
ap.add_argument('--top_k', dest="top_k", action="store", type=int, default=5)
pa = ap.parse_args()

ARCH                   = pa.arch
#Normalise mean and standard deviation is already provided
normMean = [0.485, 0.456, 0.406]
normStd  = [0.229, 0.224, 0.225]

isCuda = (pa.gpu == "gpu")
if (isCuda):
    device = "cuda:0"
else:
    device = "cpu"
    
#checkpoint path
CHECKPOINT_PATH = pa.save_directory

#Since machines works with numbers, we need to create a mapping from actual flower catagoreis to numbers between 1 and 102 
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

# In[3]:

#Define function to get pretrained model based on arch
def GetPreTrainedModel():
    if ARCH   == 'vgg16':
        model = models.vgg16(pretrained=True)        
    elif ARCH == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif ARCH == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("{} is not a valid model.should be among 'vgg16/densenet121/alexnet'?".format(arch))
    return model, model.classifier.in_features
	
def loadModel():
    print("loading Checkpoint '{}'".format(CHECKPOINT_PATH))
    checkpoint       = torch.load(CHECKPOINT_PATH)
    
    model, numInputFeaturesPrerainedModel = GetPreTrainedModel()
    inputFeatures    = checkpoint['inputFeatures']
    outputFeatures   = checkpoint['outputFeatures']
    dropout          = checkpoint['dropout']
    
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.class_to_idx = checkpoint['class_to_idx']
    hiddenLayers=[int(inputFeatures/2),int(inputFeatures/4),int(inputFeatures/8)]

    # Create the classifier
    classifier = nn.Sequential(OrderedDict([
        ('dropout'       , nn.Dropout(dropout)),
        ('inputs'        , nn.Linear(in_features=inputFeatures,  out_features=hiddenLayers[0])),
        ('relu1'         , nn.ReLU()),
        ('hiddenLayer1'  , nn.Linear(hiddenLayers[0], hiddenLayers[1])),
        ('relu2'         , nn.ReLU()),
        ('hiddenLayer2'  , nn.Linear(hiddenLayers[1], hiddenLayers[2])),
        ('relu3'         , nn.ReLU()),
        ('lastHiddenLayer'  , nn.Linear(hiddenLayers[2],outputFeatures)),
        ('output'        , nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    print("Loaded Model in_features  : " , model.classifier.inputs.in_features)
    print("Loaded Model out_features : " , model.classifier.lastHiddenLayer.out_features)
    print("Loaded Model dropout      : " , model.classifier.dropout.p)
    
    return model

loadedModel = loadModel()

# # Inference for classification
# 
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# First you'll need to handle processing the input image such that it can be used in your network. 
# 
# ## Image Preprocessing
# 
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 
# 
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
# 
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
# 
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 
# 
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

# In[19]:


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    # Open the image
    
    imgPil = Image.open(image)
    
    #do the adjustments
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(normMean, normStd)])
   
    return adjustments(imgPil)


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# In[44]:


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean  = np.array(normMean)
    std   = np.array(normStd)
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[24]:


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    # Process image
    img = process_image(image_path)
    
    # Add batch of size 1 to image
    img_batch = img.unsqueeze(0)
    
    model_input = img_batch.float()
    
    # Probabilities using softmax
    with torch.no_grad():
        probs = torch.exp(model.forward(model_input))
    
    probability = F.softmax(probs.data,dim=1)
    
    return probability.topk(topk)


# In[27]:


img = (pa.input_img)
val1, val2 = predict(img, loadedModel)
print(val1)
print(val2)


# ## Sanity Checking
# 
# Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:
# 
# <img src='assets/inference_example.png' width=300px>
# 
# You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.

# In[60]:


# TODO: Display an image along with the top 5 classes
def plot_solution(image_path, model):
    # Set up plot
    #plt.figure(figsize = (6,10))
    #ax = plt.subplot(2,1,1)
    # Set up title
    #flower_num = image_path.split('/')[1]
    #title_ = cat_to_name[flower_num]
    #print(title_)
    # Plot flower
    #image_path = data_dir + '/' + image_path 
    #img = process_image(image_path)
    #imshow(img, ax, title = title_);
    # Make prediction
    probabilities = predict(image_path, model,pa.top_k)
    
    probs    = np.array(probabilities[0][0])
    flowers  = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
    
    
    
    # Plot bar chart
    #plt.subplot(2,1,2)
    #sns.barplot(x=probs, y=flowers, color=sns.color_palette()[0]);
    #plt.show()
    i = 0
    while i < pa.top_k:
        print("{} with a probability of {}".format(flowers[i], probs[i]))
        i += 1
        
# In[61]:


#Check Sanity
plot_solution(pa.input_img  , loadedModel)  #first in the test folder

print("Prediction completed")