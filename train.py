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
import numpy as np                    # Import numpy library
import torch                          # Import pytorch

from torchvision import transforms   # Required to transform the data
from torchvision import datasets     # Required to load the datasets with ImageFolder

from torchvision import models       # Required to get the pre-Trained Model
from torch import nn                 # Required for classifier
from collections import OrderedDict  # Required for classifier

from torch import optim              #Required to optimize the paramaters while training the model
from torch.optim import lr_scheduler #Required to decay the Learning Rate while training the model
import time                          #Required to get the current time while training the model
import copy                          #Required to do the deep copy of model while training the model


# In[2]:
#Global constants and variables
#Create some constants so to reuse them. in C++/.Net we usually use uppercase for the constants
TRAIN = "train"
VALID = "valid"
TEST  = "test"

# Command Line arguments
import argparse
ap = argparse.ArgumentParser(description='Parder for train.py')
ap.add_argument('data_dir', nargs='*', action="store", default="./flowers")
#Set directory to save checkpoints:
ap.add_argument('--save_dir', dest="save_directory", action="store", default="./VikasClassifier.pth")
ap.add_argument('--arch', dest="arch", action="store", default="densenet121", type = str)
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=12)
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
ap.add_argument('--trainingModelFeatures', dest = "TrainingModelFeatures", action = "store", default = 102)
ap.add_argument('--stepSize', dest = "StepSize", action = "store", default = 4)
ap.add_argument('--gamma', dest = "Gamma", action = "store", default = 0.1)
pa = ap.parse_args()


ARCH                   = pa.arch
DROPOUT                = pa.dropout
TRAINEDMODEL_FEATURES  = pa.TrainingModelFeatures #we have 102 catagores of flower

LEARNING_RATE          = pa.learning_rate
STEP_SIZE              = pa.StepSize
GAMMA                  = pa.Gamma
numEpochs              = pa.epochs # Number of epochs 
                            # One iteration is full run of feedforward and backpropagation through the network.

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

# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

# In[3]:


data_dir  = pa.data_dir

train_dir = data_dir + '/' + TRAIN
valid_dir = data_dir + '/' + VALID
test_dir  = data_dir + '/' + TEST

#Create dictonary to easily get the directories
dirs = {TRAIN: train_dir, VALID: valid_dir, TEST : test_dir}

print("Data Loaded")


# In[4]:


# TODO: Define your transforms for the training, validation, and testing sets
#from torchvision import transforms #Required to transform the data
'''
Here we are doing robust transformations for Training data as images could be rotated/flipped/cropped
but doing simple transformation for validation and test data
'''
data_transforms = {
    TRAIN: transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(normMean, normStd)]),
    VALID: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(normMean, normStd)]),
    TEST: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(normMean, normStd)]),
}

# TODO: Load the datasets with ImageFolder
#from torchvision import datasets #Required to load the datasets with ImageFolder
trainData = datasets.ImageFolder(train_dir, transform = data_transforms[TRAIN])
validData = datasets.ImageFolder(valid_dir, transform = data_transforms[VALID])
testData  = datasets.ImageFolder(test_dir , transform = data_transforms[TEST])

#Get the size of each datasets
print("Size of Training dataset   = ", len(trainData))
print("Size of Validation dataset = ", len(validData))
print("Size of Test dataset       = ", len(testData))

#Create dictionary for database sizes used in calaculating loss
datasetSizes = {TRAIN: len(trainData), VALID: len(validData), TEST : len(testData)}

# TODO: Using the image datasets and the trainforms, define the dataloaders
# Tried using different batch sizes for each datasets
trainLoader = torch.utils.data.DataLoader(trainData, batch_size = 64, shuffle = True)
validLoader = torch.utils.data.DataLoader(validData, batch_size = 32, shuffle = True)
testLoader  = torch.utils.data.DataLoader(testData,  batch_size = 16, shuffle = True)

#Dataloaders dictionary
dataloaders = {TRAIN:trainLoader, VALID:validLoader, TEST:testLoader }

#Get class ids
class_idx = trainData.class_to_idx

print("Executed : Transforms for the training, validation, and testing sets")


# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[5]:


#Since machines works with numbers, we need to create a mapping from actual flower catagoreis to numbers between 1 and 102 
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
print("Executed Cell4 : Label Mapping")


# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.
# 
# One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.

# In[6]:


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


# In[7]:


# TODO: Build and train your network
def CreateModel(class_to_idx):

    #from torchvision import models #Required to get the pre-Trained Model

    #Setp1 : Get the pre-Trained Model
    model, numInputFeaturesPrerainedModel = GetPreTrainedModel()
    
    #Step2 : Stop the model to update weights for pre-Trained Model
    for param in model.parameters():
        param.requires_grad = False

    #Step3 : check the architecture of pre-Trained Model so to see the classification    
    #print(model)
    print("pre-Trained Model in_features  : ", model.classifier.in_features)
    print("pre-Trained Model out_features : ", model.classifier.out_features)

    #from torch import nn # Required for classifier
    #from collections import OrderedDict  # Required for classifier

    # Update Model Classifier
    #lets have 3 hidden layers and use RELU activation function and softmax as final layer
    hiddenLayers=[int(numInputFeaturesPrerainedModel/2),int(numInputFeaturesPrerainedModel/4),int(numInputFeaturesPrerainedModel/8)]
    classifier = nn.Sequential(OrderedDict([
        ('dropout'          , nn.Dropout(DROPOUT)),
        ('inputs'           , nn.Linear(in_features=numInputFeaturesPrerainedModel,  out_features=hiddenLayers[0])),
        ('relu1'            , nn.ReLU()),
        ('hiddenLayer1'     , nn.Linear(hiddenLayers[0], hiddenLayers[1])),
        ('relu2'            , nn.ReLU()),
        ('hiddenLayer2'     , nn.Linear(hiddenLayers[1], hiddenLayers[2])),
        ('relu3'            , nn.ReLU()),
        ('lastHiddenLayer'  , nn.Linear(hiddenLayers[2],TRAINEDMODEL_FEATURES)),
        ('output'           , nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    
    print("Created Model in_features  : ", model.classifier.inputs.in_features)
    print("Created Model out_features : ", model.classifier.lastHiddenLayer.out_features)
    print("Created Model dropout      : ", model.classifier.dropout.p)
    
    model.class_to_idx = class_to_idx
    if (isCuda):
        model.cuda()
        
    return model


# In[8]:


#Define function to train the Model
''' 
model      -> model to be trained
criterion  -> method used to evaluate the model fit.
optimizer  -> optimization technique used to update the weights.
scheduler  -> provides different methods for adjusting the learning rate and step size used during optimization.
'''
#import time #Required to get the current time
#import copy #Required to do the deep copy of model
def trainModel(model, criterion, optimizer, scheduler, dataloaders):    
    #Capture start time
    startTime = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(numEpochs):        
        print("Epoch: {:02}/{:02}...".format(epoch+1, numEpochs), end='') #end='' prevents next print statement in newline
        # Each epoch has a training and validation phase
        for phase in [TRAIN, VALID]:
            if phase == TRAIN:
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:                
                inputs = inputs.to(device)
                labels = labels.to(device)
                model  = model.to(device)
                #print("Tensor Devices: ", inputs.device, labels.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == TRAIN):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == TRAIN:
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / datasetSizes[phase]
            epoch_acc = running_corrects.double() / datasetSizes[phase]

            print("...{} [Loss: {:.4f}".format(phase, epoch_loss), "Accuracy: {:.4f}]".format(epoch_acc), end='')


            # deep copy the model
            if phase == VALID and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - startTime
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Validation Accuracy: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# In[9]:


# Train the network
#
model = CreateModel(trainData.class_to_idx)

# Criteria NLLLoss which is recommended with Softmax final layer
criterion = nn.NLLLoss()

# Observe that all parameters are being optimized
#from torch import optim #Required to optimize the paramaters while training the model
optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

# Decay LR by a factor of 0.1 every 4 epochs
#from torch.optim import lr_scheduler #Required to decay the Learning Rate while training the model
scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

TrainedModel = trainModel(model, criterion, optimizer,scheduler, dataloaders)


# ## Testing your network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# In[10]:


#Define function to Test the model
def testModel(model):
    model.eval()
    model.to(device)
    correct = 0
    total   = 0
    
    with torch.no_grad(): #we are not interested to train with this dataset.
        for idx, (images, labels) in enumerate(dataloaders[TEST]):
            if isCuda:
                images, labels = images.cuda(), labels.cuda()
            # obtain the outputs from the model
            outputs = model.forward(images)
            # max provides the (maximum probability, max value)
            _, predicted = outputs.max(dim=1)
            # check the accuracy
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
            print('Test Accuracy of the network: %d %%' % (100 * correct / total))  
    print('Testing completed')


# In[11]:


# TODO: Do validation on the test set
testModel(model)


# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[12]:


# TODO: Save the checkpoint 
model.class_to_idx = trainData.class_to_idx
model.cpu()
torch.save({'arch'          : ARCH,
            'state_dict'    : model.state_dict(),
            'inputFeatures' : model.classifier.inputs.in_features,
            'outputFeatures': model.classifier.lastHiddenLayer.out_features,
            'dropout'       : model.classifier.dropout.p,
            'class_to_idx'  : model.class_to_idx},
            CHECKPOINT_PATH)
print("Checkpoint '{}' Saved".format(CHECKPOINT_PATH))

print("The Model is Trained and checkpoint Saved")
