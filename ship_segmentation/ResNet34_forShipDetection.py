#!/usr/bin/env python
# coding: utf-8

# # Training ResNet34 for ship detection (ship/no-ship)
# ## Overview
# We've downloaded a pretrained ResNet34 model and retrained it on our dataset for ship detection task. Later we'll use this model as a backbone in our U-Net architecture model for ships segmentation. 
# After training of the head layers of the model on 256x256 rescaled images for one epoch the accuracy has reached ~94%. The following fine-tuning of entire model for 2 more epochs with learning rate annealing boosted the accuracy to ~97%. We then continued training for several epochs with a new data set composed of images of 384x384 resolution, the accuracy had boosted to ~98%. Unfortunately, continuing training the model on full resolution, 768x768, images leaded to reduction of the accuracy that is likely attributed to insufficient model capacity.

# In[1]:


# Imports
import pandas as pd
import numpy as np
import os
import sys
import random

from PIL import Image
from sklearn.model_selection import train_test_split
from old.fastai.conv_learner import *
from old.fastai.dataset import *


# In[2]:


# Paths
PATH = 'C:/Users/User/Desktop/Avshalom&Naama/jupyter_files/fastai-master/'
TRAIN = '../../data/train_v2/'
TEST = '../../data/test_v2/'
SEGMENTATION = '../../data/train_ship_segmentations_v2.csv'


# ## Data Preparation
# ### Split train-validation
# Split train data to train set and validation set. 5% of the train data is sufficient for model evaluation thus split ratio is set to 5% validation / 95% train.

# In[3]:


train_names = [f for f in os.listdir(TRAIN)]
test_names = [f for f in os.listdir(TEST)]
tr_n, val_n = train_test_split(train_names, test_size=0.05, random_state=42)


# ## Data utils
# ### Data loader
# Implements get_image(), get_grountruth() and get_num_classes().
# ### Get Data
# Generate input data for training stage. also perform augmentations and transformations on original data for better generalization performance. 

# In[4]:


# CNN architectur
arch = resnet34

# Data loader (for data handeling)
class pdFilesDataset(FilesDataset):
    
    # Constructor/Initializator - create a dictionary of the train images and their segmentation data.
    def __init__(self, fnames, path, transform):
        self.segmentation_df = pd.read_csv(SEGMENTATION).set_index('ImageId')
        super().__init__(fnames, transform, path)
    
    # Get image i
    def get_x(self, i):
        img = open_image(os.path.join(self.path, self.fnames[i]))
        if self.sz == 768: return img 
        else: return cv2.resize(img, (self.sz, self.sz))
    
    # Get segmentation for image i
    # if in test/validation - return 0
    def get_y(self, i):
        if(self.path == TEST): return 0
        masks = self.segmentation_df.loc[self.fnames[i]]['EncodedPixels']
        if(type(masks) == float): return 0 #NAN - no ship 
        else: return 1
    
    # Get number of classes in dataset
    # classes = (ship, no-ship)
    def get_c(self): return 2 #number of classes

# Generate augmented and transformed dataset for NN Train use.
def get_data(sz,bs):
    #data augmentation
    aug_tfms = [RandomRotate(20, tfm_y=TfmType.NO),
                RandomDihedral(tfm_y=TfmType.NO),
                RandomLighting(0.05, 0.05, tfm_y=TfmType.NO)]
    tfms = tfms_from_model(arch, sz, crop_type=CropType.NO, tfm_y=TfmType.NO, 
                aug_tfms=aug_tfms)
    ds = ImageData.get_ds(pdFilesDataset, (tr_n[:-(len(tr_n)%bs)],TRAIN), 
                (val_n,TRAIN), tfms, test=(test_names,TEST))
    md = ImageData(PATH, ds, bs, num_workers=4, classes=None)
    return md


# ## Construct Model
# ### Model parameters
# set image size, batch size, number of epochs, optimizer type, initial learning rate.

# In[5]:


# Model parameters
sz = 256                # image size
bs = 64                 # batch size
num_eps = 1             # number of epochs
optimizer = optim.Adam  # optimizer type
lr = 2e-3               # initial learning rate

md = get_data(sz,bs)
learn = ConvLearner.pretrained(arch, md, ps=0.5) #dropout 50%
learn.opt_fn = optimizer


# ## Train Model
# we first started to train on low resolution images (256x256) for few epochs and only then we trained on higher resolution images (384x384).  we used this method for few reasons: shorter train time, GPU memory constrains and most important – improving the model generalizing abilities by training with low-res images.

# ### Train on low resolution images
# Train the head of the model with lr=2e-3 for 1 epoch. Then, unfreeze the rest of the model and train the head, middle and base of the model with lr =  2e-3, 5e-4 and 2e-3 respectevly for 2 more epochs since low level detector do not vary much from one image data set to another.

# In[8]:


learn.metrics=[accuracy,
               Precision(),
               Recall()]
learn.fit(lr, num_eps)


# In[7]:


learn.unfreeze()
lr=np.array([1e-4,5e-4,2e-3])


# In[8]:


learn.fit(lr, 1, cycle_len=2, use_clr=(20,8))
learn.save('Resnet34_lable_256_1')
#learn.sched.plot_lr()


# ### Train on high resolution images

# In[12]:


# Training on high resolution images
sz = 384 #image size
bs = 32  #batch size

md = get_data(sz,bs)
learn = ConvLearner.pretrained(arch, md, ps=0.5) #dropout 50%
learn.opt_fn = optim.Adam
learn.unfreeze()
lr=np.array([1e-4,5e-4,2e-3])


# In[13]:


learn.load('Resnet34_lable_256_1')


# In[14]:


learn.fit(lr/2, 1, cycle_len=2, use_clr=(20,8)) #lr is smaller since bs is only 32
learn.save('Resnet34_lable_384_1')


# # Calc presicion & recall

# In[ ]:


learn.load('Resnet34_lable_384_1')


# ## Predict 
# Every prediction above probability of 0.5 is counted as ship, else - no-ship.

# In[15]:


# Prediction
log_preds_384,y_384 = learn.predict_with_targs(is_test=True)
probs_384 = np.exp(log_preds_384)[:,1]
pred_384 = (probs_384 > 0.5).astype(int)


# In[16]:


df_384 = pd.DataFrame({'id':test_names, 'p_ship':probs_384})
df_384.to_csv('ship_detection_384.csv', header=True, index=False)


# ## Classification Visuailzation

# In[17]:


#ship_detection = pd.read_csv('ship_detection_256.csv')    # for 256x256 classifier
ship_detection = pd.read_csv('ship_detection_384.csv')    # for 384x384 classifier
test_names = ship_detection.loc[ship_detection['p_ship'] > 0.5, ['id']]['id'].values.tolist()
test_names_nothing = ship_detection.loc[ship_detection['p_ship'] <= 0.5, ['id']]['id'].values.tolist()
len(test_names), len(test_names_nothing)


# In[18]:


n = 16
rands = np.random.choice(len(test_names), n)
columns = 4
rows = n//4 + 1
fig=plt.figure(figsize=(columns*4, rows*4))
fig.suptitle('Classified as SHIP', fontsize=16)
for i in range(rows):
    for j in range(columns):
        idx = j+i*columns
        if idx >= n: break
        fig.add_subplot(rows, columns, idx+1)
        plt.axis('off')
        img = np.array(Image.open(os.path.join(TEST,test_names[rands[idx]])))
        plt.imshow(img)
plt.show()


# In[19]:


rands = np.random.choice(len(test_names_nothing), n)
fig=plt.figure(figsize=(columns*4, rows*4))
fig.suptitle('Classified as NO-SHIP', fontsize=16)
for i in range(rows):
    for j in range(columns):
        idx = j+i*columns
        if idx >= n: break
        fig.add_subplot(rows, columns, idx+1)
        plt.axis('off')
        img = np.array(Image.open(os.path.join(TEST,test_names_nothing[rands[idx]])))
        plt.imshow(img)
plt.show()


# In[20]:


import torch
torch.cuda.empty_cache()

