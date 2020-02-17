#!/usr/bin/env python
# coding: utf-8

# In[1]:


from old.fastai.conv_learner import *
from old.fastai.dataset import *

import torch
import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split


# In[2]:


#os.listdir('C:/Users/User/Desktop/Avshalom&Naama/jupyter_files/fastai-master/')
#os.listdir('C:/Users/User/Desktop/Avshalom&Naama/data/')


# <h1><center>Prepare Data</center></h1>

# In[3]:


#paths
PATH = 'C:/Users/User/Desktop/Avshalom&Naama/jupyter_files/fastai-master/'
TRAIN = '../../data/train_v2/'
TEST = '../../data/test_v2/'
SEGMENTATION = '../../data/train_ship_segmentations_v2.csv'
PRETRAINED = '../../models/ResNet34_384/Resnet34_lable_384_1.h5'

corrupted_image_train = '6384c3e78.jpg' #a corrupted image in train


# In[4]:


nw = 2   #number of workers for data loader -  the number of CPUs to use, maybe in windows shuld be = 0 - 
# check in https://forums.fast.ai/t/difference-between-setting-num-workers-0-in-fastai-pytorch/40040
arch = resnet34 #specify target architecture


# In[5]:


test_names = [f for f in os.listdir(TEST)]
#train_names.remove(corrupted_image_train) #remove corrupted image from train
#5% of data in the validation set is sufficient for model evaluation (checked)
tr_n, val_n = train_test_split(train_names, test_size=0.05, random_state=42)
#get segmentation of ground truth
segmentation_df = pd.read_csv(os.path.join(PATH, SEGMENTATION)).set_index('ImageId')


# <div style="text-align: left">The data is unbalanced - there are more images with no ships and if there are ships, the masked ships are a small precentege of pixels in the image. Therefore, we dropped all images with no ships to balance the training set. It also reduces the time per each epoch.    
# </div>

# In[6]:


# if in ground truth there is no segment (no ship), remove from train set and validation set - str represents ship
def cut_empty(names):
    return [name for name in names 
            if(type(segmentation_df.loc[name]['EncodedPixels']) != float)]

#removing no ship
tr_n = cut_empty(tr_n)
val_n = cut_empty(val_n)


# In[7]:


# get mask of image from the data file
def get_mask(img_id, df):
    shape = (768,768)
    #make array of zeroes for mask
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    #read masks of image from data file
    masks = df.loc[img_id]['EncodedPixels']
    #for image with no ship mask (float) return zeros image in the original image shape
    if(type(masks) == float): return img.reshape(shape)
    #for image with ships masks (str) make new array of masks (diffrent ships) 
    if(type(masks) == str): masks = [masks]
    #for every mask of ship in image mark the mask as 1
    for mask in masks:
        s = mask.split()
        for i in range(len(s)//2):
            start = int(s[2*i]) - 1
            length = int(s[2*i+1])
            img[start:start+length] = 1
    # return reshaped mask
    return img.reshape(shape).T


# In[8]:


#extracts images and mask in shape 768X768 from file. inherits from FilesDataset class in fastsai - used by data loader
class pdFilesDataset(FilesDataset):
    def __init__(self, fnames, path, transform):
        self.segmentation_df = pd.read_csv(SEGMENTATION).set_index('ImageId')
        super().__init__(fnames, transform, path)
    
    #get image
    def get_x(self, i):
        img = open_image(os.path.join(self.path, self.fnames[i]))
        # if maximum size of image in data set is 768X768 - resize the image to maximum shape
        if self.sz == 768: return img 
        else: return cv2.resize(img, (self.sz, self.sz))
    
    #get mask
    def get_y(self, i):
        #if test set - get empty mask, else ground truth
        mask = np.zeros((768,768), dtype=np.uint8) if (self.path == TEST)             else get_mask(self.fnames[i], self.segmentation_df)
        img = Image.fromarray(mask).resize((self.sz, self.sz)).convert('RGB')
        return np.array(img).astype(np.float32)
    
    def get_c(self): return 0


# <div style="text-align: left"> 
# Data Augmentation - prevents overfitting. Transforms = rotating, changing lighting
# </div>

# In[9]:


#augmentation - lighting (because kaggle dont have the new version of fastai we added this implementation)
class RandomLighting(Transform):
    def __init__(self, b, c, tfm_y=TfmType.NO):
        super().__init__(tfm_y)
        self.b,self.c = b,c # b for balance, c for contrast  to adjust random picture lighting

    def set_state(self):
        self.store.b_rand = rand0(self.b)
        self.store.c_rand = rand0(self.c)

    def do_transform(self, x, is_y):
        if is_y and self.tfm_y != TfmType.PIXEL: return x  #add this line to fix the bug
        b = self.store.b_rand
        c = self.store.c_rand
        c = -1/(c-1) if c<0 else c+1
        x = lighting(x, b, c)
        return x


# <div style="text-align: left"> 
# Data Loader - creats data set and data for the model
# </div>

# In[10]:


def get_data(sz,bs):
    #data augmentation 
    aug_tfms = [RandomRotate(20, tfm_y=TfmType.CLASS), # x degree random rotation
                RandomDihedral(tfm_y=TfmType.CLASS), # Rotates images by random multiples of 90 deg and/or reflection (flips)
                RandomLighting(0.05, 0.05, tfm_y=TfmType.CLASS)] # random picture lighting
    #resizing, image cropping, initial normalization, needs the pretraind arch to know normalization way
    tfms = tfms_from_model(arch, sz, crop_type=CropType.NO, tfm_y=TfmType.CLASS, 
                aug_tfms=aug_tfms)
    tr_names = tr_n if (len(tr_n)%bs == 0) else tr_n[:-(len(tr_n)%bs)] #cut incomplete batch
    #import dataset (train,test,validation) from data file with transforms (augmentations) 
    ds = ImageData.get_ds(pdFilesDataset, (tr_names,TRAIN), 
                (val_n,TRAIN), tfms, test=(test_names,TEST))
    # import model data
    md = ImageData(PATH, ds, bs, num_workers=nw, classes=None)
    #md.is_multi = False
    return md


# <h1><center>The Model</center></h1>

# <div style="text-align: left"> 
# The U-net model is composed of a ResNet34 based encoder and a simple upsampling decoder. Skip connections are added between encoder and decoder to facilitate the information flow at different detalization levels. Meanwhile, using a pretrained ResNet34 model allows us to have a powerful encoder capable of handling elaborated feature, in comparison with the original U-net, without a risk of overfitting and necessity of training a big model from scratch. Before using, the original ResNet34 model was further fine-tuned on ship/no-ship classification task.
# </div>

# In[11]:


#gives the meta data of the architecture - which layers to cut from resnet to do transfer learning
cut,lr_cut = model_meta[arch]


# <div style="text-align: left"> 
# Load ResNet34 model
# </div>

# In[12]:


#get the relevent layers from the pretrained model
def get_base():                  
    layers = cut_model(arch(True), cut)
    #nn.Sequential builds a neural net by specifying sequentially the building blocks of the net
    return nn.Sequential(*layers)

#load a model pretrained on ship/no-ship classification
def load_pretrained(model, path): 
    weights = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(weights, strict=False)     
    return model


# In[13]:


#implementation of unet blocks
class UnetBlock(nn.Module):
    #up_in - number of channels for upsampeling  
    #x_in - number of channels of the activations features from an intermediate layer of the encoder (the connection)
    #n_out - number of out channels
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out//2
        # set 1X1 convolution func for reducing the number of channels  
        self.x_conv  = nn.Conv2d(in_channels=x_in, out_channels=x_out, kernel_size=1)
        #set upsampling func - Applies a 2x2 transposed convolution operator over an input image
        self.tr_conv = nn.ConvTranspose2d(in_channels=up_in, out_channels=up_out, kernel_size=2, stride=2)
        #set batch normalization func
        self.bn = nn.BatchNorm2d(n_out)
    #up_p - downsampled input
    #x_p - the activation from the encoder
    def forward(self, up_p, x_p):
        #upsample
        up_p = self.tr_conv(up_p)
        #decrease channels from activation
        x_p = self.x_conv(x_p)
        #concat the upsampled with activation
        cat_p = torch.cat([up_p,x_p], dim=1)
        #return after relu
        return self.bn(F.relu(cat_p))

class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()

#the net   
class Unet34(nn.Module):
    def __init__(self, rn):
        super().__init__()
        #rn - the ResNet
        self.rn = rn
        self.sfs = [SaveFeatures(rn[i]) for i in [2,4,5,6]]
        self.up1 = UnetBlock(512,256,256)
        self.up2 = UnetBlock(256,128,256)
        self.up3 = UnetBlock(256,64,256)
        self.up4 = UnetBlock(256,64,256)
        self.up5 = nn.ConvTranspose2d(256, 1, 2, stride=2)
       
    def forward(self,x):
        x = F.relu(self.rn(x))
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        x = self.up5(x)
        return x[:,0]
    
    def close(self):
        for sf in self.sfs: sf.remove()
            
class UnetModel():
    def __init__(self,model,name='Unet'):
        self.model,self.name = model,name

    def get_layer_groups(self, precompute):
        lgs = list(split_by_idxs(children(self.model.rn), [lr_cut]))
        return lgs + [children(self.model)[1:]]


# <h1><center>Loss function</center></h1>

# In[14]:


def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


# In[15]:


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val +             ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.mean()


# In[16]:


class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)
        
    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target) - torch.log(dice_loss(input, target))
        return loss.mean()


# In[17]:


def dice(pred, targs):
    pred = (pred>0).float()
    return 2.0 * (pred*targs).sum() / ((pred+targs).sum() + 1.0)

def IoU(pred, targs):
    pred = (pred>0).float()
    intersection = (pred*targs).sum()
    return intersection / ((pred+targs).sum() - intersection + 1.0)


# <h1><center>Training</center></h1>

# In[18]:


m_base = load_pretrained(get_base(),PRETRAINED)
m = to_gpu(Unet34(m_base))
models = UnetModel(m)


# In[19]:


models.model


# <div style="text-align: left"> 
# Training (256x256)
# </div>

# In[20]:


sz = 256 #image size
bs = 64 #batch size

md = get_data(sz,bs)


# In[21]:


learn = ConvLearner(md, models)
learn.opt_fn=optim.Adam
learn.crit = MixedLoss(10.0, 2.0)
learn.metrics=[accuracy_thresh(0.5),dice,IoU]
wd=1e-7
lr = 1e-2


# In[22]:


learn.freeze_to(1)


# In[23]:


#learn.fit(lr,1,wds=wd,cycle_len=1,use_clr=(5,8))
learn.fit(lr,2,wds=wd,cycle_len=1,use_clr=(5,8))


# In[24]:


learn.save('Unet34_256_0')


# In[25]:


lrs = np.array([lr/100,lr/10,lr])
learn.unfreeze() #unfreeze the encoder
learn.bn_freeze(True)


# In[26]:


#learn.fit(lrs,2,wds=wd,cycle_len=1,use_clr=(20,8))
learn.fit(lrs,6,wds=wd,cycle_len=1,use_clr=(20,8))


# In[27]:


#learn.fit(lrs/3,2,wds=wd,cycle_len=2,use_clr=(20,8))
learn.fit(lrs/3,3,wds=wd,cycle_len=2,use_clr=(20,8))


# In[28]:


learn.sched.plot_lr()


# In[29]:


learn.save('Unet34_256_1')


# <div style="text-align: left"> 
# Visualization (256x256)
# </div>

# In[30]:


def Show_images100(x,yp,yt):
    columns = 3
    rows = min(bs,8)
    fig=plt.figure(figsize=(columns*4, rows*4))
    for i in range(rows):
        fig.add_subplot(rows, columns, 3*i+1)
        plt.axis('off')
        plt.imshow(x[i])
        fig.add_subplot(rows, columns, 3*i+2)
        plt.axis('off')
        plt.imshow(yp[i])
        fig.add_subplot(rows, columns, 3*i+3)
        plt.axis('off')
        plt.imshow(yt[i])
    plt.show()


# In[31]:


learn.model.eval();
x,y = next(iter(md.val_dl))
yp = to_np(F.sigmoid(learn.model(V(x))))


# In[32]:


Show_images(np.asarray(md.val_ds.denorm(x)), yp, y)


# <div style="text-align: left"> 
# Training (384x384)
# </div>

# In[33]:


sz = 384 #image size
bs = 16 #original 32  #batch size

md = get_data(sz,bs)
learn.set_data(md)
learn.unfreeze()
learn.bn_freeze(True)


# In[34]:


#learn.fit(lrs/5,1,wds=wd,cycle_len=2,use_clr=(10,8))
learn.fit(lrs/5,4,wds=wd,cycle_len=2,use_clr=(10,8))


# In[35]:


learn.save('Unet34_384_1')


# <div style="text-align: left"> 
# Visualization (384x384)
# </div>

# In[36]:


learn.model.eval();
x,y = next(iter(md.val_dl))
yp = to_np(F.sigmoid(learn.model(V(x))))


# In[37]:


Show_images(np.asarray(md.val_ds.denorm(x)), yp, y)


# <div style="text-align: left"> 
# Training (768x768)
# </div>

# In[38]:


sz = 768 #image size
bs = 6  #batch size

md = get_data(sz,bs)
learn.set_data(md)
learn.unfreeze()
learn.bn_freeze(True)


# In[39]:


#learn.fit(lrs/10,1,wds=wd,cycle_len=1,use_clr=(10,8))
learn.fit(lrs/10,5,wds=wd,cycle_len=1,use_clr=(10,8))


# In[40]:


learn.save('Unet34_768_1')


# <div style="text-align: left"> 
# Visualization (768x768)
# </div>

# In[41]:


learn.model.eval();
x,y = next(iter(md.val_dl))
yp = to_np(F.sigmoid(learn.model(V(x))))


# In[42]:


Show_images(np.asarray(md.val_ds.denorm(x)), yp, y)


# In[43]:


import torch
torch.cuda.empty_cache()

