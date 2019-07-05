from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from sklearn.feature_extraction.image import extract_patches_2d
import numpy as np 
import os
import glob
import skimage.io as io
import random
import skimage.transform as trans


Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

'''
def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)
'''

def adjustData(x, y):
    x = x/255
    y = y/255
    return (x, y)
    


def trainGenerator(batch_size,train_path,x_folder,y_folder,aug_dict,x_color_mode = "rgb",
                    y_color_mode = "rgb",x_save_prefix  = "blur",y_save_prefix  = "org",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,save_format='jpg',x_target_size = (572, 572), y_target_size = (572, 572),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    x_datagen = ImageDataGenerator(**aug_dict)
    y_datagen = ImageDataGenerator(**aug_dict)
    x_generator = x_datagen.flow_from_directory(
        train_path,
        classes = [x_folder],
        class_mode = None,
        color_mode = x_color_mode,
        target_size = x_target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = x_save_prefix,
        save_format = save_format,
        seed = seed)
    y_generator = y_datagen.flow_from_directory(
        train_path,
        classes = [y_folder],
        class_mode = None,
        color_mode = y_color_mode,
        target_size = y_target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = y_save_prefix,
        save_format = save_format,
        seed = seed)
    train_generator = zip(x_generator, y_generator)
    for (x, y) in train_generator:
        x, y = adjustData(x, y)
        yield (x, y)



def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.jpg"%i),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img

'''
def geneTrainNpy(x_path,y_path,flag_multi_class = False,num_class = 2,x_prefix = "blur",y_prefix = "org",x_as_gray = False,y_as_gray = False):
    x_name_arr = glob.glob(os.path.join(x_path,"%s*.jpg"%x_prefix))
    x_arr = []
    y_arr = []
    for index,item in enumerate(x_name_arr):
        x = io.imread(item,as_gray = x_as_gray)
        y = io.imread(item.replace(x_path,y_path).replace(x_prefix,y_prefix),as_gray = y_as_gray)
        x,y = adjustData(x,y)
        x_arr.append(x)
        y_arr.append(y)
    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)
    return x_arr,y_arr
'''
def geneTrainNpy(x_path,y_path,x_prefix = "blur",y_prefix = "org",x_as_gray = False,y_as_gray = False):
    x_name_arr = glob.glob(os.path.join(x_path,"%s*.jpg"%x_prefix))
    x_arr = []
    y_arr = []
    for index,item in enumerate(x_name_arr):    
        x = io.imread(item) #array
        y = io.imread(item.replace(x_path,y_path).replace(x_prefix,y_prefix))
        x_patches = extract_patches_2d(x, (80, 80))
        y_patches = extract_patches_2d(y, (80, 80))
        size = x_patches.shape[0]
        res = [random.randint(0, size - 1) for i in range(70)]
        x_selected = x_patches[res, :, :, :]
        y_selected = y_patches[res, :, :, :]
        x_selected = x_selected / 255
        y_selected = y_selected / 255
        if(index == 0):
            x_arr = x_selected
            y_arr = y_selected
        else:
            x_arr = np.append(x_arr, x_selected, axis = 0)
            y_arr = np.append(y_arr, y_selected, axis = 0)
        print(x_arr.shape)
    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)
    return x_arr,y_arr

def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255



def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,:]
        io.imsave(os.path.join(save_path,"%d_predict.jpg"%i),img)