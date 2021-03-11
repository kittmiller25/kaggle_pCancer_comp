# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 21:46:44 2021

@author: Kitt Miller
"""
# General packages
import os
import pandas as pd
import numpy as np
import zipfile
import pickle
from tqdm.notebook import tqdm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from tqdm.notebook import tqdm

# Image processing packages
from fastai.vision import *
import skimage.io
from PIL import Image
import openslide
from openslide import deepzoom
import cv2
import plotly.graph_objs as go

def plot_count(df, feature, title='', size=2):
    """Displays a plot of the total counts of one feature from the datafrane
    Inputs:
    df: dataframe holding the desired feature
    feature: feature to be counted
    title: resulting chart title
    
    Output:
    None
    """
    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))
    total = float(len(df))
    sns.countplot(df[feature],order = df[feature].value_counts().index, palette='Set2')
    plt.title(title)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 
    plt.show()
    
    
def plot_relative_distribution(df, feature, hue, title='', size=2, order = None,hue_order=None):
    """Displays a plot of the counts of one feature from the dataframe as seperately grouped by a second feature
    Inputs:
    df: dataframe holding the desired feature
    feature: feature to be counted
    hue: feature for grouping the counts by
    title: resulting chart title
    order: List specifying the order of values in the feature column as you would like them to be displayed
    hue_order: List specifying the order of values in the hue column as you would like them to be displayed
    
    Output:
    None
    """
    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))
    total = float(len(df))
    sns.countplot(x=feature, hue=hue, data=df, order = order,hue_order = hue_order, palette='Set2')
    plt.title(title)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 
    plt.show()

def open_slide_level(file_id, level=1):
    """ Opens .tiff file via openslide at a specified resolution level in one step
    Inputs:
    file_id: path to .tiff file to be opened
    level: resolution level at which to open the image, 0 = full resolution, 1 = 1/4 resolution, 2 = 1/16 resolution

    Output:
    An np array containing the 3-channel array respresenting the pixel values of the image
    """
    temp_img = openslide.open_slide(file_id)
    dims = temp_img.level_dimensions[level]
    temp_img_data = np.array(temp_img.read_region((0,0),level,(dims)))
    temp_img_data = rgba2rgb(temp_img_data)
    return temp_img_data

def rgba2rgb(rgba, background=(255,255,255)):
    """ Converts a 4-channel image array to a 3-channel image array
    Inputs:
    rbga: 4-channel array of image pixel data

    Output:
    An np array containing the 3-channel array respresenting the pixel values of the image
    """  
    row, col, ch = rgba.shape
    if ch == 3:
        return rgba
    assert ch == 4, 'RGBA image has 4 channels.'
    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]
    a = np.asarray( a, dtype='float32' ) / 255.0
    R, G, B = background
    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B
    return np.asarray( rgb, dtype='uint8' )

def img_mask_check(img_label,img_path,mask_path,data_df):
    """ Displays the actual image side-by-side with the mask in a labeled subplot
    Inputs:
    img_label: id associated with a sample from the training set ID list
    img_path: path to the directory where the images are located
    mask_path: path to the directory where the masks are located
    data_df: dataframe holding the training information with image IDs and associated scores

    Output:
    None
    """ 
    test_im_path = os.path.join(img_path,f'{img_label}.tiff')
    test_mask_path = os.path.join(mask_path,f'{img_label}_mask.tiff')
    print('Test image file: ',test_im_path)
    img_check = openslide.open_slide(test_im_path)
    dims = img_check.level_dimensions
    img_check.close()
    print('\nFull-size image dimensions: \t\t',dims[0],
          '\nOne-fourth size image dimensions: \t', dims[1], 
          '\nOne-sixteenth size image dimensions: \t',dims[2],
          '\n\nImage preview:')
    test_img  = open_slide_level(test_im_path, level=2)
    test_mask = open_slide_level(test_mask_path, level=2)
    cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])
    f, ax = plt.subplots(1,2, figsize=(10,6))
    ax[0].imshow(test_img)
    ax[1].imshow(test_mask[:,:,0], cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
    data_provider, isup_grade, gleason_score= data_df.loc[img_label]
    plt.suptitle(f"ID: {img_label}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score}")
    
    
def display_many_images(slides,img_path,data_df):
    """ Displays 15 labeled thumbnails of images
    Inputs:
    slides: List of 15 slide IDs to be displayed
    img_path: path to the directory where the images are located
    data_df: dataframe holding the training information with image IDs and associated scores

    Output:
    None
    """     
    f, ax = plt.subplots(5,3, figsize=(18,22))
    for i, slide in enumerate(slides):
        image = openslide.OpenSlide(os.path.join(img_path, f'{slide}.tiff'))
        spacing = 1 / (float(image.properties['tiff.XResolution']) / 10000)
        patch = image.read_region((1500,1700), 0, (512, 512))
        ax[i//3, i%3].imshow(patch) 
        image.close()       
        ax[i//3, i%3].axis('off')
        
        image_id = slide
        data_provider = data_df.loc[slide, 'data_provider']
        isup_grade = data_df.loc[slide, 'isup_grade']
        gleason_score = data_df.loc[slide, 'gleason_score']
        ax[i//3, i%3].set_title(f"ID: {image_id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score}")

    plt.show()
    
def display_many_masks(slides,mask_path,data_df):
    """ Displays 15 labeled thumbnails of the masks of the input images
    Inputs:
    slides: List of 15 slide IDs to be displayed
    mask_path: path to the directory where the masks are located
    data_df: dataframe holding the training information with image IDs and associated scores

    Output:
    None
    """  
    f, ax = plt.subplots(5,3, figsize=(18,22))
    for i, slide in enumerate(slides):
        
        mask = openslide.OpenSlide(os.path.join(mask_path, f'{slide}_mask.tiff'))
        mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])
        cmap = matplotlib.colors.ListedColormap(['white', 'gray', 'green', 'yellow', 'orange', 'red'])

        ax[i//3, i%3].imshow(np.asarray(mask_data)[:,:,0], cmap=cmap, interpolation='nearest', vmin=0, vmax=5) 
        mask.close()       
        ax[i//3, i%3].axis('off')
        
        image_id = slide
        data_provider = data_df.loc[slide, 'data_provider']
        isup_grade = data_df.loc[slide, 'isup_grade']
        gleason_score = data_df.loc[slide, 'gleason_score']
        ax[i//3, i%3].set_title(f"ID: {image_id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score}")
        f.tight_layout()
        
    plt.show()    
