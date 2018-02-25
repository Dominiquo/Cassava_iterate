import ENet
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.utils import to_categorical
from scipy.ndimage.filters import convolve
import numpy as np
import os


def get_image_mask_path(annotated_path, target_size):
	'''
	get the mask of the image given the path of the file

	annotated_path: path to the image with labeled pixels according to the established protocol (0,0,255)
	target_size: tuple (int,int) representing the target size of the image dimensions.
	'''
	ann_arr = load_img(annotated_path, target_size=target_size)
	# target_size shaped array with True where pixel is annotated as a leaf
	return get_image_mask_arr(ann_arr)


def get_image_mask_arr(ann_img):
	'''
	returns a mask of the same dimensions as the input image but where the red_channel
	is equal to zero. I made where red_channel == 0 rather than blue_channel == 255
	because I did some checking on the annotated images and some of the blue pixels were 
	254 or 255 and red_channel == 0 followed the same logic but eliminated variability

	ann_img: image object from tensorflow's image library
	'''
	ann_arr = img_to_array(ann_img)
	red_channel = ann_arr[:,:,0]
	return red_channel == 0


def get_image_label(annotated_path, target_size, classes):
	'''
	takes an images and produces a matrix for each pixel is given a class value by checking the mask

	annotated_path: path to the image with labeled pixels according to the established protocol (0,0,255)
	pixels for category 'leaf'
	target_size: tuple (int,int) representing the target size of the image dimensions.
	classes: number of distince classes that will be identified by the pixel labels.

	'''
	mask = get_image_mask_path(annotated_path, target_size) #size target_size array of True/False values
	int_mask = mask.astype(int) #convert mask to have 0/1 instead of False/True

	# to_categorical is a built in keras function to take a integer, n, to an array of len(v) == classes 
	# where the nth value is set to 1 and the rest to 0. Same logic is applied to an entire matrix here
	# e.g. N = 2 classes = 4 -- > [0., 0., 1.0, 0.]

	return to_categorical(int_mask, classes)


def get_training_image(original_path, target_size):
	'''
	Use the tensorflow library to take an image directly from file to 
	array 

	original_path: path to image that will be made into an array as is
	target_size: tuple (int,int) representing the target size of the image dimensions. 
	'''
	return img_to_array(load_img(original_path, target_size=target_size))	


def get_samples(original_dir, annotated_dir, target_size, classes=2):
	'''
	IMAGES IN ORIGNAL DIRECTORY MUST HAVE A MATCHING FILENAME IMAGE IN THE LABELED
		E.G original/disease_image_1.jpg needs a labeled/disease_image_1.jpg

	original_dir: String representing the path to the directory of original, unlabeled images
    annotated_dir: String representing the path to the directory of labeled images
    target_size: tuple (int,int) representing the target size of the image dimensions. 
    classes: integer > 1; the number of possible classifications for the pixels in the image
	'''

	X = []
	Y = []
	masks = []
	all_original_files = set(os.listdir(original_dir))
	for fname in os.listdir(annotated_dir): # iteratte through all of the labeled images
		if fname in all_original_files: #check that there is an existing partner in the og images
			original_path = os.path.join(original_dir, fname) 
			annotated_path = os.path.join(annotated_dir, fname)

			# append the appropriate type of vector for X, Y, Mask
			X.append(get_training_image(original_path, target_size))
			Y.append(get_image_label(annotated_path, target_size, classes))
			masks.append(get_image_mask_path(annotated_path, target_size))
	return np.array(X), np.array(Y), np.array(masks)
			

def extract_leaf(X, mask):
	'''
	White everything else out except the leaf based on the 
	mask 
	'''
	X_copy = np.copy(X)
	mask = np.logical_not(mask)
	X_copy[:,:,0][mask] = 255
	X_copy[:,:,1][mask] = 255
	X_copy[:,:,2][mask] = 255
	return X_copy


def paste_leaf(leaf_img, annotate_img,  background_img):
	# all images MUST be the same size
	mask = get_image_mask_arr(annotate_img)
	leaf_arr = extract_leaf(img_to_array(leaf_img), mask)
	back_arr = img_to_array(background_img)
	return np.where(leaf_arr != 255, leaf_arr, back_arr)


def smooth_mask(mask, self_weight=0, size=11, threshold=.05):
	'''
	more research can be done on optimizing this, but this gives the ability for the user
	to test different kernel techniques to fill in pixels that should be filled and erase others

	'''
	center = size/2
	kernal_area = size*size
	k = np.ones((size,size))
	center_val = kernal_area
	k[center][center] = center_val
	inp = mask.astype(int)
	zero_mask = np.zeros(mask.shape)
	one_mask = np.ones(mask.shape)
	smoothed = convolve(inp, k)
	threshold_val = kernal_area*self_weight + (kernal_area * threshold) - 1 
	return np.where(smoothed > threshold_val, one_mask, zero_mask).astype(bool)

