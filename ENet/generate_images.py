from ENet import enet, train_utils
from keras.preprocessing.image import load_img, img_to_array, array_to_img
import numpy as np
import random
import os


def create_new_sample(leaf_img, ann_img, back_img, raw_dir, ann_dir, leaf_file, back_file):
	mask = train_utils.get_image_mask_arr(ann_img)
	mask = train_utils.smooth_mask(mask)

	extracted_leaf = train_utils.extract_leaf(img_to_array(leaf_img), mask)
	pasted_arr = np.where(extracted_leaf != 255, extracted_leaf, img_to_array(back_img))
	ext = '.jpg'
	leaf_fname = os.path.basename(leaf_file).split('.')[0]
	back_fname = os.path.basename(back_file).split('.')[0]
	dest_fname = leaf_fname + '_' + back_fname + ext
	raw_path = os.path.join(raw_dir, dest_fname)
	array_to_img(pasted_arr).save(raw_path)

	pasted_arr[:,:,0][mask] = 0
	pasted_arr[:,:,1][mask] = 0
	pasted_arr[:,:,2][mask] = 254

	ann_path = os.path.join(ann_dir, dest_fname)
	array_to_img(pasted_arr).save(ann_path)

	return True

def populate_all_backgrounds():
	annotated_dir = 'data/enet_data/labeled/'
	original_dir = 'data/goodbadcombined/clear_leaves/'
	backgrounds = 'data/enet_data/rand_backgrounds/'
	paste_dir = 'data/enet_data/paste_leaves/'
	gen_label = 'data/enet_data/gen_labeled'
	gen_og = 'data/enet_data/gen_original'
	target_size = (256, 256)

	
	ann_paths = [fname for fname in os.listdir(paste_dir) if fname != '.DS_Store']
	background_paths = [os.path.join(backgrounds, fname) for fname in os.listdir(backgrounds) if fname != '.DS_Store']
	for leaf_fname in ann_paths:
		for back_path in background_paths:

			leaf_path = os.path.join(original_dir, leaf_fname)
			ann_path = os.path.join(paste_dir, leaf_fname)

			leaf_img = load_img(leaf_path, target_size=target_size)
			back_img = load_img(back_path, target_size=target_size)
			ann_img = load_img(ann_path, target_size=target_size)

			random_val = random.randint(0,3)

			# Skip half of all backgrounds
			if random_val % 2 == 0:
				leaf_img = leaf_img.rotate(random_val*90)
				ann_img = ann_img.rotate(random_val*90)
				create_new_sample(leaf_img, ann_img, back_img, gen_og, gen_label, leaf_path, back_path)

	return True

populate_all_backgrounds()