from ENet import enet, train_utils
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def train_val_generators(original_dir, annotated_dir, target_size=(256,256), classes=2, batch_size=16, train_size=.8):
        '''
        Developer's note: This can be updated to give more flexibility in how the images are produced by the generators.
        Right now, I'm just using the ImageDataGenerator class from tensorflow to make the images more digestable. This 
        means that the augmentations are hardcoded into the model. The generators don't mess with the images prior
        to going through the model save horizontal_flip/vertical_flip. It's possible to update this for training to
        'produce' more data based on rotation, shearing, zooming, etc. 



        original_dir: String representing the path to the directory of original, unlabeled images
        annotated_dir: String representing the path to the directory of labeled images
        target_size: tuple (int,int) representing the target size of the image dimensions. 
                        increase size --> longer training time
                        increase size --> higher accuracy when applying other models

        classes: integer > 1; the number of possible classifications for the pixels in the image
        batch_size: size of set the generator will produce dim: (batch_size, target_size[0], target_size[1], n_channels)
        train_size: real number [0,1.0]; the ratio of images to be used for training. The complement will be used 
                for training
        '''
	random_state = 42
	X,y,masks = train_utils.get_samples(original_dir, annotated_dir, target_size, classes=2)

	traingen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0.,
        width_shift_range=0.,
        height_shift_range=0.,
        shear_range=0.,
        zoom_range=0.,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=True)

	valgen = ImageDataGenerator(featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0.,
        width_shift_range=0.,
        height_shift_range=0.,
        shear_range=0.,
        zoom_range=0.,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=True)

        if train_size < 1:
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)
                return traingen.flow(X_train,y_train, batch_size=batch_size, shuffle=True), valgen.flow(X_test,y_test, batch_size=batch_size, shuffle=True)
        else:
                return traingen.flow(X,y, batch_size=batch_size, shuffle=True),  None



