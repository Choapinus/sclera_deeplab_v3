import os
import sys

sys.path.append(os.path.abspath(os.path.join('..')))

import cv2
import numpy as np
import seaborn as sns
from metrics import *
import tensorflow as tf
from model import Deeplabv3
from utils import ScleraDataset
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from imgaug import augmenters as iaa # https://github.com/aleju/imgaug (pip3 install imgaug)

fix_cuda = False

if fix_cuda:
	from keras.backend.tensorflow_backend import set_session

	config = tf.ConfigProto()
	# dynamically grow GPU memory
	config.gpu_options.allow_growth = True
	set_session(tf.Session(config=config))

mode = 'constant'

aug = iaa.SomeOf(
	(0, 12), [
        iaa.Sometimes(0.5, iaa.Fliplr(1)),
        iaa.Sometimes(0.5, iaa.Flipud(1)),
		iaa.Sometimes(0.5, [
			iaa.Affine(rotate=(-25, 25), mode=mode),
		]),
		iaa.Sometimes(0.5, [
			iaa.Affine(
                scale=(0.75, 1.5), 
                mode=mode,
                translate_percent={
                    "x": (-0.5, 0.5), 
                    "y": (-0.5, 0.5),
                }
            ),
		]),
		iaa.Sometimes(0.35, [
			iaa.CoarseDropout(p=(0.05, 0.25)),
		]),
		iaa.SomeOf((0, 1), [
			iaa.Add(25),
			iaa.Add(-25),
		]),
		iaa.SomeOf((0, 2), [
			iaa.AdditiveGaussianNoise(0.05*255),
			iaa.AdditiveLaplaceNoise(0.05*255),
			iaa.AdditivePoissonNoise(16.0),
		]),
        iaa.SomeOf((0, 2), [
            iaa.GaussianBlur(sigma=0.5),
			iaa.AverageBlur(k=2),
			iaa.MotionBlur(k=3),
			iaa.GammaContrast(gamma=0.5),
            iaa.pillike.EnhanceSharpness(),
            iaa.imgcorruptlike.GaussianBlur(severity=(1, 2)),
            iaa.imgcorruptlike.GlassBlur(severity=(1, 2)),
            iaa.imgcorruptlike.DefocusBlur(severity=(1, 2)),
            iaa.imgcorruptlike.MotionBlur(severity=(1, 2)),
            # iaa.imgcorruptlike.ZoomBlur(severity=(1, 2)),
        ]),
        iaa.SomeOf((0, 1), [
            iaa.imgcorruptlike.Frost(severity=(1, 2)),
            iaa.imgcorruptlike.Spatter(severity=(1, 2)),
            iaa.Snowflakes(flake_size=(0.01, 0.2), speed=(0.01, 0.05)),
            iaa.Rain(speed=(0.01, 0.1)) 
        ]),
        iaa.SomeOf((0, 2), [
            iaa.PerspectiveTransform(scale=(0.01, 0.1), keep_size=True),
            iaa.PiecewiseAffine(scale=(0.01, 0.05)),
            iaa.ElasticTransformation(alpha=(0, 2.5), sigma=0.25)
        ])
	], random_state=True
)

OS = 16
os.makedirs('models', exist_ok=True)

print('[INFO] Loading train set')
trainG = ScleraDataset(batch_size=1, dim=(224, 224), augmentation=aug)
trainG.load_eyes('dataset_v1', "train")
trainG.prepare()

print('[INFO] Loading val set')
valG = ScleraDataset(batch_size=1, dim=(224, 224))
valG.load_eyes('dataset_v1', "val")
valG.prepare()


model = Deeplabv3(input_shape=trainG.input_shape, OS=OS, classes=trainG.num_classes, weights=None)
model.summary()

model.compile(
	optimizer=RMSprop(lr=1e-4, decay=1e-4),
	loss='categorical_crossentropy',
	metrics=[mean_iou(valG.num_classes)], 
)

mckpt = keras.callbacks.ModelCheckpoint(
	filepath='models/epoch_{epoch:03d}_miou_{val_mean_iou:.4f}.h5',
	monitor='val_mean_iou', verbose=1, save_best_only=True, mode='max'
)

tsboard = keras.callbacks.TensorBoard(
	write_images=True, write_graph=True,
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
	monitor='val_mean_iou', factor=0.1, patience=10,
	verbose=1, mode='max', min_lr=1e-10
)

early_stop = keras.callbacks.EarlyStopping(
	monitor='val_mean_iou', patience=20,
	mode='max', 
	# restore_best_weights=True,
)


hist = model.fit_generator(
	trainG, validation_data=valG, epochs=200, verbose=1,
	callbacks=[reduce_lr, mckpt, tsboard, early_stop],
	workers=6,
)

hist = hist.history

plt.title(f'DeepLabV3+ Training Loss & IoU OS={OS}')
plt.xlabel('Epochs')
plt.ylabel('Loss/IoU')
plt.plot(hist['loss'])
plt.plot(hist['val_loss'])
plt.plot(hist['mean_iou'])
plt.plot(hist['val_mean_iou'])

plt.legend(['loss', 'val_loss', 'mean_iou', 'val_mean_iou'])
plt.savefig(f'loss_iou_os_{OS}.png')