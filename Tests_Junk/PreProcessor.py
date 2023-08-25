import os
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator

# Define the path to the Sketchy dataset
data_dir = 'sketches_png/png'

# Define the ImageDataGenerator with the desired preprocessing parameters
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

# Generate the preprocessed training and validation datasets
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

print('1')
# Save the preprocessed datasets as numpy arrays
x_train = np.concatenate([train_generator.next()[0] for i in range(train_generator.__len__())])
print('2')
y_train = np.concatenate([train_generator.next()[1] for i in range(train_generator.__len__())])
print('3')
x_val = np.concatenate([validation_generator.next()[0] for i in range(validation_generator.__len__())])
print('4')
y_val = np.concatenate([validation_generator.next()[1] for i in range(validation_generator.__len__())])


print(x_train)


np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)
np.save('x_val.npy', x_val)
np.save('y_val.npy', y_val)
