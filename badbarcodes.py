import os
import numpy as np
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

###################################################
# Prepares a dataset of malware images, trains a
# Convolutional Neural Network (CNN) using 
# TensorFlow/Keras to classify them, and saves 
# the trained model along with class metadata.
# It processes the Malimg dataset (a collection
# of malware grayscale images organized by family),
# splits it into training and test sets, applies data 
# augmentation, trains a binary classifier despite
#  detecting multiple classes, and evaluates its performance
#######################################################

dataset_path = r""
output_dir = r""
save_dir = r""

classes = [folder for folder in os.listdir(dataset_path) 
          if os.path.isdir(os.path.join(dataset_path, folder))]

print(f"Detected {len(classes)} malware families: {classes}")

for split in ['train', 'test']:
    for cls in classes:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

for cls in classes:
    class_path = os.path.join(dataset_path, cls)
    images = [f for f in os.listdir(class_path) if f.lower().endswith('.png')]
    train_files, test_files = train_test_split(images, test_size=0.2, random_state=42)
    
    for f in train_files:
        shutil.copyfile(os.path.join(class_path, f), 
                       os.path.join(output_dir, 'train', cls, f))
    for f in test_files:
        shutil.copyfile(os.path.join(class_path, f), 
                       os.path.join(output_dir, 'test', cls, f))

img_size = (32, 32)
batch_size = 64

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(output_dir, 'train'),
    target_size=img_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    os.path.join(output_dir, 'test'),
    target_size=img_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary')

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])

model.fit(train_generator,
         epochs=20,
         validation_data=test_generator)

test_loss, test_acc = model.evaluate(test_generator)
print(f'\nTest accuracy: {test_acc:.4f}')

model.save(os.path.join(save_dir, 'malware_classifier.h5'))

import json
class_names = list(train_generator.class_indices.keys())
with open(os.path.join(save_dir, 'class_names.json'), 'w') as f:
    json.dump(class_names, f)