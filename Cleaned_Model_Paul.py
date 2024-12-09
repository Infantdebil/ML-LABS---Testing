#!/usr/bin/env python
# coding: utf-8

# # Model `model_p` Extraction and Documentation
# This notebook contains the cleaned and annotated implementation of the `model_p` from Paul.
# 
# ### Objective
# To ensure the implementation is clear, concise, and well-documented, following the structure of a reference notebook.

# In[1]:


# Cleaned code for `model_p`:
from keras.backend import clear_session
clear_session()

from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D


# Define input shape for the VGG-like architecture
input_shape_value3 = (64, 64, 3)  # Input as per VGG guidelines
num_classes = 10  # Number of classes for classification

model_p = keras.Sequential()

# Block 1
model_p.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape_value3))
model_p.add(BatchNormalization())
model_p.add(Dropout(0.2))
model_p.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model_p.add(BatchNormalization())
model_p.add(MaxPooling2D(pool_size=(2, 2)))

# Block 2
model_p.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model_p.add(BatchNormalization())
model_p.add(Dropout(0.5))

# Block 3
model_p.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model_p.add(BatchNormalization())
model_p.add(MaxPooling2D(pool_size=(2, 2)))
model_p.add(Dropout(0.5))

# Block 4
model_p.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model_p.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model_p.add(Dropout(0.5))  # Higher dropout for even deeper layers


# Replace Flatten with GAP
model_p.add(GlobalAveragePooling2D())  # GAP for dimensionality reduction
model_p.add(Dense(128, activation='relu'))
model_p.add(Dropout(0.5))
model_p.add(Dense(num_classes, activation='softmax'))  # Output layer

model_p.summary()  # Print the architecture summary


# In[2]:


# Cleaned code for `model_p`:
# compile and train Paul's model

batch_size_value = 512
epochs_value = 150

# Advanced Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(x_train_resized)
train_generator = datagen.flow(x_train_resized, y_train, batch_size=32)


from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_accuracy',  # Stop when accuracy stops improving
    patience=10,              # Allow 5 epochs without improvement
    restore_best_weights=True
)
from keras.callbacks import ReduceLROnPlateau

# Learning Rate Scheduler
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',     # Monitor validation loss
    factor=0.5,             # Reduce learning rate by half
    patience=5,             # Wait 3 epochs before reducing the rate
    min_lr=1e-7             # Set a minimum learning rate
)

# optimizing class cats // we took off because of time issues
#from sklearn.utils.class_weight import compute_class_weight

#class_weights = compute_class_weight(
#    "balanced", classes=np.unique(y_train.argmax(axis=1)), y=y_train.argmax(axis=1)
#)
# class_weights_dict[cat_class_index] *= 0.8  # Decrease the weight for "cat"
#class_weights_dict = dict(enumerate(class_weights))
#print("Final Class Weights:", class_weights_dict)

optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001)

# Compile and Train Model with Scheduler
model_p.compile(
    optimizer=optimizer,  # Initial LR
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_p = model_p.fit(
    train_generator,
    epochs= epochs_value,  # Use higher max epochs, but early stopping and scheduler will manage it
    batch_size=batch_size_value,
    validation_data=(x_test_resized, y_test),
    callbacks=[reduce_lr, early_stopping],
    # class_weight=class_weights_dict
)


# In[ ]:


# Cleaned code for `model_p`:
# Visualisation Cross Entropy Loss and Accuracy

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 10))  # Set figure size for better clarity

# First subplot for loss
plt.subplot(211)
plt.title('Cross Entropy Loss - Model Aurele & Paul')
plt.plot(history_a.history['loss'], color='blue', label='Model Aurele')
# plt.plot(history_e.history['loss'], color='green', label='Model Enrique')
plt.plot(history_p.history['loss'], color='red', label='Model Paul')
plt.xlabel('Epochs')  # Add x-label for consistency
plt.ylabel('Loss')    # Add y-label
plt.legend()

# Second subplot for accuracy
plt.subplot(212)
plt.title('Classification Accuracy - Model Aurele & Paul')
plt.plot(history_a.history['accuracy'], color='blue', label='Model Aurele')
# plt.plot(history_e.history['accuracy'], color='green', label='Model Enrique')
plt.plot(history_p.history['accuracy'], color='red', label='Model Paul')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Adjust spacing between plots to prevent overlap
plt.subplots_adjust(hspace=0.5)

plt.show()

# Print results for all models
print("\033[1mModel Aurele Architecture:\033[0m")
print(history_a.history.keys())
score = model_a.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1], '\n')

# print("\033[1mModel #2 Architecture:\033[0m")
# print(history_e.history.keys())
# score2 = model_e.evaluate(x_test_resized, y_test, verbose=0)
# print("Test loss:", score2[0])
# print("Test accuracy:", score2[1], '\n')

print("\033[1mG3 Model Architecture:\033[0m")
print(history_p.history.keys())
score3 = model_p.evaluate(x_test_resized, y_test, verbose=0)
print("Test loss:", score3[0])
print("Test accuracy:", score3[1])


# In[ ]:


# Cleaned code for `model_p`:
# Comparison by Confiusion matrix

import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Predictions and ground truth for Model #1
predictions_a = model_a.predict(x_test)
predictions_a = np.argmax(predictions_a, axis=1)
gt_a = np.argmax(y_test, axis=1)
cm_a = confusion_matrix(gt_a, predictions_a)

# Resize x_test to match the input size for Model #3
x_test_resized = tf.image.resize(x_test, (64, 64), method=tf.image.ResizeMethod.BICUBIC)  # Assuming Model #3 was trained on 64x64 images

# Predictions and ground truth for Model #3
predictions_p = model_p.predict(x_test_resized)  # Use resized images for Model #3
predictions_p = np.argmax(predictions_p, axis=1)
gt_p = np.argmax(y_test, axis=1)
cm_p = confusion_matrix(gt_p, predictions_p)

# Create subplots for 2 models
fig, axs = plt.subplots(1, 2, figsize=(14, 5))  # 1 row, 2 columns

# Confusion matrix for Model #1
sns.heatmap(cm_a, annot=True, fmt='d', cmap='hot', xticklabels=class_names, yticklabels=class_names, ax=axs[0])
axs[0].set_title('Confusion Matrix Model #1')
axs[0].set_xlabel('Predicted Labels')
axs[0].set_ylabel('True Labels')

# Confusion matrix for Model #3
sns.heatmap(cm_p, annot=True, fmt='d', cmap='hot', xticklabels=class_names, yticklabels=class_names, ax=axs[1])
axs[1].set_title('Confusion Matrix Model #3')
axs[1].set_xlabel('Predicted Labels')
axs[1].set_ylabel('True Labels')

# Adjust layout for better spacing
plt.tight_layout()
plt.show()

