#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import os
import random
import matplotlib.pyplot as plt
from glob import glob


# In[3]:


os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/')


# In[4]:


import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Lambda, Dense, Flatten

from keras.models import Sequential
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model


# In[5]:


train_path = '../input/chest-xray-pneumonia/chest_xray/train/'


# In[6]:


test_path = '../input/chest-xray-pneumonia/chest_xray/test/'


# In[7]:


val_path = '../input/chest-xray-pneumonia/chest_xray/val/'


# In[8]:


h=7
w=7
fig, axes = plt.subplots(w,h, figsize = (15,15))

axes = axes.ravel() # flaten the matrix into array
for i in np.arange(0, w * h): 
    label ='NORMAL'
    class_dir = os.path.join(train_path,label)
    # Select a random image
    image = random.choice(os.listdir(class_dir))
    # read and display an image with the selected index    
    img = plt.imread(os.path.join(class_dir,image))
    axes[i].imshow( img )
    axes[i].set_title(label, fontsize = 8) # the label
    axes[i].axis('off')


# In[9]:


h=7
w=7
fig, axes = plt.subplots(w,h, figsize = (15,15))

axes = axes.ravel() # flaten the matrix into array
for i in np.arange(0, w * h): 
    label ='PNEUMONIA'
    class_dir = os.path.join(train_path,label)
    # Select a random image
    image = random.choice(os.listdir(class_dir))
    # read and display an image with the selected index    
    img = plt.imread(os.path.join(class_dir,image))
    axes[i].imshow( img )
    axes[i].set_title(label, fontsize = 8) # the label
    axes[i].axis('off')


# In[ ]:





# In[10]:


print(list(len(os.listdir(os.path.join(train_path,os.listdir(train_path)[x]))) for x in range(0,2)))


# In[11]:


print(list(len(os.listdir(os.path.join(test_path,os.listdir(test_path)[x]))) for x in range(0,2)))


# In[12]:


print(list(len(os.listdir(os.path.join(val_path,os.listdir(val_path)[x]))) for x in range(0,2)))


# In[13]:


def data_generators(TRAINING_DIR, VALIDATION_DIR):

      train_datagen = ImageDataGenerator(rescale=1./255,
                                         rotation_range=0.5,
                                         zoom_range=0.3,
                                         height_shift_range=0.3,
                                         width_shift_range=0.3,
                                         shear_range=0.3,
                                         horizontal_flip=True,
                                         fill_mode='nearest')

      train_generator = train_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                          batch_size=32,
                                                          class_mode='binary',
                                                          target_size=(224, 224))

      validation_datagen = ImageDataGenerator(rescale=1./255)

      validation_generator = validation_datagen.flow_from_directory(directory=VALIDATION_DIR,
                                                                    batch_size=32,
                                                                    class_mode='binary',
                                                                    target_size=(224, 224))
      return train_generator, validation_generator


# In[14]:


train_generator, validation_generator = data_generators(train_path, test_path)


# In[15]:


def chest_model():

          model = tf.keras.models.Sequential([ 
              tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(224,224,3)),
              tf.keras.layers.MaxPooling2D(2,2),
              tf.keras.layers.BatchNormalization(),
              tf.keras.layers.Dropout(0.3),

              tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
              tf.keras.layers.MaxPooling2D(2,2),
              tf.keras.layers.Dropout(0.3),

              tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
              tf.keras.layers.MaxPooling2D(2,2),
              tf.keras.layers.Dropout(0.3),

              tf.keras.layers.Flatten(),
              tf.keras.layers.Dense(512,activation='relu'),
              tf.keras.layers.Dropout(0.3),
              tf.keras.layers.Dense(128,activation='relu'),
              tf.keras.layers.Dense(2,activation='softmax')   # we used sigmoid also
          ])


          model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']) 
    

          return model


# In[16]:


model= chest_model()


# In[17]:


model.summary()


# In[18]:


from tensorflow.keras.utils import plot_model
plot_model(model, show_shapes=True)


# In[19]:


pip install visualkeras


# In[20]:


get_ipython().system('pip install visualkeras')


# In[21]:


import visualkeras


# In[22]:


visualkeras.layered_view(model)


# In[23]:


history = model.fit(train_generator,
                    epochs=30,
                    verbose=True,
                    validation_data=validation_generator,
                    steps_per_epoch= 100,
                    validation_steps= 10)


# In[24]:


acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.show()
print("")

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.show()


# In[25]:


train_evaluation = model.evaluate(train_generator)
print("\n Train loss: ", train_evaluation[0])
print("\n Train Accuracy: ", train_evaluation[1])


# In[26]:


test_evaluation= model.evaluate(validation_generator)
print("\nTest loss: ", test_evaluation[0])
print("Test Accuracy: ", test_evaluation[1])


# In[27]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf


# **Tesing a sample**

# In[28]:


from keras.preprocessing import image
test_img=image.load_img('../input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA/person1947_bacteria_4876.jpeg',target_size=(224,224,3))
test_img_arr=image.img_to_array(test_img)
test_img_arr=np.expand_dims(test_img_arr, axis=0)
model_classes=model.predict(test_img_arr)


# In[29]:


print("affected" if int(model_classes[0][0]) is 0 else " not affected")


# **VGG16**

# In[30]:


from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input


# In[31]:


vgg = VGG16(input_shape=[224,224] + [3], include_top=False,  weights='imagenet')
for layer in vgg.layers:
    layer.trainable = False


# In[32]:


folders = glob('../input/chest-xray-pneumonia/chest_xray/train/*')
x = Flatten()(vgg.output)


# In[33]:


prediction = Dense(len(folders), activation='softmax')(x)
# create a model object
VGG_model = Model(inputs=vgg.input, outputs=prediction)
# view the structure of the model
VGG_model.summary()


# In[34]:


from tensorflow.keras.utils import plot_model
plot_model(VGG_model, show_shapes=True)


# In[35]:


VGG_model.compile(
  loss='sparse_categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[36]:


vgg_history_1 = VGG_model.fit(train_generator,
                    epochs=10,
                    verbose=1,
                    validation_data=validation_generator,
                    steps_per_epoch=len(train_generator),
                    validation_steps=len(validation_generator))


# In[37]:


train_vgg = VGG_model.evaluate(train_generator)
print("\nTrain loss: ", train_vgg[0])
print("Train Accuracy: ", train_vgg[1])


# In[38]:


test_vgg = VGG_model.evaluate(validation_generator)
print("\nTest loss: ", test_vgg[0])
print("Test Accuracy: ", test_vgg[1])


# **RESNET50**

# In[39]:


from tensorflow.keras.applications import ResNet50
plot_model(ResNet50(), show_shapes=True)


# 

# In[40]:


def create_resnet_model():
    resnet_50 = Sequential()

    resnet_50.add(ResNet50(include_top = False, pooling = 'avg', weights = 'imagenet'))
    resnet_50.add(Dense(64, activation = 'relu'))
    resnet_50.add(Dense(2, activation = 'softmax'))

    # Say not to train first layer (ResNet) model as it is already trained
    resnet_50.layers[0].trainable = False

    return resnet_50


# In[41]:


resnet_50 = create_resnet_model()
resnet_50.summary()


# In[42]:


plot_model(resnet_50, show_shapes=True)


# In[43]:


resnet_50.compile(optimizer = keras.optimizers.Adam(learning_rate=0.001), 
                  loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True), 
                  metrics = ['accuracy'])


# In[44]:


resnet_history_1 = resnet_50.fit(
        train_generator,
        steps_per_epoch=50,
        epochs = 10,
        validation_data=validation_generator,
        validation_steps=20,
)


# In[45]:


acc_resnet_1=resnet_history_1.history['accuracy']
val_acc_resnet_1=resnet_history_1.history['val_accuracy']
loss_resnet_1=resnet_history_1.history['loss']
val_loss_resnet_1=resnet_history_1.history['val_loss']

epochs_resnet_1=range(len(acc_resnet_1)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs_resnet_1, acc_resnet_1, 'r', "Training Accuracy")
plt.plot(epochs_resnet_1, val_acc_resnet_1, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.show()
print("")

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs_resnet_1, loss_resnet_1, 'r', "Training Loss")
plt.plot(epochs_resnet_1, val_loss_resnet_1, 'b', "Validation Loss")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




