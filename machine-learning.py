# Import requirements for the machine learning
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.utils.vis_utils import plot_model
from collections import Counter
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt

# Defining the image height and width
img_width, img_height = 640, 480

# Defining the location of the training data
train_data_dir = 'modelData/training'

# Defining the parameters for the machine learning model
epochs = 250
batch_size = 16
start_epoch = 0
nb_classes = 5

#Defining the input shape
input_shape = (img_width, img_height, 3)

# Generating the dataset using Keras, with a validation data split of 30% and greyscaling all the images (Converting colour ranges from 0 - 255 o 0 - 1)
train_datagen = ImageDataGenerator(validation_split=0.3, rescale=1. / 255)

# Defining the layers of the model - These layers were experimentally added to optimise the machine learning model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.8))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# Defining the optimiser and learning rate
opt = SGD(learning_rate=0.001)

# Compiling the model with the chosen loss function and metrics
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.summary()

# Creating the training generator
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

# Creating the validation generator
validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

# Weighting the data set due to the un-even distribution of the dataset
counter = Counter(train_generator.classes)
max_val = float(max(counter.values()))

class_weights = {class_id: max_val / num_images for class_id, num_images in counter.items()}

test_datagen = ImageDataGenerator(rescale=1. / 255)

# Conducting the training of the model with the specified paramaters
result = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    class_weight=class_weights,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    verbose=1,
    callbacks=[],
    initial_epoch=start_epoch
)

# Output the model for visulisation
plot_model(model, show_shapes=True)

# Save the model & the weights for future reference and testing
model.save('modelData/models/model.h5')
model.save_weights('modelData/models/modelWeights.h5')

# Graph visulisation
plt.plot(result.history['accuracy'])
plt.plot(result.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()

plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
