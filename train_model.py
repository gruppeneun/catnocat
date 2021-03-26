##
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf


##
data_path = 'C:/git_zhaw/catnocat/data'

trainvalid_path = os.path.join(data_path, 'trainvalid')
holytest_path = os.path.join(data_path, 'holytest')

##
image_size = (224, 224)
batch_size = 32

##
image_generator = ImageDataGenerator(rescale=1 / 255,
                                     validation_split=0.2)

train_dataset = image_generator.flow_from_directory(batch_size=batch_size,
                                                    directory=trainvalid_path,
                                                    shuffle=True,
                                                    target_size=image_size,
                                                    subset="training",
                                                    class_mode='categorical',
                                                    seed=42,
                                                    )

valid_dataset = image_generator.flow_from_directory(batch_size=batch_size,
                                                    directory=trainvalid_path,
                                                    shuffle=True,
                                                    target_size=image_size,
                                                    subset="validation",
                                                    class_mode='categorical',
                                                    seed=42,
                                                    )
##
# # Lets now use MobileNet as it is quite lightweight (17Mb), freeze the base layers and lets add and train the top
# # few layers. Note only two classifiers.

base_model = MobileNet(weights='imagenet',
                       include_top=False)  # imports the mobilenet model and discards the last 1000 neuron layer.

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(
    x)  # we add dense layers so that the model can learn more complex functions and classify for better results.
x = Dense(1024, activation='relu')(x)  # dense layer 2
x = Dense(512, activation='relu')(x)  # dense layer 3
preds = Dense(2, activation='softmax')(x)  # final layer with softmax activation

model = Model(inputs=base_model.input, outputs=preds)
# specify the inputs
# specify the outputs
# now a model has been created based on our architecture

##
# check the architecture
for i, layer in enumerate(model.layers):
    print(i, layer.name)

##
# We will use pre-trained weights as the model has been trained already on the Imagenet dataset. We ensure all the
# weights are non-trainable. We will only train the last few dense layers.

for layer in model.layers:
    layer.trainable = False
# or if we want to set the first n layers of the network to be non-trainable
n = 79
for layer in model.layers[:n]:
    layer.trainable = False
for layer in model.layers[n:]:
    layer.trainable = True


##
# Compile and train the model.

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

model.summary()

##
epochs = 10
step_size_train = train_dataset.n // train_dataset.batch_size
history = model.fit(train_dataset,
                    validation_data=valid_dataset,
                    steps_per_epoch=step_size_train,
                    epochs=epochs,
                    )

##

# list all data in history
print(history.history.keys())

##
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


##
# PREDICTION
class_names = list(valid_dataset.class_indices.keys())
print(class_names)

x, y = next(valid_dataset)

img_array = keras.preprocessing.image.img_to_array(x[0])
img_array_expanded = np.expand_dims(img_array, axis=0) # Create a batch

predictions = model.predict(img_array_expanded)
score = tf.nn.softmax(predictions[0])

plt.imshow(img_array)
plt.title("{}, with {:.2f}% confidence".format(class_names[np.argmax(score)].upper(), 100 * np.max(score)))
plt.show()

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)



##

