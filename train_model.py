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

from sklearn.metrics import classification_report, confusion_matrix
import pickle
import itertools

##
data_path = 'C:/git_zhaw/catnocat/data'

train_data_path = os.path.join(data_path, 'train')
valid_data_path = os.path.join(data_path, 'valid')
# holytest_data_path = os.path.join(data_path, 'holytest')

##
image_size = (224, 224)
batch_size = 32

##
# Set Data Generator for training, testing and validataion.

# Note for testing, set shuffle = false (For proper Confusion matrix)
train_datagen = ImageDataGenerator(rescale=1 / 255)
train_dataset = train_datagen.flow_from_directory(train_data_path,
                                                  target_size=image_size,
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  shuffle=True)

valid_datagen = ImageDataGenerator(rescale=1 / 255)
valid_dataset = valid_datagen.flow_from_directory(valid_data_path,
                                                  target_size=image_size,
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  shuffle=False)

# test_datagen = ImageDataGenerator(rescale=1/255)
# test_dataset = test_datagen.flow_from_directory(holytest_data_path,
#                                                 target_size=image_size,
#                                                 batch_size=batch_size,
#                                                 class_mode='categorical',
#                                                 shuffle=False)


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
epochs = 1
step_size_train = train_dataset.n // train_dataset.batch_size
history = model.fit(train_dataset,
                    steps_per_epoch=step_size_train,
                    epochs=epochs,
                    validation_data=valid_dataset,
                    # validation_steps=step_size_train,
                    )

filename = 'trained_model.pickle'
outfile = open(filename, 'wb')
pickle.dump(history, outfile)
outfile.close()

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

Y_pred = model.predict_generator(valid_dataset, step_size_train + 1)
y_pred = np.argmax(Y_pred, axis=1)

##
# Confusion Matrix and Classification Report

print('Confusion Matrix')
print(confusion_matrix(valid_dataset.classes, y_pred))
print('Classification Report')
target_names = ['cat', 'no cat']
print(classification_report(valid_dataset.classes, y_pred, target_names=target_names))


##
# Plot the confusion matrix. Set Normalize = True/False
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


##
# Confusion Matrix
print('Confusion Matrix')
cm = confusion_matrix(valid_dataset.classes, y_pred)
plot_confusion_matrix(cm, target_names, title='Confusion Matrix')

# Print Classification Report

print('Classification Report')
print(classification_report(valid_dataset.classes, y_pred, target_names=target_names))

##
# Save the model
model.save("tutorial.hdf5")

##
x, y = next(valid_dataset)
# x,y = valid_dataset[0]

img_array = keras.preprocessing.image.img_to_array(valid_dataset)
img_array_expanded = np.expand_dims(img_array, axis=0)  # Create a batch

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
valid_dataset.reset()  # reset the generator to the first batch

##
x, y = next(valid_dataset)
print("Batch Index: ", valid_dataset.batch_index)
print("x: ", x.shape)
print("y: ", y.shape)

ground_truth = valid_dataset.classes
predictions = model.predict_generator(valid_dataset)
prediction_table = {}
for index, val in enumerate(predictions):
    index_of_highest_probability = np.argmax(val)
    value_of_highest_probability = val[index_of_highest_probability]
    prediction_table[index] = [value_of_highest_probability, index_of_highest_probability, ground_truth[index]]
    assert len(predictions) == len(ground_truth) == len(prediction_table)

plt.figure(figsize=(20, 10))
for i in range(32):
    ax = plt.subplot(4, 8, i + 1)
    plt.imshow(x[i])
    plt.title("{}, {}".format(class_names[np.argmax(y[i])], "BLA"))
    plt.axis("off")

plt.show()


##

img_array = keras.preprocessing.image.img_to_array(x)
img_array_expanded = np.expand_dims(img_array, axis=0)  # Create a batch

predictions = model.predict(img_array_expanded)
score = tf.nn.softmax(predictions[0])

plt.imshow(img_array)
plt.title("{}, with {:.2f}% confidence".format(class_names[np.argmax(score)].upper(), 100 * np.max(score)))
plt.show()

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
)
