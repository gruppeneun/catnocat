##
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
import itertools
import matplotlib.image as mpimg

##
data_path = 'C:/git_zhaw/catnocat/data'

train_data_path = os.path.join(data_path, 'train')
valid_data_path = os.path.join(data_path, 'valid')
# holytest_data_path = os.path.join(data_path, 'holytest')

##
image_size = (224, 224)
batch_size = 32

##
# Set Data Generator for training, testing and validation.

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
epochs = 12
step_size_train = train_dataset.n // train_dataset.batch_size
history = model.fit(train_dataset,
                    steps_per_epoch=step_size_train,
                    epochs=epochs,
                    validation_data=valid_dataset,
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
model.save("cats_mobilenet.hdf5")

##
# PREDICTIONS 2
ground_truth = valid_dataset.classes
print(ground_truth[:10])
print(len(ground_truth))

##
# Then we get the predictions. This will be a list of probability values that express how confident
# the model is about the presence of each category in each image. This step might take several minutes.

predictions = model.predict_generator(valid_dataset,
                                      steps=None)
print(predictions[:10])

##
prediction_table = {}
for index, val in enumerate(predictions):
    index_of_highest_probability = np.argmax(val)
    value_of_highest_probability = val[index_of_highest_probability]
    prediction_table[index] = [
        value_of_highest_probability,
        index_of_highest_probability,
        ground_truth[index]
    ]
assert len(predictions) == len(ground_truth) == len(prediction_table)


##
def get_images_with_sorted_probabilities(prediction_table,
                                         get_highest_probability,
                                         label,
                                         number_of_items,
                                         only_false_predictions=False):
    sorted_prediction_table = [(k, prediction_table[k])
                               for k in sorted(prediction_table,
                                               key=prediction_table.get,
                                               reverse=get_highest_probability)
                               ]
    result = []
    for index, key in enumerate(sorted_prediction_table):
        image_index, [probability, predicted_index, gt] = key
        if predicted_index == label:
            if only_false_predictions == True:
                if predicted_index != gt:
                    result.append(
                        [image_index, [probability, predicted_index, gt]])
            else:
                result.append(
                    [image_index, [probability, predicted_index, gt]])
    return result[:number_of_items]


def plot_images(filenames, distances, classification_txt, title_txt):
    images = []
    for filename in filenames:
        images.append(mpimg.imread(filename))
    plt.figure(figsize=(20, 15))
    columns = 5
    for i, image in enumerate(images):
        ax = plt.subplot(int(len(images) / columns + 1), columns, i + 1)
        ax.set_title("\n\n" + filenames[i].split("/")[-1] + "\n" +
                     "\n" + classification_txt + ", prob=" + str(float("{0:.2f}".format(distances[i])))
                     )
        plt.suptitle(title_txt, fontsize=20, fontweight='bold')
        plt.axis('off')
        plt.imshow(image)
    plt.show()


filenames = valid_dataset.filenames


def display(sorted_indices, title_txt):
    similar_image_paths = []
    distances = []
    for name, value in sorted_indices:
        [probability, predicted_index, gt] = value
        if predicted_index == gt:
            classification_txt = "CORRECT"
        else:
            classification_txt = "WRONG"
        similar_image_paths.append(os.path.join(valid_data_path, filenames[name]))
        distances.append(probability)
    plot_images(similar_image_paths, distances, classification_txt, title_txt)


##
img_list = get_images_with_sorted_probabilities(prediction_table,
                                                             get_highest_probability=True,
                                                             label=0,
                                                             number_of_items=20,
                                                             only_false_predictions=False)
message = 'Images with highest probability of containing cats'
display(img_list, message)

##
img_list = get_images_with_sorted_probabilities(prediction_table,
                                                             get_highest_probability=True,
                                                             label=1,
                                                             number_of_items=20,
                                                             only_false_predictions=False)
message = 'Images with highest probability of containing no cats'
display(img_list, message)

##
img_list = get_images_with_sorted_probabilities(prediction_table,
                                                              get_highest_probability=False,
                                                              label=0,
                                                              number_of_items=20,
                                                              only_false_predictions=False)
message = 'Images classified as Cat, with lowest probability'
display(img_list, message)

##
img_list = get_images_with_sorted_probabilities(prediction_table,
                                                        get_highest_probability=False,
                                                        label=1,
                                                        number_of_items=25,
                                                        only_false_predictions=True)
message = 'Wrongly classified images, with lowest probability'
display(img_list, message)

##
img_list = get_images_with_sorted_probabilities(prediction_table,
                                                        get_highest_probability=False,
                                                        label=0,
                                                        number_of_items=20,
                                                        only_false_predictions=True)
message = 'Wrongly classified images, with lowest probability'
display(img_list, message)


##

