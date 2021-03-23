##
# import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import matplotlib.pyplot as plt

##
full_dataset = 'C:/git_zhaw/catnocat/data/raw/'

##

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=47)
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=22)

# print("nr of traning samples = ", len(x_train))
# print("nr of validation samples = ", len(x_val))
# print("nr of test samples = ", len(x_test))


##
image_size = (280, 280)
batch_size = 32

##
image_generator = ImageDataGenerator(rescale=1/255,
                                     validation_split=0.4)

train_dataset = image_generator.flow_from_directory(batch_size=batch_size,
                                                    directory=full_dataset,
                                                    shuffle=True,
                                                    target_size=image_size,
                                                    subset="training",
                                                    class_mode='categorical')

validation_dataset = image_generator.flow_from_directory(batch_size=batch_size,
                                                         directory=full_dataset,
                                                         shuffle=True,
                                                         target_size=image_size,
                                                         subset="validation",
                                                         class_mode='categorical')

##

print("Number of samples: ", train_dataset.samples)
print("Number of classes: ", len(train_dataset.class_indices))
print("Classes ", train_dataset.class_indices)
print("files: ", train_dataset.filenames[0:5])

##
x, y = next(train_dataset)
print("x: ", type(x))
print("y: ", type(y))
print("x: ", x.shape)
print("y: ", y.shape)


##
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(x[i])
    plt.title(y[i])
    plt.axis("off")

plt.show()

