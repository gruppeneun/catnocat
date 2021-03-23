##
# import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import matplotlib.pyplot as plt
import os, shutil
import random

##
data_path = 'C:/git_zhaw/catnocat/data'
orig_dataset = os.path.join(data_path, 'raw')

trainvalid_path = os.path.join(data_path, 'trainvalid')
holytest_path = os.path.join(data_path, 'holytest')

##

#
if os.path.exists(trainvalid_path):
    shutil.rmtree(trainvalid_path)

# copy the whole dataset into a new directory
print('Copying the original images to a new directory. This takes a while....')
shutil.copytree(orig_dataset, trainvalid_path)

##
# move 20% of the data out of the testvalid set into the holy test set
holy_testset_proportion = 0.2

cat_files = os.listdir(os.path.join(orig_dataset ,'cat'))
nocat_files = os.listdir(os.path.join(orig_dataset ,'nocat'))

random.seed(2384)
holy_test_filenames = {'cat':   random.sample(cat_files, int(len(cat_files) * holy_testset_proportion)),
                       'nocat': random.sample(nocat_files, int(len(nocat_files) * holy_testset_proportion))
                       }

print("Holy Test files for Cats: ", len(holy_test_filenames['cat']))
print("Holy Test files for NoCats: ", len(holy_test_filenames['nocat']))

# move the holy test files to a dedicated directory
if os.path.exists(holytest_path):
    shutil.rmtree(holytest_path)

##
os.mkdir(holytest_path)

# move the cat test files:

for c in holy_test_filenames:
    print('Moving files for ', c)
    subFolder = os.path.join(holytest_path, c)
    os.makedirs(subFolder)

    for file in holy_test_filenames[c]:
        shutil.move(os.path.join(trainvalid_path, c, file), subFolder)



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
                                     validation_split=0.2)

train_dataset = image_generator.flow_from_directory(batch_size=batch_size,
                                                    directory=trainvalid_path,
                                                    shuffle=True,
                                                    target_size=image_size,
                                                    subset="training",
                                                    class_mode='categorical')

valid_dataset = image_generator.flow_from_directory(batch_size=batch_size,
                                                         directory=trainvalid_path,
                                                         shuffle=True,
                                                         target_size=image_size,
                                                         subset="validation",
                                                         class_mode='categorical')

##
print("TrainingSet Total")
print("Number of samples: ", train_dataset.samples)
print("Number of classes: ", len(train_dataset.class_indices))
print("Classes ", train_dataset.class_indices)
#print("files: ", train_dataset.filenames[0:5])

print("\nValidation Set Total")
print("Number of samples: ", valid_dataset.samples)
print("Number of classes: ", len(valid_dataset.class_indices))
print("Classes ", valid_dataset.class_indices)


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



