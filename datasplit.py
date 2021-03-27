##
# import matplotlib.pyplot as plt
import os
import random
import shutil

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

##
data_path = 'C:/git_zhaw/catnocat/data'
orig_dataset = os.path.join(data_path, 'raw')
cat_orig_files = os.listdir(os.path.join(orig_dataset, 'cat'))
nocat_orig_files = os.listdir(os.path.join(orig_dataset, 'nocat'))

train_data_path = os.path.join(data_path, 'train')
valid_data_path = os.path.join(data_path, 'valid')
holytest_data_path = os.path.join(data_path, 'holytest')

##
if os.path.exists(train_data_path):
    shutil.rmtree(train_data_path)

# copy the whole dataset into a new directory
print('Copying the original images to a new directory. This takes a while....')
shutil.copytree(orig_dataset, train_data_path)

##
cat_files = os.listdir(os.path.join(orig_dataset, 'cat'))
nocat_files = os.listdir(os.path.join(orig_dataset, 'nocat'))

print("Number of cat files: ", len(cat_files))
print("Number of nocat files: ", len(nocat_files))

##
if os.path.exists(holytest_data_path):
    shutil.rmtree(holytest_data_path)
os.mkdir(holytest_data_path)

if os.path.exists(valid_data_path):
    shutil.rmtree(valid_data_path)
os.mkdir(valid_data_path)


##
# move 20% of the data out of the training set into the validation set
proportion = 0.2
num_cat_files = int(len(cat_orig_files) * proportion)
num_nocat_files = int(len(nocat_orig_files) * proportion)

cat_files = os.listdir(os.path.join(train_data_path, 'cat'))
nocat_files = os.listdir(os.path.join(train_data_path, 'nocat'))

##
random.seed(2384)
validset_filenames = {'cat': random.sample(cat_files, num_cat_files),
                      'nocat': random.sample(nocat_files, num_nocat_files)
                      }

print("Validation files for Cats: ", len(validset_filenames['cat']))
print("Validation files for NoCats: ", len(validset_filenames['nocat']))

# move the validation files to a dedicated directory
for c in validset_filenames:
    print('Moving files for ', c)
    subFolder = os.path.join(valid_data_path, c)
    os.makedirs(subFolder)

    for file in validset_filenames[c]:
        shutil.move(os.path.join(train_data_path, c, file), subFolder)

print("Number of cat files left:   ", len(os.listdir(os.path.join(train_data_path, 'cat'))))
print("Number of nocat fiels left: ", len(os.listdir(os.path.join(train_data_path, 'nocat'))))

##
# move 20% of the data out of the train set into the holy test set
cat_files = os.listdir(os.path.join(train_data_path, 'cat'))
nocat_files = os.listdir(os.path.join(train_data_path, 'nocat'))

random.seed(2384)
random.seed(2384)
holytest_filenames = {'cat': random.sample(cat_files, num_cat_files),
                      'nocat': random.sample(nocat_files, num_nocat_files)
                      }

print("Holy Test files for Cats: ", len(holytest_filenames['cat']))
print("Holy Test files for NoCats: ", len(holytest_filenames['nocat']))

# move the holy test files to a dedicated directory
for c in holytest_filenames:
    print('Moving files for ', c)
    subFolder = os.path.join(holytest_data_path, c)
    os.makedirs(subFolder)

    for file in holytest_filenames[c]:
        shutil.move(os.path.join(train_data_path, c, file), subFolder)


print("Number of cat files left:   ", len(os.listdir(os.path.join(train_data_path, 'cat'))))
print("Number of nocat fiels left: ", len(os.listdir(os.path.join(train_data_path, 'nocat'))))

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
# Set Data Generator for training, testing and validataion.

# Note for testing, set shuffle = false (For proper Confusion matrix)


train_datagen = ImageDataGenerator(rescale=1/255)
train_dataset = train_datagen.flow_from_directory(train_data_path,
                                                  target_size=image_size,
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  shuffle=True)

valid_datagen = ImageDataGenerator(rescale=1/255)
valid_dataset = valid_datagen.flow_from_directory(valid_data_path,
                                                  target_size=image_size,
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  shuffle=True)

test_datagen = ImageDataGenerator(rescale=1/255)
test_dataset = test_datagen.flow_from_directory(holytest_data_path,
                                                target_size=image_size,
                                                batch_size=batch_size,
                                                class_mode='categorical',
                                                shuffle=False)

##
print("TrainingSet Total")
print("Number of samples: ", train_dataset.samples)
print("Number of classes: ", len(train_dataset.class_indices))
print("Classes ", train_dataset.class_indices)
# print("files: ", train_dataset.filenames[0:5])

print("\nValidation Set Total")
print("Number of samples: ", valid_dataset.samples)
print("Number of classes: ", len(valid_dataset.class_indices))
print("Classes ", valid_dataset.class_indices)

print("\nTest Set Total")
print("Number of samples: ", test_dataset.samples)
print("Number of classes: ", len(test_dataset.class_indices))
print("Classes ", test_dataset.class_indices)

##
x, y = next(train_dataset)
print("Batch Index: ", train_dataset.batch_index)
print("x: ", x.shape)
print("y: ", y.shape)

plt.figure(figsize=(20, 10))
for i in range(32):
    ax = plt.subplot(4, 8, i + 1)
    plt.imshow(x[i])
    plt.title(y[i])
    plt.axis("off")

plt.show()

##

