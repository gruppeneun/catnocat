##
import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
import numpy as np
from IPython.display import Image
from keras.optimizers import Adam
import os
import matplotlib.pyplot as plt

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
                                                    class_mode='categorical')

valid_dataset = image_generator.flow_from_directory(batch_size=batch_size,
                                                    directory=trainvalid_path,
                                                    shuffle=True,
                                                    target_size=image_size,
                                                    subset="validation",
                                                    class_mode='categorical')

##
mobile = keras.applications.mobilenet.MobileNet()

##
#x, y = train_dataset[3]
x, y = next(train_dataset)

img_array = image.img_to_array(x[0])

plt.imshow(img_array)
plt.show()

img_array_expanded_dims = np.expand_dims(img_array, axis=0)

predictions = mobile.predict(img_array_expanded_dims)
results = imagenet_utils.decode_predictions(predictions)
results



