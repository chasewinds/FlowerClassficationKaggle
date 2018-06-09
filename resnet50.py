from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import numpy as np
from PIL import Image

tf.logging.set_verbosity(tf.logging.INFO)

# Instantiate a Keras inception v3 model.
keras_resnet50 = tf.keras.applications.resnet50.ResNet50(weights=None, classes=4)
# Compile model with the optimizer, loss, and metrics you'd like to train with.
keras_resnet50.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0003),
                          loss='categorical_crossentropy',
                          metric='accuracy')
# Create an Estimator from the compiled Keras model. Note the initial model
# state of the keras model is preserved in the created Estimator.
estimator_resnet50 = tf.keras.estimator.model_to_estimator(keras_model=keras_resnet50)

image_path = '/home/rubans/proj/kaggle/flower/dataset/flowers'
def deal_with_one_floder(floder_path, label_class, img_label_dict):
    for image_name in os.listdir(floder_path):
        image_path = os.path.join(floder_path, image_name)
        if label_class == 0:
            img_label_dict[image_path] = '1000'
        elif label_class == 1:
            img_label_dict[image_path] = '0100'
        elif label_class == 2:
            img_label_dict[image_path] = '0010'
        elif label_class == 3:
            img_label_dict[image_path] = '0001'

label_class = 0
image_label_dict = {}
for floders in os.listdir(image_path):
    if floders == '.DS_Store': continue
    floders_path = os.path.join(image_path, floders)
    print(floders_path)
    deal_with_one_floder(floders_path, label_class, image_label_dict)
    label_class += 1
    
print("Finish process images")

def read_jpg(image_path):
    decode_image = tf.image.decode_jpeg(image_path)
    resized_image = tf.image.resize_images(decode_image, [224, 224])
    return resized_image

image_arr = []
label_arr = []
# init size = (333, 500, 3)
# size = 28, 28
for k, v in image_label_dict.items():
    try:
        img = Image.open(k)
        img = img.resize((224, 224),Image.BICUBIC)
        img.load()
        np_img = np.asarray(img, dtype="int32")
#         print(np_img.shape)
        image_arr.append(np_img) 
        label_arr.append(v)
        iters += 1
    except:
        continue

np_img_arr = np.asarray(image_arr, dtype=np.float32)
print(np_img_arr.shape)

np_label_arr = np.asarray(label_arr, dtype=np.int32)
print(np_label_arr.shape)
print(np_label_arr)

def main(unused_argv):
    # Load training and eval data
#     mnist = tf.contrib.learn.datasets.load_dataset("mnist")
#     train_data = mnist.train.images  # Returns np.array
#     train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
#     eval_data = mnist.test.images  # Returns np.array
#     eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    train_data = np_img_arr  # Returns np.array
    train_labels = np_label_arr
    eval_data = np_img_arr  # Returns np.array
    eval_labels = np_label_arr


    
    
    
    # Create the Estimator
#     mnist_classifier = tf.estimator.Estimator(
#         model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model1")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
#     tensors_to_log = {"probabilities": "softmax_tensor"}
#     logging_hook = tf.train.LoggingTensorHook(
#         tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    input_str = str(keras_resnet50.input_names)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"input_1": train_data},
        y=train_labels,
        batch_size=16,
        num_epochs=None,
        shuffle=True)
    estimator_resnet50.train(
        input_fn=train_input_fn,
        steps=200000)

    # Evaluate the model and print results
#     eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#         x={"x": eval_data},
#         y=eval_labels,
#         num_epochs=1,
#         shuffle=False)
#     eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
#     print(eval_results)


if __name__ == "__main__":
    tf.app.run()
