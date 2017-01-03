"""Implements a traffic sign classifier in Tensorflow based on a LeNet
architecture, trained on the [German Traffic Sign
dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)"""

import sys
import csv
import glob
import pickle
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten


def get_jittered_images(image, extend_by):
    """Makes the network more translation and scale invariant by moving the
    image left and right and zooming in on the sign"""
    images = []
    w_h = image.shape[0]
    new_size = w_h + (2*extend_by)
    grown = cv2.resize(image, (new_size, new_size), interpolation=cv2.INTER_CUBIC)
    for x0 in [0, extend_by, extend_by*2]:
        for y0 in [0, extend_by, extend_by*2]:
            x1 = x0 + w_h
            y1 = y0 + w_h
            images.append(np.array(grown[x0:x1, y0:y1], np.int32))
    return images

def augment_data(images, labels_list):
    """augments a data set by making altered copies of images and append them
    with their corresponding labels to the data set"""
    aug_images = []
    aug_labels = []
    for img, label in zip(images, labels_list):
        aug_images.append(img)
        aug_labels.append(label)
        jittered_images = get_jittered_images(img, 1)
        aug_images.extend(jittered_images)
        aug_labels.extend([label for i in range(len(jittered_images))])
    return aug_images, aug_labels



training_file = 'traffic-signs-data/train.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
X_train, y_train = augment_data(X_train, y_train)
X_val, y_val = augment_data(X_val, y_val)
n_classes = 43

print("Number of training examples =", len(X_train))
print("Number of testing examples =", len(X_test))
print("Image data shape =", X_train[0].shape)
print("Number of classes =", n_classes)



def lenet(x_input):
    """Implements an alter version of LeNet in Tensorflow"""
    # Hyperparameters
    mu = 0
    sigma = 0.1

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean=mu, stddev=sigma), name="conv1_W")
    conv1_b = tf.Variable(tf.zeros(6), name="conv1_b")
    conv1 = tf.nn.conv2d(x_input, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma), name="conv2_W")
    conv2_b = tf.Variable(tf.zeros(16), name="conv2_b")
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)

    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma), name="fc1_W")
    fc1_b = tf.Variable(tf.zeros(120), name="fc1_b")
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1 = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma), name="fc2_W")
    fc2_b = tf.Variable(tf.zeros(84), name="fc2_b")
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2 = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean=mu, stddev=sigma), name="fc3_W")
    fc3_b = tf.Variable(tf.zeros(n_classes), name="fc3_b")
    return tf.matmul(fc2, fc3_W) + fc3_b


EPOCHS = 10
BATCH_SIZE = 128
rate = 0.001

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

logits = lenet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

classify = tf.argmax(logits, 1)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

infile = open('signnames.csv', mode='r')
reader = csv.reader(infile)
labels = {row[0]:row[1] for row in reader}

def logits_to_labels_sorted(logits_result):
    as_pairs = [{'p':logits_result[i], 'name': labels[str(i)]} for i in range(len(logits_result))]
    return sorted(as_pairs, key=lambda k: -k['p'])

def evaluate(X_data, y_data):
    """Calculates the accuracy of the x vs y"""
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

def train_and_save():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)

        print("Training...")
        print()
        for i in range(EPOCHS):
            e_X_train, e_y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = e_X_train[offset:end], e_y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

            validation_accuracy = evaluate(X_val, y_val)
            print("EPOCH {} ...".format(i+1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print()

        saver.save(sess, 'lenet')
        print("Model saved")
        print(sess.run(classify, feed_dict={x: X_train[0:1]}))
        print(sess.run(classify, feed_dict={x: X_train[1:2]}))
        print(sess.run(classify, feed_dict={x: X_train[1000:1005]}))

def find_all_images_of_class(imgs, labels_list, label):
    """Search the labels list for labels that match the label argument and then
    extract imgs in those same indexes"""
    filtered = []
    for img, lbl in zip(imgs, labels_list):
        if lbl == label:
            filtered.append(img)
    return filtered

def classify_from_file(filename, count=3):
    """Reads an image from a file, and performs classification using the default
    tf session"""
    img = cv2.imread(filename)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sess = tf.get_default_session()
    l_res = sess.run(logits, feed_dict={x: [img]})[0]
    return logits_to_labels_sorted(l_res.tolist())[0:count]

def load_and_test(sess_filename='lenet', files=None):
    with tf.Session() as sess:
        saver.restore(sess, './'+sess_filename)
        #print(sess.run(classify, feed_dict={x: X_train[0:1]}))
        #stop_signs = find_all_images_of_class(X_train, y_train, 14)

        for filename in files:
            print(filename[17:], classify_from_file(filename))


TRAIN = len(sys.argv) > 1 and sys.argv[1] != None

if TRAIN:
    train_and_save()
else:
    load_and_test(files=glob.glob('hand-test-images/*.jpg'))


