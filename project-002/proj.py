# Load pickled data
import sys
import pickle
import cv2
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten

import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
#%matplotlib inline

def get_jittered_images(image, new_size):
    images = []
    w_h = image.shape[0]
    grown = cv2.resize(image, (new_size, new_size), interpolation = cv2.INTER_CUBIC)
    xy = int((new_size - w_h)/2)
    for x in [0, xy, xy*2]:
        for y in [0, xy, xy*2]:
            xy1 = xy + w_h
            images.append(np.array(grown[xy:xy1, xy:xy1], np.int32))
    return images

def augment_data(images, labels):
    aug_images = []
    aug_labels = []
    for i in range(len(images)):
        img = images[i]
        label = labels[i]
        aug_images.append(img)
        aug_labels.append(label)
        jittered_images = get_jittered_images(img, 40)
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

X_train, y_train = augment_data(X_train, y_train)
n_train = len(X_train)
n_test = len(X_test)
image_shape = X_train[0].shape
n_classes = 43

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

def LeNet(x):
    # Hyperparameters
    mu = 0
    sigma = 0.1

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma), name="conv1_W")
    conv1_b = tf.Variable(tf.zeros(6), name="conv1_b")
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma), name="conv2_W")
    conv2_b = tf.Variable(tf.zeros(16), name="conv2_b")
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)

    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma), name="fc1_W")
    fc1_b = tf.Variable(tf.zeros(120), name="fc1_b")
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma), name="fc2_W")
    fc2_b  = tf.Variable(tf.zeros(84), name="fc2_b")
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma), name="fc3_W")
    fc3_b  = tf.Variable(tf.zeros(43), name="fc3_b")
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits


EPOCHS = 10
BATCH_SIZE = 128
rate = 0.001

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

classify = tf.argmax(logits, 1)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

infile = open('signnames.csv', mode='r')
reader = csv.reader(infile)
labels = {row[0]:row[1] for row in reader}
#print(labels)

def logits_to_labels_sorted(logits_result):
  as_pairs = [{'p':logits_result[i], 'name': labels[str(i)]} for i in range(len(logits_result))]
  return sorted(as_pairs, key=lambda k: -k['p'])

def evaluate(X_data, y_data):
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

            validation_accuracy = evaluate(X_validation, y_validation)
            print("EPOCH {} ...".format(i+1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print()

        saver.save(sess, 'lenet')
        print("Model saved")
        print(sess.run(classify, feed_dict={x: X_train[0:1]}))
        print(sess.run(classify, feed_dict={x: X_train[1:2]}))
        print(sess.run(classify, feed_dict={x: X_train[1000:1005]}))

def find_all_images_of_class(imgs, labels, label):
    filtered = []
    for i in range(len(labels)):
        if labels[i] == label:
          filtered.append(imgs[i])
    return filtered

def classify_from_file(filename):
    img = cv2.imread(filename)
    img = cv2.resize(img,(32,32), interpolation = cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sess = tf.get_default_session()
    l_res = sess.run(logits, feed_dict={x: [img]})[0]
    return logits_to_labels_sorted(l_res.tolist())[0:5]

def load_and_test(filename='lenet'):
    with tf.Session() as sess:
        saver.restore(sess, './'+filename)
        #print(sess.run(classify, feed_dict={x: X_train[0:1]}))
        #stop_signs = find_all_images_of_class(X_train, y_train, 14)

        print(classify_from_file('stopsign32blur2.jpg'))
        print(classify_from_file('stopsign32wide.jpg'))


TRAIN = len(sys.argv) > 1 and sys.argv[1] != None

if TRAIN:
    train_and_save()
else:
    load_and_test()


