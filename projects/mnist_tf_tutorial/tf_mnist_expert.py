from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
# Download and read MNIST dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

print("Training set images shape: {0}".format(mnist.train.images.shape))
print("Training set labels shape: {0}".format(mnist.train.labels.shape))

print("Test set images shape: {0}".format(mnist.test.images.shape))
print("Training set labels shape: {0}".format(mnist.train.labels.shape))

print("Validation set images shape: {0}".format(mnist.validation.images.shape))
print("Training set labels shape: {0}".format(mnist.train.labels.shape))

train_size = mnist.train.images.shape[0]
test_size = mnist.test.images.shape[0]
validation_size = mnist.validation.images.shape[0]

n_features = mnist.train.images.shape[1]
n_classes = mnist.train.labels.shape[1]

print("Training set size: {0}".format(train_size))
print("Test set size: {0}".format(test_size))
print("Validation set size: {0}".format(validation_size))

print("n_features: {0}".format(n_features))
print("n_classes: {0}".format(n_classes))
x = tf.placeholder(tf.float32, [None, 784], name='x')
y_ = tf.placeholder(tf.float32, [None, 10], name='y_')

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# First layer weight and bias tensors
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# Reshape the image [None, 784] -> [-1, 28, 28, 1]
x_image = tf.reshape(x, [-1, 28, 28, 1])

# First layer: convolutional + max pooling
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second layer weight and bias tensors
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

# Second layer: convolutional + max pooling
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2((h_conv2))

# Densely connected layer (feature map size is 7*7*64)
# Add a fully connected layer with 1024 neurons

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Apply dropout to reduce overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Add the output layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Train and evaluate the model
sess = tf.InteractiveSession()
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%20 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))