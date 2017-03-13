"""
LeNet Architecture
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import flatten


EPOCHS = 10
BATCH_SIZE = 50

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W, strides):
    return tf.nn.conv2d(x, W, strides, padding='VALID')

def max_pool(x, ksize, strides):
    return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding='VALID')

def build_net(x):
    print('x shape: {0}'.format(x.get_shape()))
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    # TODO: Activation
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    # Convolutional layer output dimensions (padding='VALID')
    # out_height = ceil(float(in_height - filter_height + 1)/float(stripes[1]))
    # = ceil((32 - 5 + 1)/1) = 28
    # out_width = ceil(float(in_width - filter_width + 1)/float(stripes[2]))
    # = ceil((32 - 5 + 1)/1) = 28
    # First layer weight and bias tensors
    W_conv1 = weight_variable([5, 5, 1, 6])
    b_conv1 = bias_variable([6])
    # First convolutional layer + activation
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1, strides=[1, 1, 1, 1]) + b_conv1)
    print('h_conv1 shape: {0}'.format(h_conv1.get_shape()))
    # Max pooling layer computation (padding='VALID')
    # out_height = ceil(float(in_height - ksize[1] + 1)/float(stripes[1]))
    # = ceil((28 - 2 + 1)/2) = ceil(13.5) = 14
    # out_height = ceil(float(in_height - ksize[1] + 1)/float(stripes[1]))
    # = ceil((28 - 2 + 1)/2) = ceil(13.5) = 14
    # First layer pooling
    h_pool1 = max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
    print('h_pool1 shape: {0}'.format(h_pool1.get_shape()))

    # Convolutional layer output dimensions (padding='VALID')
    # out_height = ceil(float(in_height - filter_height + 1)/float(stripes[1]))
    # = ceil((14 - 5 + 1)/1) = 10
    # out_width = ceil(float(in_width - filter_width + 1)/float(stripes[2]))
    # = ceil((14 - 5 + 1)/1) = 10
    # First layer weight and bias tensors
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    # TODO: Activation
    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    # Second layer weight and bias tensors
    W_conv2 = weight_variable([5, 5, 6, 16])
    b_conv2 = bias_variable([16])
    # Second layer convolution + activation
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1]) + b_conv2)
    print('h_conv2 shape: {0}'.format(h_conv2.get_shape()))
    # Max pooling layer computation (padding='VALID')
    # out_height = ceil(float(in_height - ksize[1] + 1)/float(stripes[1]))
    # = ceil((10 - 2 + 1)/2) = ceil(4.5) = 5
    # out_height = ceil(float(in_height - ksize[1] + 1)/float(stripes[1]))
    # = ceil((28 - 2 + 1)/2) = ceil(4.5) = 5
    # Second layer pooling
    h_pool2 = max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
    print('h_pool2 shape: {0}'.format(h_pool2.get_shape()))

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    # Flatten
    h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 16])
    print('h_pool2_flat shape: {0}'.format(h_pool2_flat.get_shape()))

    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    # TODO: Activation
    # First fully connected layer weight and bias
    W_fc1 = weight_variable([400, 120])
    b_fc1 = bias_variable([120])
    # First fully connected later + activation
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    print('h_fc1 shape: {0}'.format(h_fc1.get_shape()))

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    # TODO: Activation
    # Second fully connected layer weight and bias
    W_fc2 = weight_variable([120, 84])
    b_fc2 = bias_variable([84])
    # First fully connected later + activation
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    print('h_fc2 shape: {0}'.format(h_fc2.get_shape()))

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    # Add the output layer
    W_out = weight_variable([84, 10])
    b_out = bias_variable([10])
    y_hat = tf.matmul(h_fc2, W_out) + b_out
    print('y_hat shape: {0}'.format(y_hat.get_shape()))

    return y_hat

# LeNet architecture:
# INPUT -> CONV -> ACT -> POOL -> CONV -> ACT -> POOL -> FLATTEN -> FC -> ACT -> FC
#
# Don't worry about anything else in the file too much, all you have to do is
# create the LeNet and return the result of the last fully connected layer.
def LeNet(x):
    # Reshape from 2D to 4D. This prepares the data for
    # convolutional and pooling layers.
    x = tf.reshape(x, (-1, 28, 28, 1))
    # Pad 0s to 32x32. Centers the digit further.
    # Add 2 rows/columns on each side for height and width dimensions.
    x = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="CONSTANT")
    # TODO: Define the LeNet architecture.
    x = build_net(x)
    # Return the result of the last fully connected layer.
    return x


# MNIST consists of 28x28x1, grayscale images
x = tf.placeholder(tf.float32, (None, 784))
# Classify over 10 digits 0-9
y = tf.placeholder(tf.float32, (None, 10))
fc2 = LeNet(x)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc2, y))
opt = tf.train.AdamOptimizer()
train_op = opt.minimize(loss_op)
correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def eval_data(dataset):
    """
    Given a dataset as input returns the loss and accuracy.
    """
    # If dataset.num_examples is not divisible by BATCH_SIZE
    # the remainder will be discarded.
    # Ex: If BATCH_SIZE is 64 and training set has 55000 examples
    # steps_per_epoch = 55000 // 64 = 859
    # num_examples = 859 * 64 = 54976
    #
    # So in that case we go over 54976 examples instead of 55000.
    steps_per_epoch = dataset.num_examples // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE
    total_acc, total_loss = 0, 0
    sess = tf.get_default_session()
    for step in range(steps_per_epoch):
        batch_x, batch_y = dataset.next_batch(BATCH_SIZE)
        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={x: batch_x, y: batch_y})
        total_acc += (acc * batch_x.shape[0])
        total_loss += (loss * batch_x.shape[0])
    return total_loss/num_examples, total_acc/num_examples


if __name__ == '__main__':
    # Load data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        steps_per_epoch = mnist.train.num_examples // BATCH_SIZE
        num_examples = steps_per_epoch * BATCH_SIZE

        # Train model
        for i in range(EPOCHS):
            for step in range(steps_per_epoch):
                batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
                loss = sess.run(train_op, feed_dict={x: batch_x, y: batch_y})

            val_loss, val_acc = eval_data(mnist.validation)
            print("EPOCH {} ...".format(i+1))
            print("Validation loss = {:.3f}".format(val_loss))
            print("Validation accuracy = {:.3f}".format(val_acc))
            print()

        # Evaluate on the test data
        test_loss, test_acc = eval_data(mnist.test)
        print("Test loss = {:.3f}".format(test_loss))
        print("Test accuracy = {:.3f}".format(test_acc))

