from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
# Download and read MNIST dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

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
W = tf.Variable(tf.zeros([784, 10]), name='W')
b = tf.Variable(tf.zeros([10]), name='b')

# softmax activation already computed by softmax_cross_entropy_with_logits
# y = tf.nn.softmax(tf.matmul(x, W) + b)
# Using the cross entropy as follows is numerically unstable
# cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=1))

# Internally computes the softmax activation so we apply it to (tf.matmul(x, W) + b)
y = tf.matmul(x, W) + b
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

sess = tf.InteractiveSession()

# Initialize variables
tf.global_variables_initializer().run()

n_epochs = 500
batch_size = 100

for epoch in range(n_epochs):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Evaluating performance of the model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print("Epoch: {0} accuracy: {1}".format(epoch, acc))
    val_acc = sess.run(accuracy, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
    print("Epoch: {0} validation accuracy: {1}".format(epoch, val_acc))

test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print("Test accuracy: {0}".format(test_acc))
