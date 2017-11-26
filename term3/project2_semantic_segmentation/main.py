import argparse
import cv2
import os.path
import tensorflow as tf
import helper
import warnings
import datetime
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def print_duration(td):
    print('Duration days: {0} hours: {1} mins: {2}'.format(td.days, td.seconds//3600, (td.seconds//60)%60))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    # Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # Load the pre-trained VGG model (used as the FCN encoder)
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # TODO: include skip layers
    conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1,1), padding='SAME',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
        kernel_initializer= tf.random_normal_initializer(stddev=0.01))
    
    # upsample x2
    layer4 = tf.layers.conv2d_transpose(conv_1x1, num_classes, 4, strides=(2,2), padding='SAME',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
        kernel_initializer= tf.random_normal_initializer(stddev=0.01))
    # make sure the shapes are the same!
    # 1x1 convolution of vgg layer 4
    layer4_conv_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides=(1,1), padding='SAME',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
        kernel_initializer= tf.random_normal_initializer(stddev=0.01))
    # skip connection (element-wise addition)
    layer4 = tf.add(layer4, layer4_conv_1x1)
    
    # upsample x2
    layer3 = tf.layers.conv2d_transpose(layer4, num_classes, 4, strides=(2,2), padding='SAME',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
        kernel_initializer= tf.random_normal_initializer(stddev=0.01))
    # 1x1 convolution of vgg layer 3
    layer3_conv_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides=(1,1), padding='SAME',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
        kernel_initializer= tf.random_normal_initializer(stddev=0.01))
    # skip connection (element-wise addition)
    layer3 = tf.add(layer3, layer3_conv_1x1)
    
    # upsample x8
    nn_last_layer = tf.layers.conv2d_transpose(layer3, num_classes, 16, strides=(8,8), padding='SAME',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
        kernel_initializer= tf.random_normal_initializer(stddev=0.01))
    
    return nn_last_layer
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1,num_classes))

    # Loss function
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    sess.run(tf.global_variables_initializer())

    print('Training')
    print()

    for epoch in range(epochs):
        print('Epoch: {0}'.format(epoch + 1))
        start_time = datetime.datetime.now()
        print('Start: {0}'.format(start_time.strftime('%Y-%m-%d %H:%M:%S')))
        for (image, label) in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], 
                               feed_dict={
                                input_image: image,
                                correct_label: label,
                                keep_prob: 0.5,
                                learning_rate: 0.00005
                                })
            print("Loss: {0:.4f}".format(loss))
        end_time = datetime.datetime.now()
        print('End: {0}'.format(end_time.strftime('%Y-%m-%d %H:%M:%S')))
        delta_time = end_time - start_time
        print_duration(delta_time)
tests.test_train_nn(train_nn)


def run(epochs, batch_size, data_path='./data', runs_path='./output'):
    num_classes = 2
    image_shape = (160, 576)
    data_dir = data_path
    runs_dir = runs_path
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained VGG model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to VGG model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        output = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(output, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
            correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        
        # OPTIONAL: Apply the trained model to a video


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description='Process command line arguments')

    parser.add_argument('--epochs', type=int, required=True, help='Training epochs')
    parser.add_argument('--batch-size', type=int, required=True, help='Batch size')
    parser.add_argument('--data-dir', type=str, default='./data', required=False, help='Training data drectory')
    parser.add_argument('--runs-dir', type=str, default='./output', required=False, help='Image output directory')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_command_line_arguments()

    if args.epochs <= 0:
        raise ValueError('Epochs value must be positive: {0}'.format(args.epochs))

    if args.batch_size <= 0:
        raise ValueError('Batch size must be positive: {0}'.format(args.batch_size))

    print('Epochs: {0}'.format(args.epochs))
    print('Batch size: {0}'.format(args.batch_size))
    print('Data directory: {0}'.format(args.data_dir))
    print('Output directory: {0}'.format(args.runs_dir))
    run(epochs=args.epochs, batch_size=args.batch_size, data_path=args.data_dir, runs_path=args.runs_dir)