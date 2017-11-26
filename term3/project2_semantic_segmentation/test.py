import os.path
import tensorflow as tf

def main1():
    x_init = tf.random_normal_initializer(mean=0.0, stddev=0.1)
    x1 = tf.get_variable('x1', [1, 224, 224, 3], initializer=x_init)
    x2 = tf.get_variable('x2', [1, 224, 224, 64], initializer=x_init)
    # We have x3 output which we can use for skip
    x3 = tf.get_variable('x3', [1, 112, 112, 128], initializer=x_init)
    # We have x4 output which we can use for skip
    x4 = tf.get_variable('x4', [1, 56, 56, 256], initializer=x_init)
    x5 = tf.get_variable('x5', [1, 28, 28, 512], initializer=x_init)
    x6 = tf.get_variable('x6', [1, 14, 14, 512], initializer=x_init)
    x7 = tf.get_variable('x7', [1, 7, 7, 512], initializer=x_init)
    
    conv_1x1 = tf.layers.conv2d(x7, 2, 1, strides=(1,1), padding='SAME')
    
    # 8x upsample
    x44 = tf.layers.conv2d(x4, 2, 1, strides=(1,1), padding='SAME')
    y4 = tf.layers.conv2d_transpose(conv_1x1, 2, 16, strides=(8,8), padding='SAME')
    y4 = tf.add(y4, x44)
    
    # x2 upsample
    x33 = tf.layers.conv2d(x3, 2, 1, strides=(1,1), padding='SAME')
    y3 = tf.layers.conv2d_transpose(y4, 2, 4, strides=(2,2), padding='SAME')
    y3 = tf.add(y3, x33)

    # x2 upsample
    y1 = tf.layers.conv2d_transpose(y3, 2, 4, strides=(2,2), padding='SAME')
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output = sess.run(y1)
        print('conv_1x1 shape: {}'.format(conv_1x1.get_shape()))
        print('y4 shape: {}'.format(y4.get_shape()))
        print('y3 shape: {}'.format(y3.get_shape()))
        print('y1 shape: {}'.format(y1.get_shape()))

def main2():
    a_init = tf.random_normal_initializer(mean=0.0, stddev=0.1)
    a1 = tf.get_variable('a1', [1, 224, 224, 3], initializer=a_init)
    a2 = tf.get_variable('a2', [1, 224, 224, 64], initializer=a_init)
    # We have x3 output which we can use for skip
    a3 = tf.get_variable('a3', [1, 112, 112, 128], initializer=a_init)
    # We have x4 output which we can use for skip
    a4 = tf.get_variable('a4', [1, 56, 56, 256], initializer=a_init)
    a5 = tf.get_variable('a5', [1, 28, 28, 512], initializer=a_init)
    a6 = tf.get_variable('a6', [1, 14, 14, 512], initializer=a_init)
    a7 = tf.get_variable('a7', [1, 7, 7, 512], initializer=a_init)
    

    conv_1x1 = tf.layers.conv2d(a7, 2, 1, 1, padding='SAME')
    
    # upsample x2
    layer4 = tf.layers.conv2d_transpose(conv_1x1, 2, 4, 2, padding='SAME')
    # make sure the shapes are the same!
    # 1x1 convolution of vgg layer 4
    layer4_conv_1x1 = tf.layers.conv2d(a4, 2, 1, 1, padding='SAME')
    # skip connection (element-wise addition)
    layer4 = tf.add(layer4, layer4_conv_1x1)
    
    # upsample x2
    layer3 = tf.layers.conv2d_transpose(layer4, 2, 4, 2, padding='SAME')
    # 1x1 convolution of vgg layer 3
    layer3_conv_1x1 = tf.layers.conv2d(a3, 2, 1, 1, padding='SAME')
    # skip connection (element-wise addition)
    layer3 = tf.add(layer3, layer3_conv_1x1)
    
    # upsample x8
    nn_last_layer = tf.layers.conv2d_transpose(layer3, 2, 16, 8, padding='SAME')
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output = sess.run(nn_last_layer)
        print('conv_1x1 shape: {}'.format(conv_1x1.get_shape()))
        print('layer4 shape: {}'.format(layer4.get_shape()))
        print('layer3 shape: {}'.format(layer3.get_shape()))
        print('nn_last_layer shape: {}'.format(nn_last_layer.get_shape()))
        

if __name__ == '__main__':
    print('main1')
    main1()
    print()

    print('main2')
    main2()
    print()
