import tensorflow as tf

# The file path to save the data
save_file = './model.ckpt'

# Two tensor variables: weights and bias
weights = tf.Variable(tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]))

# Class used to save and/or restore tensr variables
saver = tf.train.Saver()

with tf.Session() as session:
    # Initialize all the variables
    session.run(tf.global_variables_initializer())
    # Show the values of weights and bias
    print('Weights: ')
    print(session.run(weights))
    print('Bias: ')
    print(session.run(bias))

    # Save the model
    saver.save(session, save_file)

# Reload variables back

# Remove the previous weights and bias
tf.reset_default_graph()

# Two variables: weights and bias
weights = tf.Variable(tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]))

# Class used to save and/or restore tensor variables
saver = tf.train.Saver()

with tf.Session() as session:
    # Load weights and bias from file
    saver.restore(session, save_file)

    # Show the values of weights and bias
    print('Weights: ')
    print(session.run(weights))
    print('Bias: ')
    print(session.run(bias))
