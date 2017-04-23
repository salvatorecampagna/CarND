import csv
import numpy as np
import zipfile
import os
import tqdm
import PIL.Image as Image

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Conv2D, Cropping2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def unzip(input_path, output_path):
    """
    Extract a zip archive
    """
    print("Extracting {0} to {1}".format(input_path, output_path))
    zip_ref = zipfile.ZipFile(input_path, 'r')
    zip_ref.extractall(output_path)
    zip_ref.close()


def histogram(data, title, num_bins=50):
    """
    Plot steering angle distribution
    """
    plt.figure()
    plt.hist(data, num_bins)
    plt.title(title)
    plt.xlabel('Steering angle')
    plt.ylabel('Frequency')
    plt.savefig('{}.png'.format(title))


def load_dataset(csvfile_path='./data/driving_log.csv'):
    """
    Load a CSV driving log file
    """
    lines = []
    images = []
    measurements = []

    with open(csvfile_path) as csv_file:
        lines.extend(csv.reader(csv_file))
    lines = lines[1:]

    for i in tqdm.trange(len(lines)):
        line = lines[i]

        filename_center = line[0].split('/')[-1]
        filename_left = line[1].split('/')[-1]
        filename_right = line[2].split('/')[-1]

        center_path = './data/IMG/' + filename_center
        left_path = './data/IMG/' + filename_left
        right_path = './data/IMG/' + filename_right

        # Defer image loading into the generator
        images.extend([center_path, left_path, right_path])

        measure_center = float(line[3])
        # Apply correction to the steering angle
        # for left and right camera images
        correction = 0.1
        measure_left = measure_center + correction
        measure_right = measure_center - correction
        measurements.extend([measure_center, measure_left, measure_right])

        i += 1

    return images, measurements


def model_build(input_shape):
    """
    Build a Keras model to predict the steering angle
    """
    model = Sequential()
    # Normalization and mean centering
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=input_shape))
    # Crop the input image to discard top 70 lines and bottom 25 lines
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Conv2D(24, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    # Add a dropout layer to prevent overfitting
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model


def model_plot_history(history_object, title, xlabel, ylabel, legend):
    """
    Plot the model training and validation loss
    """
    plt.figure()
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(legend, loc='upper right')
    plt.savefig('loss_plot.png')


def generator(X, y, batch_size=32):
    """
    Training and validation batch generator
    """
    nums_samples = len(X)
    while 1:
        shuffle(X, y)
        for offset in range(0, nums_samples, batch_size):
            batch_x = X[offset:offset + batch_size]
            batch_y = y[offset:offset + batch_size]
            images = []
            measurements = []
            for bx, by in zip(batch_x, batch_y):
                image = np.array(Image.open(bx))
                measure = float(by)
                images.append(image)
                measurements.append(measure)
            X_gen = np.array(images)
            y_gen = np.array(measurements)
            yield shuffle(X_gen, y_gen)


save_path = 'model.h5'

if(not os.path.isdir('./data')):
    unzip('./data.zip', '.')

X, y = load_dataset()

# Shuffle the dataset
X, y = shuffle(X, y)

print("X len: {}".format(len(X)))
print("y len: {}".format(len(y)))
print()

# Split the dataset in training, validation and test set
X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, test_size=0.6, random_state=0)

print("X_train shape: {}".format(len(X_train)))
print("y_train shape: {}".format(len(y_train)))
print("X_valid shape: {}".format(len(X_valid)))
print("y_valid shape: {}".format(len(y_valid)))
print("X_test shape: {}".format(len(X_test)))
print("y_test shape: {}".format(len(y_test)))
print()

histogram(y, title="Steering angle distribution")
histogram(y_train, title="Steering angle distribution (training set)")
histogram(y_valid, title="Steering angle distribution (validation set)")
histogram(y_test, title="Steering angle distribution (test set)")

# Generators for the training and validation set
train_generator = generator(X_train, y_train)
valid_generator = generator(X_valid, y_valid)

# Build, compile and train the model
model = model_build(input_shape=(160, 320, 3))
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model_history = model.fit_generator(train_generator,
                                    samples_per_epoch=len(X_train),
                                    validation_data=valid_generator,
                                    nb_val_samples=len(X_valid),
                                    nb_epoch=10)

print("Saving model to: {}".format(save_path))
model.save(save_path)

# Evaluate the model on the test dataset
test_x, test_y = [], []
for tx, ty in zip(X_test, y_test):
    test_x.append(np.array(Image.open(tx)))
    test_y.append(np.array(ty))
test_x, test_y = np.array(test_x), np.array(test_y)

# Evaluate the model on the test set
metrics = model.evaluate(test_x, test_y, verbose=1)
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print("{}: {}".format(metric_name, metric_value))

# Plot training and validation loss
model_plot_history(
    model_history,
    title='Training/Validation Loss',
    xlabel='Epochs',
    ylabel='MSE Loss',
    legend=['Training loss', 'Validation loss'])

model.summary()
