import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tensorflow.examples.tutorials.mnist import input_data
import os
import itertools
import time
from tqdm import tqdm



def load_dataset(name='MNIST'):

    if name == 'MNIST':
        data = input_data.read_data_sets("MNIST_data/", one_hot = True, reshape = [])
    elif name == 'Fashion':
        data = input_data.read_data_sets('data/fashion',
                                  source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/',
                                  one_hot=True, reshape=[])

    # normalization; range: -1 ~ 1
    train_set = data.train.images
    train_set = (train_set - 0.5) / 0.5
    train_setY = data.train.labels
    test_set = data.test.images
    test_set = (test_set - 0.5) / 0.5
    test_setY = data.test.labels
    val_set = data.validation.images
    val_set = (val_set - 0.5) / 0.5
    val_setY = data.validation.labels

    return train_set, train_setY, val_set, val_setY, test_set, test_setY


def save_result(sess, num_epoch, G_z, flatG_z, z, fixed_z_, show=False, save=False, path='result.png'):
    test_images, flat = sess.run([G_z, flatG_z], {z: fixed_z_})
    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)
    for k in range(size_figure_grid * size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        ax[i, j].imshow(np.reshape(test_images[k], (28, 28)), cmap='gray')
    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)
    plt.close()

def generator(x):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        conv1 = tf.layers.conv2d_transpose(x, 1024, [4, 4], strides=(1, 1), padding='valid')
        lrelu1 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv1, training=True), 0.2)
        conv2 = tf.layers.conv2d_transpose(lrelu1, 512, [3, 3], strides=(1, 1), padding='valid')
        lrelu2 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv2, training=True), 0.2)
        conv3 = tf.layers.conv2d_transpose(lrelu2, 256, [2, 2], strides=(1, 1), padding='valid')
        lrelu3 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv3, training=True), 0.2)
        conv4 = tf.layers.conv2d_transpose(lrelu3, 128, [4, 4], strides=(2, 2), padding='same')
        lrelu4 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv4, training=True), 0.2)
        conv5 = tf.layers.conv2d_transpose(lrelu4, 1, [4, 4], strides=(2, 2), padding='same')
        o = tf.nn.tanh(conv5)
    return o



def discriminator(input, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        conv1 = tf.nn.leaky_relu(tcl.conv2d(input, 64, 5, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv1'))
        conv2 = tf.nn.leaky_relu(tcl.conv2d(conv1, 128, 5, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv2'))
        conv3 = tf.nn.leaky_relu(tcl.conv2d(conv2, 256, 5, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv3'))
        conv4 = tf.nn.leaky_relu(tcl.conv2d(conv3, 512, 5, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv4'))
        conv5 = tf.nn.leaky_relu(tcl.conv2d(conv4, 1, 4, 1, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv5'))
        return conv5

# training parameters
batch_size = 1024
lr = 0.0002
train_epoch = 50

#load dataset
train_set, train_setY, val_set, val_setY, test_set, test_setY = load_dataset(name='MNIST')

x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100))

fixed_z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))

# networks : generator
G_z = generator(z)
flatG_z = tf.reshape(G_z, [batch_size, -1])

# networks : discriminator
D_real = discriminator(x)
D_fake = discriminator(G_z, reuse=True)

#Losses
D_loss = (tf.reduce_mean(D_real) - tf.reduce_mean(D_fake))
G_loss = tf.reduce_mean(D_fake)

epsilon = tf.random_uniform([], 0.0, 1.0)
x_hat = x*epsilon + (1-epsilon)*G_z
d_hat = discriminator(x_hat, reuse=True)
gradients = tf.gradients(d_hat, x_hat)[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = 10*tf.reduce_mean((slopes-1.0)**2)
D_loss += gradient_penalty


T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]


with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss, var_list=G_vars)


# results save folder
root = 'Results/'
model = 'Improved_Wasserstein_DCGAN'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Images'):
    os.mkdir(root + 'Images')


#store trainig hitory
train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

start_time = time.time()
for epoch in range(train_epoch):
    G_losses = []
    D_losses = []
    epoch_start_time = time.time()
    for iter in tqdm(range(train_set.shape[0] // batch_size)):
        # update discriminator
        x_ = train_set[iter * batch_size:(iter + 1) * batch_size]
        y_ = train_setY[iter * batch_size:(iter + 1) * batch_size]
        z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))

        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_})
        D_losses.append(loss_d_)

        # update generator
        z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))
        loss_g_, _, flatImage = sess.run([G_loss, G_optim, flatG_z],
                                         {z: z_, x: x_})
        G_losses.append(loss_g_)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % (
    (epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
    fixed_p = root + 'Images/' + model + str(epoch + 1)

    print('Generating Results')
    save_result(sess, (epoch + 1), G_z, flatG_z, z, fixed_z_ = fixed_z_, save=True, path=fixed_p + '.png')
    train_hist['D_losses'].append(np.mean(D_losses))
    train_hist['G_losses'].append(np.mean(G_losses))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)


