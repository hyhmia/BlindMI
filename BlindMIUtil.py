import numpy as np
import tensorflow as tf
from functools import partial
import cv2 as cv
import random

def gaussian_noise(img_set, mean=0, var=0.001):
    ret = np.empty(img_set.shape)
    for m, image in enumerate(img_set):
        image = np.array(image/255, dtype=float)
        noise = np.random.normal(mean, var ** 0.5, image.shape)
        out = image + noise
        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = np.clip(out, low_clip, 1.0)
        out = np.uint8(out*255)
        ret[m, :] = out
    return ret


def sobel(img_set):
    ret = np.empty(img_set.shape)
    for i, img in enumerate(img_set):
        grad_x = cv.Sobel(np.float32(img), cv.CV_32F, 1, 0)
        grad_y = cv.Sobel(np.float32(img), cv.CV_32F, 0, 1)
        gradx = cv.convertScaleAbs(grad_x)
        grady = cv.convertScaleAbs(grad_y)
        gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
        ret[i, :] = gradxy
    return ret


def sp_noise(img_set, prob=0.001):
    ret = np.empty(img_set.shape)
    for m, image in enumerate(img_set):
        out = np.zeros(image.shape, np.uint8)
        thres = 1 - prob
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    out[i][j] = 0
                elif rdn > thres:
                    out[i][j] = 255
                else:
                    out[i][j] = image[i][j]
        ret[m,:] = out
    return ret


def scharr(img_set):
    ret = np.empty(img_set.shape)
    for i, img in enumerate(img_set):
        grad_x = cv.Scharr(np.float32(img), cv.CV_32F, 1, 0)
        grad_y = cv.Scharr(np.float32(img), cv.CV_32F, 0, 1)
        gradx = cv.convertScaleAbs(grad_x)
        grady = cv.convertScaleAbs(grad_y)
        gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
        ret[i, :] = gradxy

    return ret


def laplace(img_set):
    ret = np.empty(img_set.shape)
    for i, img in enumerate(img_set):
        gray_lap = cv.Laplacian(np.float32(img), cv.CV_32F, ksize=3)
        dst = cv.convertScaleAbs(gray_lap)
        ret[i, :] = dst
    return ret


def compute_pairwise_distances(x, y):
    """Computes the squared pairwise Euclidean distances between x and y.
    Args:
      x: a tensor of shape [num_x_samples, num_features]
      y: a tensor of shape [num_y_samples, num_features]
    Returns:
      a distance matrix of dimensions [num_x_samples, num_y_samples].
    Raises:
      ValueError: if the inputs do no matched the specified dimensions.
    """

    if not len(x.get_shape()) == len(y.get_shape()) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError('The number of features should be the same.')

    norm = lambda x: tf.reduce_sum(tf.square(x), 1)

    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))


def gaussian_kernel_matrix(x, y, sigmas):
    r"""Computes a Guassian Radial Basis Kernel between the samples of x and y.
    We create a sum of multiple gaussian kernels each having a width sigma_i.
    Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      sigmas: a tensor of floats which denote the widths of each of the
        gaussians in the kernel.
    Returns:
      A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
    """
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

    dist = compute_pairwise_distances(x, y)

    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))

    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))


def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    '''
    Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
    the distributions of x and y. Here we use the kernel two sample estimate
    using the empirical mean of the two distributions.
    MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
                = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },
    where K = <\phi(x), \phi(y)>,
      is the desired kernel function, in this case a radial basis kernel.
    Args:
        x: a tensor of shape [num_samples, num_features]
        y: a tensor of shape [num_samples, num_features]
        kernel: a function which computes the kernel in MMD. Defaults to the
                GaussianKernelMatrix.
    Returns:
        a scalar denoting the squared maximum mean discrepancy loss.
    '''
    with tf.name_scope('MaximumMeanDiscrepancy'):
        # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
        cost = tf.reduce_mean(kernel(x, x))
        cost += tf.reduce_mean(kernel(y, y))
        cost -= 2 * tf.reduce_mean(kernel(x, y))

        # We do not allow the loss to become negative.
        cost = tf.where(cost > 0, cost, 0, name='value')
    return cost


def mmd_loss(source_samples, target_samples, weight, scope=None):
    """Adds a similarity loss term, the MMD between two representations.
    This Maximum Mean Discrepancy (MMD) loss is calculated with a number of
    different Gaussian kernels.
    Args:
      source_samples: a tensor of shape [num_samples, num_features].
      target_samples: a tensor of shape [num_samples, num_features].
      weight: the weight of the MMD loss.
      scope: optional name scope for summary tags.
    Returns:
      a scalar tensor representing the MMD loss value.
    """
    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    gaussian_kernel = partial(
        gaussian_kernel_matrix, sigmas=tf.constant(sigmas))

    loss_value = maximum_mean_discrepancy(
        source_samples, target_samples, kernel=gaussian_kernel)
    loss_value = tf.maximum(1e-4, loss_value) * weight

    return loss_value


def probe_model(model, x_, y_true, m_true):
    c_ = model.predict(x_)
    shuffled_index = tf.random.shuffle(tf.range(c_.shape[0]))
    return x_[shuffled_index], c_[shuffled_index], y_true[shuffled_index], m_true[shuffled_index]


def evaluate_attack(m_true, m_pred):
    accuracy = tf.keras.metrics.Accuracy()
    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()
    accuracy.update_state(m_true, m_pred)
    precision.update_state(m_true, m_pred)
    recall.update_state(m_true, m_pred)
    F1_Score = 2 * (precision.result() * recall.result()) / (precision.result() + recall.result())
    print('accuracy:%.4f precision:%.4f recall:%.4f F1_Score:%.4f'
          % (accuracy.result(), precision.result(), recall.result(), F1_Score))
