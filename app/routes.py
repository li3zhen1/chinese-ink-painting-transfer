# os
from app import app
import numpy as np
import PIL
import scipy
import tensorflow as tf
from flask import (Flask, flash, jsonify, make_response, redirect,
                   render_template, request, send_from_directory, url_for)


import io
import os
import random
import string
import tempfile
from os import listdir, mkdir, sep
from os.path import exists, join, splitext


# flask extensions
from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, FileField, FileRequired
from PIL import ImageEnhance
from scipy.misc import imread, imresize, imsave
from tensorflow.contrib.slim.python.slim.nets import inception_v3
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops as framework_ops
from tensorflow.python.ops import random_ops, variable_scope
from werkzeug.utils import secure_filename
from wtforms import SubmitField

import cv2


config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.98)
config.gpu_options.allow_growth = True

# style transfer
slim = tf.contrib.slim


# image_utils
def load_np_image_uint8(image_file):

    with tempfile.NamedTemporaryFile() as f:
        f.write(tf.gfile.GFile(image_file, 'rb').read())
        f.flush()
        image = scipy.misc.imread(f)
        # Workaround for black-and-white images
        if image.ndim == 2:
            image = np.tile(image[:, :, None], (1, 1, 3))
        return image


def save_np_image(image, output_file, save_format='jpeg'):

    image = np.uint8(image * 255.0)
    buf = io.BytesIO()
    scipy.misc.imsave(buf, np.squeeze(image, 0), format=save_format)
    buf.seek(0)
    f = tf.gfile.GFile(output_file, 'w')
    f.write(buf.getvalue())
    f.close()


def _smallest_size_at_least(height, width, smallest_side):

    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    height = tf.to_float(height)
    width = tf.to_float(width)
    smallest_side = tf.to_float(smallest_side)

    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)
    return new_height, new_width


def _aspect_preserving_resize(image, smallest_side):
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    input_rank = len(image.get_shape())
    if input_rank == 3:
        image = tf.expand_dims(image, 0)

    shape = tf.shape(image)
    height = shape[1]
    width = shape[2]
    new_height, new_width = _smallest_size_at_least(
        height, width, smallest_side)
    resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                             align_corners=False)
    if input_rank == 3:
        resized_image = tf.squeeze(resized_image)
        resized_image.set_shape([None, None, 3])
    else:
        resized_image.set_shape([None, None, None, 3])
    return resized_image


def resize_image(image, image_size):

    image = _aspect_preserving_resize(image, image_size)
    image = tf.to_float(image) / 255.0

    return tf.expand_dims(image, 0)


# vgg
def vgg_16(inputs, reuse=False, pooling='avg', final_endpoint='fc8'):

    inputs *= 255.0
    inputs -= tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

    pooling_fns = {'avg': slim.avg_pool2d, 'max': slim.max_pool2d}
    pooling_fn = pooling_fns[pooling]

    with tf.variable_scope('vgg_16', [inputs], reuse=reuse) as sc:
        end_points = {}

        def add_and_check_is_final(layer_name, net):
            end_points['%s/%s' % (sc.name, layer_name)] = net
            return layer_name == final_endpoint

        with slim.arg_scope([slim.conv2d], trainable=False):
            net = slim.repeat(inputs, 2, slim.conv2d,
                              64, [3, 3], scope='conv1')
            if add_and_check_is_final('conv1', net):
                return end_points
            net = pooling_fn(net, [2, 2], scope='pool1')
            if add_and_check_is_final('pool1', net):
                return end_points
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            if add_and_check_is_final('conv2', net):
                return end_points
            net = pooling_fn(net, [2, 2], scope='pool2')
            if add_and_check_is_final('pool2', net):
                return end_points
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            if add_and_check_is_final('conv3', net):
                return end_points
            net = pooling_fn(net, [2, 2], scope='pool3')
            if add_and_check_is_final('pool3', net):
                return end_points
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            if add_and_check_is_final('conv4', net):
                return end_points
            net = pooling_fn(net, [2, 2], scope='pool4')
            if add_and_check_is_final('pool4', net):
                return end_points
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            if add_and_check_is_final('conv5', net):
                return end_points
            net = pooling_fn(net, [2, 2], scope='pool5')
            if add_and_check_is_final('pool5', net):
                return end_points
            # Use conv2d instead of fully_connected layers.
            net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
            if add_and_check_is_final('fc6', net):
                return end_points
            net = slim.dropout(net, 0.5, is_training=False, scope='dropout6')
            net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
            if add_and_check_is_final('fc7', net):
                return end_points
            net = slim.dropout(net, 0.5, is_training=False, scope='dropout7')
            net = slim.conv2d(net, 1000, [1, 1], activation_fn=None,
                              scope='fc8')
            end_points[sc.name + '/predictions'] = slim.softmax(net)
            if add_and_check_is_final('fc8', net):
                return end_points

        raise ValueError('final_endpoint (%s) not recognized' % final_endpoint)


# ops
@slim.add_arg_scope
def conditional_instance_norm(inputs,
                              labels,
                              num_categories,
                              center=True,
                              scale=True,
                              activation_fn=None,
                              reuse=None,
                              variables_collections=None,
                              outputs_collections=None,
                              trainable=True,
                              scope=None):

    with tf.variable_scope(scope, 'InstanceNorm', [inputs],
                           reuse=reuse) as sc:
        inputs = tf.convert_to_tensor(inputs)
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims
        if inputs_rank is None:
            raise ValueError('Inputs %s has undefined rank.' % inputs.name)
        if inputs_rank != 4:
            raise ValueError('Inputs %s is not a 4D tensor.' % inputs.name)
        dtype = inputs.dtype.base_dtype
        axis = [1, 2]
        params_shape = inputs_shape[-1:]
        if not params_shape.is_fully_defined():
            raise ValueError('Inputs %s has undefined last dimension %s.' % (
                inputs.name, params_shape))

        def _label_conditioned_variable(name, initializer, labels, num_categories):
            """Label conditioning."""
            shape = tf.TensorShape([num_categories]).concatenate(params_shape)
            var_collections = slim.utils.get_variable_collections(
                variables_collections, name)
            var = slim.model_variable(name,
                                      shape=shape,
                                      dtype=dtype,
                                      initializer=initializer,
                                      collections=var_collections,
                                      trainable=trainable)
            conditioned_var = tf.gather(var, labels)
            conditioned_var = tf.expand_dims(
                tf.expand_dims(conditioned_var, 1), 1)
            return conditioned_var

        # Allocate parameters for the beta and gamma of the normalization.
        beta, gamma = None, None
        if center:
            beta = _label_conditioned_variable(
                'beta', tf.zeros_initializer(), labels, num_categories)
        if scale:
            gamma = _label_conditioned_variable(
                'gamma', tf.ones_initializer(), labels, num_categories)
        # Calculate the moments on the last axis (instance activations).
        mean, variance = tf.nn.moments(inputs, axis, keep_dims=True)
        # Compute layer normalization using the batch_normalization function.
        variance_epsilon = 1E-5
        outputs = tf.nn.batch_normalization(
            inputs, mean, variance, beta, gamma, variance_epsilon)
        outputs.set_shape(inputs_shape)
        if activation_fn:
            outputs = activation_fn(outputs)
        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.original_name_scope,
                                                outputs)


@slim.add_arg_scope
def conditional_style_norm(inputs,
                           style_params=None,
                           activation_fn=None,
                           reuse=None,
                           outputs_collections=None,
                           check_numerics=True,
                           scope=None):

    with variable_scope.variable_scope(
            scope, 'StyleNorm', [inputs], reuse=reuse) as sc:
        inputs = framework_ops.convert_to_tensor(inputs)
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims
        if inputs_rank is None:
            raise ValueError('Inputs %s has undefined rank.' % inputs.name)
        if inputs_rank != 4:
            raise ValueError('Inputs %s is not a 4D tensor.' % inputs.name)
        axis = [1, 2]
        params_shape = inputs_shape[-1:]
        if not params_shape.is_fully_defined():
            raise ValueError('Inputs %s has undefined last dimension %s.' %
                             (inputs.name, params_shape))

        def _style_parameters(name):
            """Gets style normalization parameters."""
            var = style_params[('{}/{}'.format(sc.name, name))]

            if check_numerics:
                var = tf.check_numerics(var, 'NaN/Inf in {}'.format(var.name))
            if var.get_shape().ndims < 2:
                var = tf.expand_dims(var, 0)
            var = tf.expand_dims(tf.expand_dims(var, 1), 1)

            return var

        # Allocates parameters for the beta and gamma of the normalization.
        beta = _style_parameters('beta')
        gamma = _style_parameters('gamma')

        # Calculates the moments on the last axis (instance activations).
        mean, variance = tf.nn.moments(inputs, axis, keep_dims=True)

        # Compute layer normalization using the batch_normalization function.
        variance_epsilon = 1E-5
        outputs = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma,
                                            variance_epsilon)
        outputs.set_shape(inputs_shape)
        if activation_fn:
            outputs = activation_fn(outputs)
        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.original_name_scope, outputs)


# model
def transform(input_, normalizer_fn=conditional_instance_norm,
              normalizer_params=None, reuse=False):

    if normalizer_params is None:
        normalizer_params = {'center': True, 'scale': True}
    with tf.variable_scope('transformer', reuse=reuse):
        with slim.arg_scope(
            [slim.conv2d],
            activation_fn=tf.nn.relu,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params,
            weights_initializer=tf.random_normal_initializer(0.0, 0.01),
                biases_initializer=tf.constant_initializer(0.0)):
            with tf.variable_scope('contract'):
                h = model_util_conv2d(input_, 9, 1, 32, 'conv1')
                h = model_util_conv2d(h, 3, 2, 64, 'conv2')
                h = model_util_conv2d(h, 3, 2, 128, 'conv3')
            with tf.variable_scope('residual'):
                h = model_util_residual_block(h, 3, 'residual1')
                h = model_util_residual_block(h, 3, 'residual2')
                h = model_util_residual_block(h, 3, 'residual3')
                h = model_util_residual_block(h, 3, 'residual4')
                h = model_util_residual_block(h, 3, 'residual5')
            with tf.variable_scope('expand'):
                h = model_util_upsampling(h, 3, 2, 64, 'conv1')
                h = model_util_upsampling(h, 3, 2, 32, 'conv2')
                return model_util_upsampling(h, 9, 1, 3, 'conv3', activation_fn=tf.nn.sigmoid)


def model_util_conv2d(input_,
                      kernel_size,
                      stride,
                      num_outputs,
                      scope,
                      activation_fn=tf.nn.relu):

    if kernel_size % 2 == 0:
        raise ValueError('kernel_size is expected to be odd.')
    padding = kernel_size // 2
    padded_input = tf.pad(
        input_, [[0, 0], [padding, padding], [padding, padding], [0, 0]],
        mode='REFLECT')
    return slim.conv2d(
        padded_input,
        padding='VALID',
        kernel_size=kernel_size,
        stride=stride,
        num_outputs=num_outputs,
        activation_fn=activation_fn,
        scope=scope)


def model_util_upsampling(input_,
                          kernel_size,
                          stride,
                          num_outputs,
                          scope,
                          activation_fn=tf.nn.relu):

    if kernel_size % 2 == 0:
        raise ValueError('kernel_size is expected to be odd.')
    with tf.variable_scope(scope):
        shape = tf.shape(input_)
        height = shape[1]
        width = shape[2]
        upsampled_input = tf.image.resize_nearest_neighbor(
            input_, [stride * height, stride * width])
        return model_util_conv2d(
            upsampled_input,
            kernel_size,
            1,
            num_outputs,
            'conv',
            activation_fn=activation_fn)


def model_util_residual_block(input_, kernel_size, scope, activation_fn=tf.nn.relu):

    if kernel_size % 2 == 0:
        raise ValueError('kernel_size is expected to be odd.')
    with tf.variable_scope(scope):
        num_outputs = input_.get_shape()[-1].value
        h_1 = model_util_conv2d(input_, kernel_size, 1,
                                num_outputs, 'conv1', activation_fn)
        h_2 = model_util_conv2d(h_1, kernel_size, 1,
                                num_outputs, 'conv2', None)
        return input_ + h_2


# learning

def total_variation_loss(stylized_inputs, total_variation_weight):

    shape = tf.shape(stylized_inputs)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    channels = shape[3]
    y_size = tf.to_float((height - 1) * width * channels)
    x_size = tf.to_float(height * (width - 1) * channels)
    y_loss = tf.nn.l2_loss(
        stylized_inputs[:, 1:, :, :] - stylized_inputs[:, :-1, :, :]) / y_size
    x_loss = tf.nn.l2_loss(
        stylized_inputs[:, :, 1:, :] - stylized_inputs[:, :, :-1, :]) / x_size
    loss = (y_loss + x_loss) / tf.to_float(batch_size)
    weighted_loss = loss * total_variation_weight
    return weighted_loss, {
        'total_variation_loss': loss,
        'weighted_total_variation_loss': weighted_loss
    }


def gram_matrix(feature_maps):
    batch_size, height, width, channels = tf.unstack(tf.shape(feature_maps))
    denominator = tf.to_float(height * width)
    feature_maps = tf.reshape(
        feature_maps, tf.stack([batch_size, height * width, channels]))
    matrix = tf.matmul(feature_maps, feature_maps, adjoint_a=True)
    return matrix / denominator

# outer
# loss


def total_loss(content_inputs, style_inputs, stylized_inputs, content_weights,
               style_weights, total_variation_weight, reuse=False):

    with tf.name_scope('content_endpoints'):
        content_end_points = vgg_16(content_inputs, reuse=reuse)
    with tf.name_scope('style_endpoints'):
        style_end_points = vgg_16(style_inputs, reuse=True)
    with tf.name_scope('stylized_endpoints'):
        stylized_end_points = vgg_16(stylized_inputs, reuse=True)

    # Compute the content loss
    with tf.name_scope('content_loss'):
        total_content_loss, content_loss_dict = content_loss(
            content_end_points, stylized_end_points, content_weights)

    # Compute the style loss
    with tf.name_scope('style_loss'):
        total_style_loss, style_loss_dict = style_loss(
            style_end_points, stylized_end_points, style_weights)

    # Compute the total variation loss
    with tf.name_scope('total_variation_loss'):
        tv_loss, total_variation_loss_dict = total_variation_loss(
            stylized_inputs, total_variation_weight)

    # Compute the total loss
    with tf.name_scope('total_loss'):
        loss = total_content_loss + total_style_loss + tv_loss

    loss_dict = {'total_loss': loss}
    loss_dict.update(content_loss_dict)
    loss_dict.update(style_loss_dict)
    loss_dict.update(total_variation_loss_dict)

    return loss, loss_dict


def content_loss(end_points, stylized_end_points, content_weights):

    total_content_loss = np.float32(0.0)
    content_loss_dict = {}

    for name, weight in content_weights.items():
        loss = tf.reduce_mean(
            (end_points[name] - stylized_end_points[name]) ** 2)
        weighted_loss = weight * loss

        content_loss_dict['content_loss/' + name] = loss
        content_loss_dict['weighted_content_loss/' + name] = weighted_loss
        total_content_loss += weighted_loss

    content_loss_dict['total_content_loss'] = total_content_loss

    return total_content_loss, content_loss_dict


def style_loss(style_end_points, stylized_end_points, style_weights):
    total_style_loss = np.float32(0.0)
    style_loss_dict = {}

    for name, weight in style_weights.items():
        loss = tf.reduce_mean(
            (gram_matrix(stylized_end_points[name]) -
             gram_matrix(style_end_points[name])) ** 2)
        weighted_loss = weight * loss

        style_loss_dict['style_loss/' + name] = loss
        style_loss_dict['weighted_style_loss/' + name] = weighted_loss
        total_style_loss += weighted_loss

    style_loss_dict['total_style_loss'] = total_style_loss
    return total_style_loss, style_loss_dict


# nza model

def nza_transform(input_, normalizer_fn=None, normalizer_params=None,
                  reuse=False, trainable=True, is_training=True):

    with tf.variable_scope('transformer', reuse=reuse):
        with slim.arg_scope(
            [slim.conv2d],
            activation_fn=tf.nn.relu,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params,
            weights_initializer=tf.random_normal_initializer(0.0, 0.01),
            biases_initializer=tf.constant_initializer(0.0),
                trainable=trainable):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                                normalizer_params=None,
                                trainable=trainable):
                with slim.arg_scope([slim.batch_norm], is_training=is_training,
                                    trainable=trainable):
                    with tf.variable_scope('contract'):
                        h = model_util_conv2d(input_, 9, 1, 32, 'conv1')
                        h = model_util_conv2d(h, 3, 2, 64, 'conv2')
                        h = model_util_conv2d(h, 3, 2, 128, 'conv3')
            with tf.variable_scope('residual'):
                h = model_util_residual_block(h, 3, 'residual1')
                h = model_util_residual_block(h, 3, 'residual2')
                h = model_util_residual_block(h, 3, 'residual3')
                h = model_util_residual_block(h, 3, 'residual4')
                h = model_util_residual_block(h, 3, 'residual5')
            with tf.variable_scope('expand'):
                h = model_util_upsampling(h, 3, 2, 64, 'conv1')
                h = model_util_upsampling(h, 3, 2, 32, 'conv2')
                return model_util_upsampling(
                    h, 9, 1, 3, 'conv3', activation_fn=tf.nn.sigmoid)


def style_normalization_activations(pre_name='transformer',
                                    post_name='StyleNorm'):
    scope_names = ['residual/residual1/conv1',
                   'residual/residual1/conv2',
                   'residual/residual2/conv1',
                   'residual/residual2/conv2',
                   'residual/residual3/conv1',
                   'residual/residual3/conv2',
                   'residual/residual4/conv1',
                   'residual/residual4/conv2',
                   'residual/residual5/conv1',
                   'residual/residual5/conv2',
                   'expand/conv1/conv',
                   'expand/conv2/conv',
                   'expand/conv3/conv']
    scope_names = ['{}/{}/{}'.format(pre_name, name, post_name)
                   for name in scope_names]
    depths = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 64, 32, 3]

    return scope_names, depths


# build

def build_model(content_input_,
                style_input_,
                trainable,
                is_training,
                reuse=None,
                inception_end_point='Mixed_6e',
                style_prediction_bottleneck=100,
                adds_losses=True,
                content_weights=None,
                style_weights=None,
                total_variation_weight=None):
    [activation_names,
     activation_depths] = style_normalization_activations()

    # Defines the style prediction network.
    style_params, bottleneck_feat = style_prediction(
        style_input_,
        activation_names,
        activation_depths,
        is_training=is_training,
        trainable=trainable,
        inception_end_point=inception_end_point,
        style_prediction_bottleneck=style_prediction_bottleneck,
        reuse=reuse)

    # Defines the style transformer network.
    stylized_images = nza_transform(
        content_input_,
        normalizer_fn=conditional_style_norm,
        reuse=reuse,
        trainable=trainable,
        is_training=is_training,
        normalizer_params={'style_params': style_params})

    # Adds losses.
    loss_dict = {}
    total_loss = []

    return stylized_images, total_loss, loss_dict, bottleneck_feat


def style_prediction(style_input_,
                     activation_names,
                     activation_depths,
                     is_training=True,
                     trainable=True,
                     inception_end_point='Mixed_6e',
                     style_prediction_bottleneck=100,
                     reuse=None):

    with tf.name_scope('style_prediction') and tf.variable_scope(
            tf.get_variable_scope(), reuse=reuse):
        with slim.arg_scope(_inception_v3_arg_scope(is_training=is_training)):
            with slim.arg_scope(
                [slim.conv2d, slim.fully_connected, slim.batch_norm],
                    trainable=trainable):
                with slim.arg_scope(
                        [slim.batch_norm, slim.dropout], is_training=is_training):
                    _, end_points = inception_v3.inception_v3_base(
                        style_input_,
                        scope='InceptionV3',
                        final_endpoint=inception_end_point)

        # Shape of feat_convlayer is (batch_size, ?, ?, depth).
        # For Mixed_6e end point, depth is 768, for input image size of 256x265
        # width and height are 14x14.
        feat_convlayer = end_points[inception_end_point]
        with tf.name_scope('bottleneck'):
            # (batch_size, 1, 1, depth).
            bottleneck_feat = tf.reduce_mean(
                feat_convlayer, axis=[1, 2], keep_dims=True)

        if style_prediction_bottleneck > 0:
            with slim.arg_scope(
                [slim.conv2d],
                activation_fn=None,
                normalizer_fn=None,
                    trainable=trainable):
                # (batch_size, 1, 1, style_prediction_bottleneck).
                bottleneck_feat = slim.conv2d(bottleneck_feat,
                                              style_prediction_bottleneck, [1, 1])

        style_params = {}
        with tf.variable_scope('style_params'):
            for i in range(len(activation_depths)):
                with tf.variable_scope(activation_names[i], reuse=reuse):
                    with slim.arg_scope(
                        [slim.conv2d],
                        activation_fn=None,
                        normalizer_fn=None,
                            trainable=trainable):

                        # Computing beta parameter of the style normalization for the
                        # activation_names[i] layer of the style transformer network.
                        # (batch_size, 1, 1, activation_depths[i])
                        beta = slim.conv2d(
                            bottleneck_feat, activation_depths[i], [1, 1])
                        # (batch_size, activation_depths[i])
                        beta = tf.squeeze(beta, [1, 2], name='SpatialSqueeze')
                        style_params['{}/beta'.format(
                            activation_names[i])] = beta

                        # Computing gamma parameter of the style normalization for the
                        # activation_names[i] layer of the style transformer network.
                        # (batch_size, 1, 1, activation_depths[i])
                        gamma = slim.conv2d(
                            bottleneck_feat, activation_depths[i], [1, 1])
                        # (batch_size, activation_depths[i])
                        gamma = tf.squeeze(
                            gamma, [1, 2], name='SpatialSqueeze')
                        style_params['{}/gamma'.format(
                            activation_names[i])] = gamma

    return style_params, bottleneck_feat


def _inception_v3_arg_scope(is_training=True,
                            weight_decay=0.00004,
                            stddev=0.1,
                            batch_norm_var_collection='moving_vars'):

    batch_norm_params = {
        'is_training': is_training,
        # Decay for the moving averages.
        'decay': 0.9997,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # collection containing the moving mean and moving variance.
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection],
        }
    }
    normalizer_fn = slim.batch_norm

    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
            weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
            [slim.conv2d],
            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
            activation_fn=tf.nn.relu6,
            normalizer_fn=normalizer_fn,
                normalizer_params=batch_norm_params) as sc:
            return sc


checkpoint_path = './app/arbitrary_style_transfer/model.ckpt'
style_image_size = 256
image_size = 512


tf.Graph().as_default()
sess = tf.Session(config=config)
style_img_ph = tf.placeholder(tf.float32, shape=[None, None, 3])
style_img_preprocessed = resize_image(
    style_img_ph, style_image_size)
content_img_ph = tf.placeholder(tf.float32, shape=[None, None, 3])
content_img_preprocessed = resize_image(
    content_img_ph, image_size)
stylized_images, _, _, bottleneck_feat = build_model(
    content_img_preprocessed,
    style_img_preprocessed,
    trainable=False,
    is_training=False,
    inception_end_point='Mixed_6e',
    style_prediction_bottleneck=100,
    adds_losses=False)

checkpoint = checkpoint_path

init_fn = slim.assign_from_checkpoint_fn(checkpoint,
                                         slim.get_variables_to_restore())
sess.run([tf.local_variables_initializer()])
init_fn(sess)


def Stylize(style_images_paths,
            content_images_paths,
            output_dir,
            casename):
    style_img_list = tf.gfile.Glob(style_images_paths+casename+'/*.jpg')
    content_img_list = tf.gfile.Glob(content_images_paths+casename+'/*.jpg')

    for content_img_path in content_img_list:
        content_img_np = load_np_image_uint8(content_img_path)[
            :, :, :3]

        for style_img_path in style_img_list:
            style_image_np = load_np_image_uint8(style_img_path)[
                :, :, :3]

            style_params = sess.run(
                bottleneck_feat, feed_dict={style_img_ph: style_image_np})

            stylized_image_res = sess.run(
                stylized_images,
                feed_dict={
                    bottleneck_feat:
                    style_params,
                    content_img_ph:
                    content_img_np
                })
            save_np_image(stylized_image_res,
                          output_dir+casename+'.jpg')


# forms
class TransferForm(FlaskForm):
    StyleImg = FileField('Select an image as Style', validators=[
                         FileRequired(),
                         FileAllowed(['jpg', 'bmp', 'jpeg', 'JPG', 'JPEG'],
                                     message='Please upload an image in jpg/png/bmp format.')
                         ])
    ContentImg = FileField('Select an image as Content', validators=[
                           FileRequired(),
                           FileAllowed(['jpg', 'bmp', 'jpeg', 'JPG', 'JPEG'],
                                       message='Please upload an image in jpg/png/bmp format.')
                           ])
    submit = SubmitField('Get Started!')


class InkTransferForm(FlaskForm):
    ContentImg = FileField('Select an image as Content', validators=[
                           FileRequired('No Image Selected'),
                           FileAllowed(
                               ['jpg', 'png', 'jpeg', 'bmp'], 'Supported format')
                           ])
    submit = SubmitField('Get Started!')


def BilateralBlur(imagePath, Gaussian_standard_deviation, Grey_scale_deviation):
    src_img = cv2.imread(imagePath)
    rst_img = (cv2.bilateralFilter(
        src_img, 0, Gaussian_standard_deviation, Grey_scale_deviation))
    cv2.imwrite(imagePath, rst_img)


def _BilateralBlur(imagePath, Gaussian_standard_deviation, Grey_scale_deviation):
    src_img = cv2.imread(imagePath)
    rst_img = (cv2.bilateralFilter(
        src_img, 0, Gaussian_standard_deviation, Grey_scale_deviation))[:, :, ::-1]
    return rst_img


def edgeMask(imagePath):
    srcImage = PIL.Image.open(imagePath)
    contrast = PIL.ImageEnhance.Contrast(srcImage)
    srcImage = contrast.enhance(1.17)
    mask = PIL.Image.open('./app/mask/mask.jpg')
    mask = mask.resize(srcImage.size)
    srcImage = PIL.ImageChops.screen(srcImage, mask)
    srcImage.save(imagePath)


def Oldify(imagePath):
    srcImage = PIL.Image.fromarray(_BilateralBlur(imagePath, 30, 20))
    contrast = PIL.ImageEnhance.Contrast(srcImage)
    srcImage = contrast.enhance(1.22)
    bright = PIL.ImageEnhance.Brightness(srcImage)
    srcImage = bright.enhance(1.1)
    texture_inkpainting = PIL.Image.open('./app/mask/texture.jpg')
    texture_inkpainting = texture_inkpainting.resize(srcImage.size)
    result_ink = PIL.ImageChops.multiply(srcImage, texture_inkpainting)
    result_ink.save(imagePath)


def inkStylize(style_images_paths,
               content_images_paths,
               output_dir,
               casename):
    style_img_list = tf.gfile.Glob(style_images_paths+'/ink/*.jpg')
    content_img_list = tf.gfile.Glob(content_images_paths+casename+'/*.jpg')
    for content_img_path in content_img_list:
        content_img_np = load_np_image_uint8(content_img_path)[
            :, :, :3]

        for style_img_path in style_img_list:
            style_image_np = load_np_image_uint8(style_img_path)[
                :, :, :3]

            style_params = sess.run(
                bottleneck_feat, feed_dict={style_img_ph: style_image_np})

            stylized_image_res = sess.run(
                stylized_images,
                feed_dict={
                    bottleneck_feat:
                    style_params,
                    content_img_ph:
                    content_img_np
                })
            save_np_image(stylized_image_res,
                          output_dir+casename+'.jpg')


def ContrastEnhance(imagePath):
    srcImage = PIL.Image.open(imagePath)
    contrast = PIL.ImageEnhance.Contrast(srcImage)
    srcImage = contrast.enhance(1.1)
    srcImage.save(imagePath)


# parameter
STYLE_DIR = './app/style/'
CONTENT_DIR = './app/content/'
OUTPUT_DIR = './app/output/'


def environmentCreate(caseName):
    mkdir(STYLE_DIR+caseName)
    mkdir(CONTENT_DIR+caseName)


# routes
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/result')
def result():
    return render_template('result.html')


@app.route('/about')
def members():
    return render_template('about.html')


@app.route('/credits')
def credits():
    return render_template('credits.html')


@app.route('/<caseresult>', methods=['GET'])
def resultImg(caseresult):
    imgPath = './output'
    return send_from_directory(imgPath, caseresult)


@app.route('/fonts/<fontname>', methods=['GET'])
def fontfile(fontname):
    fontPath = './templates/fonts'
    return send_from_directory(fontPath, fontname)


@app.route('/imagesrc/<srcname>', methods=['GET'])
def srcfile(srcname):
    srcPath = './templates/imagesrc'
    return send_from_directory(srcPath, srcname)


@app.route('/start', methods=['GET', 'POST'])
def start():
    picForm = TransferForm()
    if picForm.validate_on_submit():
        transferCaseName = ''.join(random.sample(
            string.ascii_letters + string.digits, 12))
        environmentCreate(transferCaseName)
        backslash_Style_Path = '/style.jpg'
        backslash_Content_Path = '/content.jpg'

        contentImage_Uploaded = picForm.ContentImg.data
        contentImage_Uploaded.save(
            CONTENT_DIR + transferCaseName + backslash_Content_Path)

        styleImage_Uploaded = picForm.StyleImg.data
        styleImage_Uploaded.save(
            STYLE_DIR + transferCaseName + backslash_Style_Path)

        Stylize(STYLE_DIR, CONTENT_DIR, OUTPUT_DIR, transferCaseName)
        BilateralBlur(OUTPUT_DIR+transferCaseName+'.jpg', 24, 12)
        ContrastEnhance(OUTPUT_DIR+transferCaseName+'.jpg')
        return render_template('result.html', resultPath='../'+transferCaseName+'.jpg', goback='/start')
    return render_template('transfer.html', form=picForm)


@app.route('/ink', methods=['GET', 'POST'])
def inkstart():
    inkForm = InkTransferForm()
    if inkForm.validate_on_submit():
        transferCaseName = ''.join(random.sample(
            string.ascii_letters + string.digits, 12))

        environmentCreate(transferCaseName)
        backslash_Content_Path = '/content.jpg'

        contentImage_Uploaded = inkForm.ContentImg.data
        contentPath_temp = CONTENT_DIR + transferCaseName + backslash_Content_Path
        contentImage_Uploaded.save(contentPath_temp)

        edgeMask(contentPath_temp)

        inkStylize(STYLE_DIR, CONTENT_DIR, OUTPUT_DIR, transferCaseName)

        Oldify(OUTPUT_DIR+transferCaseName+'.jpg')

        return render_template('result.html',  resultPath='../'+transferCaseName+'.jpg', goback='/ink')
    return render_template('inktransfer.html', form=inkForm)
