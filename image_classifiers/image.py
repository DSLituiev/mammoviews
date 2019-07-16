"""Fairly basic set of tools for real-time data augmentation on image data.
Can easily be extended to include new transformations,
new preprocessing methods, etc...
"""
from __future__ import absolute_import
from __future__ import print_function

import sys
import numpy as np
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
import threading
from warnings import warn
import multiprocessing.pool
from functools import partial
from collections import Counter

from keras import backend as K
from keras.utils.data_utils import Sequence

from histeq import histeq, ztransform
try:
    from PIL import ImageEnhance
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, 'HAMMING'):
        _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
    if hasattr(pil_image, 'BOX'):
        _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, 'LANCZOS'):
        _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS


import cv2

_OPENCV_INTERPOLATION_METHODS = {
    "nearest":cv2.INTER_NEAREST,
    "bilinear":cv2.INTER_LINEAR,
    "linear":cv2.INTER_LINEAR,
    "area":cv2.INTER_AREA,
    "bicubic":cv2.INTER_CUBIC,
    "cubic":cv2.INTER_CUBIC,
    "lanczos":cv2.INTER_LANCZOS4,
    "lanczos4":cv2.INTER_LANCZOS4,
}

import json
from croppad import crop_pad_center

try:
    from pycocotools.mask import encode, decode
except:
    warn('no pycocotools found')


def random_rotation(x, rg, row_axis=1, col_axis=2, channel_axis=0,
                    fill_mode='nearest', cval=0.):
    """Performs a random rotation of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        rg: Rotation range, in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Rotated Numpy image tensor.
    """
    theta = np.pi / 180 * np.random.uniform(-rg, rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_affine_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_shift(x, wrg, hrg, row_axis=1, col_axis=2, channel_axis=0,
                 fill_mode='nearest', cval=0.):
    """Performs a random spatial shift of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        wrg: Width shift range, as a float fraction of the width.
        hrg: Height shift range, as a float fraction of the height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Shifted Numpy image tensor.
    """
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = np.random.uniform(-hrg, hrg) * h
    ty = np.random.uniform(-wrg, wrg) * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    x = apply_affine_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_shear(x, intensity, row_axis=1, col_axis=2, channel_axis=0,
                 fill_mode='nearest', cval=0.):
    """Performs a random spatial shear of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        intensity: Transformation intensity.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Sheared Numpy image tensor.
    """
    shear = np.random.uniform(-intensity, intensity)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    x = apply_affine_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_zoom(x, zoom_range, row_axis=1, col_axis=2, channel_axis=0,
                fill_mode='nearest', cval=0.):
    """Performs a random spatial zoom of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        zoom_range: Tuple of floats; zoom range for width and height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Zoomed Numpy image tensor.

    # Raises
        ValueError: if `zoom_range` isn't a tuple.
    """
    if len(zoom_range) != 2:
        raise ValueError('`zoom_range` should be a tuple or list of two floats. '
                         'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_affine_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_channel_shift(x, intensity, channel_axis=0):
    x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_affine_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.,
                    borderMode = cv2.BORDER_TRANSPARENT,
                    interp = cv2.INTER_NEAREST,
                    use_opencv=False):
    """Apply the image transformation specified by a matrix.

    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        use_opencv: 
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        borderMode:
            cv2.BORDER_TRANSPARENT
            cv2.BORDER_CONSTANT
            BORDER_REPLICATE
            BORDER_REFLECT
            BORDER_REFLECT101
            BORDER_WRAP
        interp: 
            cv2.INTER_NEAREST
            cv2.INTER_LINEAR
            cv2.INTER_AREA
            cv2.INTER_CUBIC
            cv2.INTER_LANCZOS4

    # Returns
        The transformed version of the input.
    """
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]

    if use_opencv:
        dsize=x.shape[:2]
        init_shape = x.shape
        #dest = np.ones_like(x) * cval
        x = cv2.warpAffine(x, transform_matrix[:2,:],
                           dsize,
                           #dest,
                           borderValue=cval,
                           borderMode = borderMode,
                           flags=interp,
                           )
        if len(x.shape) < len(init_shape):
            x = x[..., np.newaxis]
        #x = dest 
    elif len(x.shape)>2 and channel_axis is not None:
        x = np.rollaxis(x, channel_axis, 0)
        channel_images = [ndi.interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=0,
            mode=fill_mode,
            cval=cval) for x_channel in x]

        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
    else:
        x = ndi.interpolation.affine_transform(
            x,
            final_affine_matrix,
            final_offset,
            order=0,
            mode=fill_mode,
            cval=cval)
    return x


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def array_to_img(x, data_format=None, scale=True):
    """Converts a 3D Numpy array to a PIL Image instance.

    # Arguments
        x: Input Numpy array.
        data_format: Image data format.
        scale: Whether to rescale image values
            to be within [0, 255].

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    x = np.asarray(x, dtype=K.floatx())
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape:', x.shape)

    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format:', data_format)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)
    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: ', x.shape[2])


def img_to_array(img, data_format=None):
    """Converts a PIL Image instance to a Numpy array.

    # Arguments
        img: PIL Image instance.
        data_format: Image data format.

    # Returns
        A 3D Numpy array.

    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=K.floatx())
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x


def load_img(path, grayscale=False, color_mode='rgb', target_size=None,
             interpolation='nearest', driver='opencv'):
    """Loads an image using a defined module
      # Arguments
        path: Path to image file.
        color_mode: One of "grayscale", "rbg", "rgba", "bgr", "bgra". Default: "rgb".
            The desired image format.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            Driver-dependent options are "lanczos", "area", "box" and
            "hamming". For details, refer to documentation of `load_img_pil()`
            and `load_img_opencv()`. By default, "nearest" is used.
    """
    if driver.lower() in ('cv2','opencv'):
        img = load_img_opencv(path, grayscale=grayscale, color_mode=color_mode,
                              target_size=target_size, interpolation=interpolation)
    elif driver.lower() in ('pil', 'pillow'):
        img = load_img_pil(path, grayscale=grayscale, color_mode=color_mode, 
                              target_size=target_size, interpolation=interpolation)
    return img
        

def load_img_opencv(path, grayscale=False, color_mode='rgb', target_size=None,
             interpolation='nearest'):
    """Loads an image using opencv format.
    # Arguments
        path: Path to image file.
        color_mode: One of "grayscale", "rbg", "rgba", "bgr", "bgra". Default: "rgb".
            The desired image format.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", "bicubic", "lanczos", "area".
            By default, "nearest" is used.
    # Returns
        A numpy array instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """
    if grayscale is True:
        warn('grayscale is deprecated. Please use '
                      'color_mode = "grayscale"')
        color_mode = 'grayscale'
        
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    
    if len(img.shape)==2:
        img_mode = 'grayscale'
    elif len(img.shape) == 3:
        if img.shape[-1]==3:
            img_mode = 'bgr'
        elif img.shape[-1]==4:
            img_mode = 'bgra'

    if img_mode != color_mode:
        if color_mode.startswith('gray'):
            if img_mode == 'bgr':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif img_mode == 'bgra':
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif color_mode == 'rgba':
            if img_mode == 'bgr':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            elif img_mode == 'grayscale':
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
        elif color_mode == 'rgb':
            if img_mode == 'bgr':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img_mode == 'bgra':
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            elif img_mode == 'grayscale':
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError('color_mode must be "grayscale", "rbg", or "rgba"')
            
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.shape[:2][::-1] != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(_OPENCV_INTERPOLATION_METHODS.keys())))
            resample = _OPENCV_INTERPOLATION_METHODS[interpolation]
            img = cv2.resize(img, width_height_tuple, resample)
    return img

    
def load_img_pil(path, grayscale=False, color_mode='rgb', target_size=None,
             interpolation='nearest'):
    """Loads an image into PIL format.
    # Arguments
        path: Path to image file.
        color_mode: One of "grayscale", "rbg", "rgba". Default: "rgb".
            The desired image format.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """
    if grayscale is True:
        warn('grayscale is deprecated. Please use '
                      'color_mode = "grayscale"')
        color_mode = 'grayscale'
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)

    if color_mode == 'grayscale':
        if img.mode not in ('L', 'I;16'):
            img = img.convert('L')
    elif color_mode == 'rgba':
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    elif color_mode == 'rgb':
        if img.mode != 'RGB':
            if img.mode not in ('I;16'):
                img = img.convert('RGB')
            else:
                img = np.asarray(img)
                img = img * (255.0/ max(1.0, img.max()))
                img = np.stack([img.astype('uint8')]*3, axis=-1)
                img = array_to_img(img)
    else:
        raise ValueError('color_mode must be "grayscale", "rbg", or "rgba"')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return img


def array_to_img(x, data_format=None, scale=True):
    """Converts a 3D Numpy array to a PIL Image instance.
    # Arguments
        x: Input Numpy array.
        data_format: Image data format.
            either "channels_first" or "channels_last".
        scale: Whether to rescale image values
            to be within `[0, 255]`.
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    x = np.asarray(x, dtype=K.floatx())
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape:', x.shape)

    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format:', data_format)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)
    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 4:
        # RGBA
        return pil_image.fromarray(x.astype('uint8'), 'RGBA')
    elif x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        if np.all(x < 255):
            return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
        else:
            return pil_image.fromarray(x[:, :, 0].astype('uint16'), 'I;16')
    else:
        raise ValueError('Unsupported channel number: ', x.shape[2])


def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f)]


class ImageDataGenerator(object):
    """Generate minibatches of image data with real-time data augmentation.

    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channel.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided. This is
            applied after the `preprocessing_function` (if any provided)
            but before any other transformation.
        preprocessing_function: function that will be implied on each input.
            The function will run before any other modification on it.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
            (the depth) is at index 1, in 'channels_last' mode it is at index 3.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    """

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 postprocessing_function=None,
                 histeq_alpha = False,
                 contrast_exp = 1.2,
                 contrast = None,
                 truncate_quantile = None,
                 z_transform = None,
                 #noise=None
                 data_format=None):
        if data_format is None:
            data_format = K.image_data_format()
        self.featurewise_center = featurewise_center
        self.samplewise_center = samplewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_std_normalization = samplewise_std_normalization
        self.zca_whitening = zca_whitening
        self.zca_epsilon = zca_epsilon
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale

        self.histeq_alpha = histeq_alpha
        self.contrast = contrast
        self.contrast_exp = contrast_exp
        self.ztransform = z_transform
        self.truncate_quantile = truncate_quantile

        #self.noise_specs = {} if noise is None else noise
        #self.intensity_gain = intensity_gain

        self.preprocessing_function = preprocessing_function
        self.postprocessing_function = postprocessing_function

        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError('`data_format` should be `"channels_last"` (channel after row and '
                             'column) or `"channels_first"` (channel before row and column). '
                             'Received arg: ', data_format)
        self.data_format = data_format
        if data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
        if data_format == 'channels_last':
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 2

        self.mean = None
        self.std = None
        self.principal_components = None

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('`zoom_range` should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', zoom_range)

    def flow(self, x, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='png', color_mode='rgb'):
        return NumpyArrayIterator(
            x, y, self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            postprocessing_function=self.postprocessing_function,
            color_mode=color_mode)

    def flow_from_directory(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            stratify=None,
                            oversampling=False,
                            subsample_factor= None,
                            subsample_num = None,
                            ):
        return DirectoryIterator(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            postprocessing_function=self.postprocessing_function,
            follow_links=follow_links,
            stratify=stratify,
            oversampling=oversampling,
            subsample_factor=subsample_factor,
            subsample_num=subsample_num,
            )


    def flow_patches(self, fn_img, fn_pnt,
                     point_sampler,
                     target_size=(256, 256),
                     color_mode='grayscale',
                     batch_size=4,
                     patches_per_image=1,
                     shuffle=True,
                     seed=0,
                     dtype='uint16',
                     fill_mode = 'reflect',
                     label_freq={1:5, 2:10}, 
                     postprocessing_functions = [None, None],
                     output_indices=False,
                     ):

        return PatchIterator(fn_img, fn_pnt,
                 point_sampler,
                 image_data_generator = self,
                 batch_size=batch_size,
                 patches_per_image=patches_per_image,
                 shuffle=shuffle,
                 color_mode=color_mode,
                 seed=seed,
                 patch_size = target_size,
                 dtype=dtype,
                 mode = self.fill_mode if fill_mode is None else fill_mode,
                 label_freq=label_freq, 
                 augmentation=True,
                 postprocessing_functions=postprocessing_functions,
                 output_indices=output_indices,
                 )

    def flow_memmap(self,
                    root_dir, csv_file,
                    classes = ["Control", "Case",],
                    binary=True,
                    postprocessing_functions=None,
                    nsamples = None,
                    batch_size = 1,
                    shuffle = False,
                    seed = None,
                    postprocessing_function=None,
                    stratify=None,
                    oversampling=False,
                    subsample_factor=None,
                    subsample_num=None,
                    batch_rate=1,
                    dtype = K.floatx(),
                    color_mode=None,
                    label_col = "label",
                    filename_col = "filename",
                    encode_label=None,
                    ):
        return MemMapIterator(root_dir, csv_file,
                 classes = classes,
                 image_data_generator = self,
                 data_format = self.data_format,
                 binary=binary,
                 transform=postprocessing_functions,
                 nsamples = None,
                 batch_size = batch_size,
                 shuffle = shuffle,
                 seed = seed,
                 stratify=stratify,
                 oversampling=oversampling,
                 subsample_factor=subsample_factor,
                 subsample_num=subsample_num,
                 batch_rate=batch_rate,
                 dtype = dtype,
                 color_mode=color_mode,
                 label_col = label_col,
                 filename_col = filename_col,
                 encode_label=encode_label,
                 )
    def standardize(self, x):
        """Apply the normalization configuration to a batch of inputs.

        # Arguments
            x: batch of inputs to be normalized.

        # Returns
            The inputs, normalized.
        """
        if self.histeq_alpha:
            alpha=np.random.random()
            if hasattr(self.histeq_alpha, '__len__'):
                #print("self.histeq_alpha,", self.histeq_alpha,)
                for ii,aa in enumerate(self.histeq_alpha):
                    if aa:
                        x[:,:,ii] = histeq(x[:,:,ii], bitdepth=16, mask=None, alpha=alpha)
            else:
                x = histeq(x, bitdepth=16, mask=None, alpha=alpha)
        if self.ztransform:
            mask = x>0
            mask &= x<x.max()
            if self.contrast is not None:
                contrast = np.clip(self.contrast_exp ** np.random.normal(0,1),
                                   *self.contrast)
            else:
                contrast = None
            x = ztransform(x, mask=None, contrast=contrast,
                           truncate_quantile=self.truncate_quantile)
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.rescale is not None:
            if hasattr(self.rescale, '__len__'):
                for ii, rr in enumerate(self.rescale):
                    if rr is not None:
                        x[:,:,ii] = rr*x[:,:,ii]
            else:
                #print("x", x.dtype)
                #print("self.rescale", type(self.rescale), self.rescale)
                x *= self.rescale
        # x is a single image, so it doesn't have image number at index 0
        img_channel_axis = self.channel_axis - 1
        if self.samplewise_center:
            x -= np.mean(x, axis=img_channel_axis, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_axis, keepdims=True) + 1e-7)

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                warn('This ImageDataGenerator specifies '
                              '`featurewise_center`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + 1e-7)
            else:
                warn('This ImageDataGenerator specifies '
                              '`featurewise_std_normalization`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.zca_whitening:
            if self.principal_components is not None:
                flatx = np.reshape(x, (-1, np.prod(x.shape[-3:])))
                whitex = np.dot(flatx, self.principal_components)
                x = np.reshape(whitex, x.shape)
            else:
                warn('This ImageDataGenerator specifies '
                              '`zca_whitening`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        return x

    
    def get_geom_transform(self, nrows, ncols, seed=None):
        """Randomly augment a single image tensor.

        # Arguments
            x: 3D tensor, single image.
            seed: random seed.

        # Returns
            A randomly transformed version of the input (same shape).
        """

        if seed is not None:
            np.random.seed(seed)

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * nrows 
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * ncols
        else:
            ty = 0

        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                    [0, np.cos(shear), 0],
                                    [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            transform_matrix = transform_matrix_offset_center(transform_matrix, nrows, ncols)
        return transform_matrix

    def get_random_transform_specs(self, x, seed=None):
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1
        nrows = x.shape[img_row_axis]
        ncols = x.shape[img_col_axis]

        transform_matrix = self.get_geom_transform(nrows, ncols, seed=seed)
        if self.horizontal_flip:
            horizontal_flip = np.random.random() < 0.5
        else:
            horizontal_flip = False

        if self.vertical_flip:
            vertical_flip = np.random.random() < 0.5
        else:
            vertical_flip = False
        return [transform_matrix, horizontal_flip, vertical_flip]
       

    def random_transform(self, x, seed=None, fill_mode=None, cval=None,
                         borderMode = cv2.BORDER_TRANSPARENT,
                         interp = cv2.INTER_NEAREST,
                         use_opencv=False):
        """fill_mode and cval can be supplied to the function directly"""
        if fill_mode is None:
            fill_mode = self.fill_mode
        if cval is None:
            cval = self.cval

        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1
        nrows = x.shape[img_row_axis]
        ncols = x.shape[img_col_axis]

        transform_matrix = self.get_geom_transform(nrows, ncols, seed=seed)
        
        if transform_matrix is not None:
            x = apply_affine_transform(x, transform_matrix, img_channel_axis,
                                fill_mode=fill_mode, cval=cval,
                                borderMode = borderMode,
                                interp = interp,
                                use_opencv=use_opencv)
        if self.channel_shift_range != 0:
            x = random_channel_shift(x,
                                     self.channel_shift_range,
                                     img_channel_axis)
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_axis)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_axis)
        return x

    def apply_transform(self, x, transform_matrix, horizontal_flip, vertical_flip,
                         fill_mode=None, cval=None,
                         borderMode = cv2.BORDER_TRANSPARENT,
                         interp = cv2.INTER_NEAREST,
                         use_opencv=False,
                         **kwargs):
        img_channel_axis = self.channel_axis - 1
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        if transform_matrix is not None:
            x = apply_affine_transform(x, transform_matrix, img_channel_axis,
                                fill_mode=fill_mode, cval=cval,
                                borderMode = borderMode,
                                interp = interp,
                                use_opencv=use_opencv)
        if horizontal_flip:
            x = flip_axis(x, img_col_axis)

        if vertical_flip:
            x = flip_axis(x, img_row_axis)
        return x
        

    def fit(self, x,
            augment=False,
            rounds=1,
            seed=None):
        """Fits internal statistics to some sample data.

        Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.

        # Arguments
            x: Numpy array, the data to fit on. Should have rank 4.
                In case of grayscale data,
                the channels axis should have value 1, and in case
                of RGB data, it should have value 3.
            augment: Whether to fit on randomly augmented samples
            rounds: If `augment`,
                how many augmentation passes to do over the data
            seed: random seed.

        # Raises
            ValueError: in case of invalid input `x`.
        """
        x = np.asarray(x, dtype=K.floatx())
        if x.ndim != 4:
            raise ValueError('Input to `.fit()` should have rank 4. '
                             'Got array with shape: ' + str(x.shape))
        if x.shape[self.channel_axis] not in {1, 3, 4}:
            warn(
                'Expected input to be images (as Numpy array) '
                'following the data format convention "' + self.data_format + '" '
                '(channels on axis ' + str(self.channel_axis) + '), i.e. expected '
                'either 1, 3 or 4 channels on axis ' + str(self.channel_axis) + '. '
                'However, it was passed an array with shape ' + str(x.shape) +
                ' (' + str(x.shape[self.channel_axis]) + ' channels).')

        if seed is not None:
            np.random.seed(seed)

        x = np.copy(x)
        if augment:
            ax = np.zeros(tuple([rounds * x.shape[0]] + list(x.shape)[1:]), dtype=K.floatx())
            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]] = self.random_transform(x[i])
            x = ax

        if self.featurewise_center:
            self.mean = np.mean(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.mean = np.reshape(self.mean, broadcast_shape)
            x -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.std = np.reshape(self.std, broadcast_shape)
            x /= (self.std + K.epsilon())

        if self.zca_whitening:
            flat_x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
            sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
            u, s, _ = linalg.svd(sigma)
            self.principal_components = np.dot(np.dot(u, np.diag(1. / np.sqrt(s + self.zca_epsilon))), u.T)


class Iterator(Sequence):
    """Abstract base class for image data iterators.

    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size:  Integer, size of a batch.
        shuffle:     Boolean, whether to shuffle the data between epochs.
        seed:        Random seeding for data shuffling.
        stratify:    Label to stratify by
        oversampling: Oversample if stratification label is provided to match the multiples of the largest class
    """

    def __init__(self, n, batch_size, shuffle, seed,
            postprocessing_function=None, stratify=None, oversampling=True,
            subsample_factor=None,
            subsample_num=None,
            batch_rate=1,
            ):
        self.stratify = stratify
        self.batch_rate = batch_rate
        self.oversampling = oversampling
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.postprocessing_function = postprocessing_function
        self.index_generator = self._flow_index()
        self.final_num = self.n
        if self.stratify is not None:
            self._prep_stratified()
            if self.oversampling:
                self.final_num = sum(np.asarray(list(self.class_size.values()))>0) * max(self.class_size.values())
        elif (subsample_factor is not None) or (subsample_num is not None):
            if subsample_num is None:
                self.final_num = int(self.n/subsample_factor)
            else:
                print("subsample_num", subsample_num)
                self.final_num = subsample_num
            self.orig_index_array = np.random.randint(0, self.n, size=self.final_num)
        else:
            self.orig_index_array = np.arange(self.n)

    def _prep_stratified(self):
        self.uniq_classes = np.unique(self.stratify)
        #self.class_size = np.bincount(self.stratify)
        self.class_size = Counter(self.stratify)
        
        self.class_inds = {}
        for cc in self.uniq_classes:
            mask = np.asarray(self.stratify) == cc
            self.class_inds[cc] = np.where(mask)[0]

        #ex_per_class = max(self.class_size)
        ex_per_class = max(self.class_size.values())
        self.orig_index_array = []
        #for cc,ss in enumerate(self.class_size):
        for cc in self.uniq_classes:
            ss = self.class_size[cc]
            if ss==0:
                continue
            self.orig_index_array.append(
                np.random.choice(self.class_inds[cc],
                                size=ex_per_class,
                                replace=ss<ex_per_class)
                            )
#         np.random.permutation(self.n)
        self.orig_index_array = np.stack(self.orig_index_array).T.ravel()
            
    def _set_index_array(self):
        self.index_array = self.orig_index_array.copy()
        if self.shuffle:
            if self.stratify is not None and not self.oversampling:
                self.index_array = np.random.choice(self.orig_index_array.copy(),
                                                    size=self.final_num, replace=False)
            else:
                self.index_array = np.random.permutation(self.orig_index_array.copy())
            
    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        _batch_size = self.batch_size // self.batch_rate
        index_array = self.index_array[_batch_size * idx:
                                       _batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)

    def __len__(self):
        if hasattr(self, "batch_rate"):
            _batch_size = self.batch_size // self.batch_rate
        else:
            _batch_size = self.batch_size
        return int(np.ceil(self.n / float(_batch_size)))

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0
        
    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.reset()
        if hasattr(self, "batch_rate"):
            _batch_size = self.batch_size // self.batch_rate
        else:
            _batch_size = self.batch_size

        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            self.batch_index = self.total_batches_seen % self.__len__()
            current_index = (self.batch_index * _batch_size) % self.final_num
            self.total_batches_seen += 1
            yield self.index_array[current_index:
                                   current_index + _batch_size]

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


class NumpyArrayIterator(Iterator):
    """Iterator yielding data from a Numpy array.

    # Arguments
        x: Numpy array of input data.
        y: Numpy array of targets data.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self, x, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 postprocessing_function=None,
                 color_mode=None,
                 stratify=None,
                 oversampling=True
                 ):
        channels_axis = 3 if data_format == 'channels_last' else 1
        self.channels_axis = channels_axis
        if y is not None and len(x) != len(y):
            raise ValueError('X (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' %
                             (np.asarray(x).shape, np.asarray(y).shape))

        if data_format is None:
            data_format = K.image_data_format()
        self.x = np.asarray(x, dtype=K.floatx())

        if self.x.ndim != 4:
            warn('Input data in `NumpyArrayIterator` '
                             'should have rank 4. You passed an array '
                             'with shape\t%s' %  str(self.x.shape))
        else:
            if self.x.shape[channels_axis] not in {1, 3, 4}:
                warn('NumpyArrayIterator is set to use the '
                              'data format convention "' + data_format + '" '
                              '(channels on axis ' + str(channels_axis) + '), i.e. expected '
                              'either 1, 3 or 4 channels on axis ' + str(channels_axis) + '. '
                              'However, it was passed an array with shape ' + str(self.x.shape) +
                              ' (' + str(self.x.shape[channels_axis]) + ' channels).')
        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.color_mode=color_mode
        super(NumpyArrayIterator, self).__init__(x.shape[0], batch_size, shuffle, seed,
                                                stratify=stratify, oversampling=oversampling,
                                                postprocessing_function=postprocessing_function)
        print("self.color_mode", self.color_mode)


    def _get_batches_of_transformed_samples(self, index_array):
        print("self.batch_index", self.batch_index,)
        print("==", index_array)
        batch_x = np.zeros(tuple([len(index_array)] + list(self.x.shape)[1:]),
                           dtype=K.floatx())
        if len(batch_x.shape)==3:
            batch_x = batch_x.reshape(batch_x.shape + (1,))
        for i, j in enumerate(index_array):
            x = self.x[j]
            if len(x.shape)==2:
                x = x.reshape(x.shape + (1,))
            x = self.image_data_generator.random_transform(x.astype(K.floatx()))
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=j,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        if self.color_mode in (3,'rgb'):
            if len(batch_x.shape)==3:
                batch_x = np.stack([ batch_x ]*3, axis=-1)
            if batch_x.shape[self.channels_axis]==1:
                batch_x = np.concatenate([ batch_x ]*3, axis=3)
            #print("batch_x", batch_x.shape)
            #raise Exception("test!!!")
        if self.postprocessing_function is not None:
            batch_x = self.postprocessing_function(batch_x)

        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


def _count_valid_files_in_directory(directory, white_list_formats, follow_links):
    """Count files with extension in `white_list_formats` contained in a directory.

    # Arguments
        directory: absolute path to the directory containing files to be counted
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.

    # Returns
        the count of files with extension in `white_list_formats` contained in
        the directory.
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links), key=lambda tpl: tpl[0])

    samples = 0
    for root, _, files in _recursive_list(directory):
        for fname in files:
            is_valid = False
            for extension in white_list_formats:
                if fname.lower().endswith('.' + extension):
                    is_valid = True
                    break
            if is_valid:
                samples += 1
    return samples


def _list_valid_filenames_in_directory(directory, white_list_formats,
                                       class_indices, follow_links):
    """List paths of files in `subdir` relative from `directory` whose extensions are in `white_list_formats`.

    # Arguments
        directory: absolute path to a directory containing the files to list.
            The directory name is used as class label and must be a key of `class_indices`.
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
        class_indices: dictionary mapping a class name to its index.

    # Returns
        classes: a list of class indices
        filenames: the path of valid files in `directory`, relative from
            `directory`'s parent (e.g., if `directory` is "dataset/class1",
            the filenames will be ["class1/file1.jpg", "class1/file2.jpg", ...]).
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links), key=lambda tpl: tpl[0])

    classes = []
    filenames = []
    subdir = os.path.basename(directory)
    basedir = os.path.dirname(directory)
    for root, _, files in _recursive_list(directory):
        for fname in sorted(files):
            is_valid = False
            for extension in white_list_formats:
                if fname.lower().endswith('.' + extension):
                    is_valid = True
                    break
            if is_valid:
                classes.append(class_indices[subdir])
                # add filename relative to directory
                absolute_path = os.path.join(root, fname)
                filenames.append(os.path.relpath(absolute_path, basedir))
    return classes, filenames


class DirectoryIterator(Iterator):
    """Iterator capable of reading images from a directory on disk.

    # Arguments
        directory: Path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        classes: Optional list of strings, names of subdirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 postprocessing_function=None,
                 follow_links=False,
                 stratify=None, oversampling=True,
                 subsample_factor= None,
                 subsample_num = None,
                 output_filenames=None,
                 ):

        self.output_filenames=output_filenames
        #self.postprocessing_function = postprocessing_function
        if data_format is None:
            data_format = K.image_data_format()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm'}

        # first, count the number of samples and classes
        self.samples = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.num_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        pool = multiprocessing.pool.ThreadPool()
        function_partial = partial(_count_valid_files_in_directory,
                                   white_list_formats=white_list_formats,
                                   follow_links=follow_links)
        self.samples = sum(pool.map(function_partial,
                                    (os.path.join(directory, subdir)
                                     for subdir in classes)))

        print('Found %d images belonging to %d classes.' % (self.samples, self.num_class))

        # second, build an index of the images in the different class subfolders
        results = []

        self.filenames = []
        self.classes = np.zeros((self.samples,), dtype='int32')
        i = 0
        for dirpath in (os.path.join(directory, subdir) for subdir in classes):
            results.append(pool.apply_async(_list_valid_filenames_in_directory,
                                            (dirpath, white_list_formats,
                                             self.class_indices, follow_links)))
        for res in results:
            classes, filenames = res.get()
            self.classes[i:i + len(classes)] = classes
            self.filenames += filenames
            i += len(classes)
        pool.close()
        pool.join()
        super(DirectoryIterator, self).__init__(self.samples, batch_size, shuffle, seed, 
                stratify=self.classes if stratify else None,
                oversampling=oversampling,
                subsample_factor=subsample_factor,
                subsample_num=subsample_num,
                postprocessing_function=postprocessing_function)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        batch_fn = []
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            batch_fn.append(fname)
            
            img = load_img(os.path.join(self.directory, fname),
                           grayscale=grayscale,
                           target_size=self.target_size)
            x = img_to_array(img, data_format=self.data_format)
            
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        if self.postprocessing_function:
                batch_x = self.postprocessing_function(batch_x)
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=j,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        if self.output_filenames:
            return batch_x, batch_y, batch_fn
        return batch_x, batch_y

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

# coding: utf-8
#from keras.utils.data_utils import Sequence
#import numpy as np
from numpy.lib.format import open_memmap
def get_slice(center, size=(256,256),
              target_size = (None, None),
              reflect = False,
              ):
    size = np.asarray(size)
    xy = np.asarray(center) - size//2
    end = xy + size
    
    margins_max = []
    margins_min = []
    slice_ = []
    for s_, e_, t_, sz in zip(xy, end, target_size, size):
        if t_:
            m_e = max(0, -(t_ - e_ ))
            e_ = min(e_, t_)
        else:
            m_e = None
        m_s = max(0, -s_)
        s_ = max(s_,0)
        if reflect and (m_s>0) or (m_e>0):
            #print("before", s_,e_, m_s, m_e)
            if (e_==0) and m_e > sz//2:
                m_e, s_ = t_-s_, t_ - m_e
            if (s_==0) and (m_e> e_):
                e_, m_s = m_s, e_
            #print("after", s_,e_, m_s, m_e)
        margins_max.append(m_e)
        margins_min.append(m_s)
        slice_.append(slice(s_, e_))
    return slice_, list(zip(margins_min, margins_max))


def pad_patch(img, slc, padseq):
    """if no padding required, returns a slice (pointer);
    otherwise returns a padded copy of a slice
    
    INPUT:
    - target_size  [ width, height ]
    """
    if any((s > 0 for s in padseq[0])) | any((s > 0 for s in padseq[1])):
        patch = np.pad(img[slc], padseq, mode='constant')
    else:
        patch = img[slc]
    return patch

class PatchIterator(Iterator):
    """
       point_sampler(filename, label=2)
    """
    def __init__(self, fn_img, fn_pnt,
                 point_sampler,
                 image_data_generator = None,
                 batch_size = 4,
                 shuffle=True,
                 seed=0,
                 patch_size = (512,512),
                 dtype='uint16',
                 mode = 'reflect',
                 label_freq={1:5, 2:10}, 
                 augmentation=None,
                 postprocessing_functions = [None, None],
                 color_mode = None,
                 output_indices = False,
                 patches_per_image = 1,
                 ):
        self.fn_img = fn_img
        self.fn_pnt = fn_pnt
        assert batch_size % patches_per_image == 0, (
            "batch_size must be multiple of patches_per_image")
        self.patches_per_image = patches_per_image
        self.imgs_per_batch = batch_size // self.patches_per_image
        assert len(fn_img)  == len(fn_pnt)
        assert len(fn_img) >0
        print("%d images supplied" % len(fn_img))
        self.point_sampler = point_sampler
        nsamples = len(self.fn_img)
        self.mode = mode
        self.postprocessing_functions = postprocessing_functions
        self.color_mode=color_mode
        self._reflect = self.mode == 'reflect'
        self.augmentation = augmentation
        self.image_data_generator = image_data_generator
        self.output_indices = output_indices
        self.transforms = []
        if self.image_data_generator is not None:
            self.transforms.append( self.image_data_generator.random_transform )
            self.transforms.append( self.image_data_generator.standardize )
        
        _norm_const = sum(label_freq.values())
        #print(_norm_const)
        self.patch_size = tuple(patch_size)
        self.dtype = dtype
        self.label_freq_dict = {kk:vv/_norm_const for kk,vv in label_freq.items()}
        self.labels = np.asarray(list(self.label_freq_dict.keys()))
        self.label_freq = list(self.label_freq_dict.values())
        self.label_cum_freq = np.cumsum(self.label_freq)

        self.index_generator = self._flow_index()
        super(PatchIterator, self).__init__(nsamples, batch_size, shuffle, seed,
                                            batch_rate = patches_per_image,
#                                                 stratify=stratify, oversampling=oversampling,
#                                                 postprocessing_function=postprocessing_function
                                                )
    def sample_label(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        matr = np.random.rand(batch_size, self.label_cum_freq.shape[0])>= self.label_cum_freq
        label_inds = np.argmin(matr, axis=1)
        return np.asarray([self.labels[x] for x in label_inds])
        
    def init_indices():
        self.indices = np.random.randint(len(self.fn_img))
        
    def open_npy(self):
        for ff in self.fn_img:
            img = open_memmap(ff,  dtype=np.uint16, mode='r', shape=target_size[::-1])
            
    def sample_points(self, index, labels):
        fm = self.fn_pnt[index]
        try:
            samplegen = self.point_sampler(fm, labels)
            for pt, label in zip(samplegen, labels):
                yield (label, pt) 
            #pt = mzl.point(fm, buf=None, label=2, position=-1)
        except OSError as ee:
            print("error on ", fm)
            raise ee

    def sample_img(self, img, pt, buffer=None, extend_dim=False, transforms=[]):
        shape = img.shape
        slc, padseq = get_slice(pt, size=self.patch_size, target_size=shape, reflect= self._reflect)
        if buffer is None:
            patch = pad_patch(img, slc, padseq)
            return patch
        else:
            if self.mode == 'constant':
                outslc = [slice(ss.start-ss.start, ss.stop-ss.start) for ss in slc]
                if extend_dim:
                    outslc += [0]
                buffer[outslc] = img[slc]
            else:
                outslc = [slice(None)]*len(slc)
                if extend_dim:
                    outslc += [0]
                buffer[outslc] = np.pad(img[slc], padseq, self.mode)#[:,:]
            if self.augmentation and len(transforms)>0:
                for tt in transforms:
                    if buffer is None:
                        print(fi, pt, tt)
                    buffer[:] = tt(buffer.copy())[:]
                #print("finally after rescale", buffer.max())
    
    def _get_batches_of_transformed_samples(self, sample_inds):
        if sample_inds is None:
            with self.lock:
                sample_inds = next(self.index_generator)
        # Repeat each image index
        # sample_inds = list(itertools.chain.from_iterable(itertools.repeat(x, self.imgs_per_batch) for x in sample_inds))
        #batch_class_inds = self.sample_label()
        curr_batch_size = len(sample_inds) * self.patches_per_image
        pts = np.zeros((curr_batch_size, 2), dtype='uint16') 
        slices = [None, slice(None), slice(None)]
        if self.color_mode is None:
            extend_dim = False
            buffer = np.zeros((curr_batch_size, ) + self.patch_size, dtype= self.dtype)
        elif self.color_mode in ('grayscale', 'greyscale', 1):
            extend_dim = True 
            buffer = np.zeros((curr_batch_size, ) + self.patch_size + (1,), dtype= self.dtype)
        else:
            raise ValueError("`color_mode` should be None or 'grayscale'")

        batch_class_inds = np.zeros(curr_batch_size, dtype='uint16')
        for nn, (ss) in enumerate(sample_inds):
            labels = self.sample_label(self.patches_per_image)
            lbl_slice = slice(self.patches_per_image*nn, self.patches_per_image*(nn+1))
            batch_class_inds[lbl_slice] = labels
            points = self.sample_points(ss, labels)
            img = open_memmap(self.fn_img[ss],  mode='r',)
            for jj, (lbl, pt) in enumerate(points):
                #range(self.patches_per_image):
                #print("nn", nn, "jj", jj, "pt",  pt)
                pts[jj, :] = pt
                slices[0] = nn * self.patches_per_image + jj
                #print("buffer size", buffer.shape)
                #print("slice", slices[0])
                self.sample_img(img, pt, buffer[slices],
                                transforms=self.transforms,
                                extend_dim=extend_dim)
            
        output = [buffer, batch_class_inds]
        if self.output_indices:
            output.append(sample_inds)
            output.append(pts)

        for ii, (dd, ff) in enumerate(zip(output, self.postprocessing_functions)):
            # print(ii, ff)
            if ff is not None:
                output[ii] = ff(dd)
        return output #, pts

#         pool = multiprocessing.pool.ThreadPool()
    
    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)#import threading

class MemMapDataset():
    def __init__(self, root_dir, csv_file, 
                 classes = ["Control", "Case",],
                 label_col = "label",
                 filename_col = "filename",
                 binary=True,
                 transform=None,
                 nsamples = None,
                 encode_label = None):
        import pandas as pd
        self.transform = transform
        self.csv_file = csv_file
        self.table = pd.read_csv(csv_file)
        if nsamples:
            self.table = self.table[:nsamples]
        self.filenames = self.table[filename_col].tolist()
        self.classes = self.table[label_col].tolist()
        self.root_dir = root_dir
        self.label_col = label_col
        self.filename_col = filename_col
        for cc in classes:
            if cc not in self.classes:
                print("class is missing in data: %s" % cc)
        #classes = np.unique(self.classes).tolist()
        self.class_set = classes
        if encode_label is None:
            if len(classes) in (1,2) and binary:
                self.onehot = MemMapDataset.encode_label_binary(self.table[label_col], self.class_set)
            else:
                self.onehot = MemMapDataset.encode_label_onehot(self.table[label_col], self.class_set)
        else:
            self.onehot = encode_label(self.table[label_col], self.class_set)

    @staticmethod
    def encode_label_binary(labelvector, class_set):
        onehot = np.stack([(labelvector == cc).values for cc in class_set[1:]], axis=-1)
        return onehot
        
    @staticmethod
    def encode_label_onehot(labelvector, class_set):
        onehot = np.stack([(labelvector == cc).values for cc in class_set], axis=-1)
        return onehot

    def __len__(self):
        return len(self.table)
    
    def __getitem__(self, idx):
        item = self.table.iloc[idx]
        img_name = os.path.join(self.root_dir, item[self.filename_col])
        
        image = open_memmap(img_name, mode='r')
        label = self.onehot[idx]

        if self.transform:
            sample = self.transform([image, label])
        else:
            sample = [image, label]

        return sample

class MemMapIterator(Iterator):
    def __init__(self, root_dir, csv_file,
                 classes = ["Control", "Case",],
                 image_data_generator = None,
                 binary=True,
                 transform=None,
                 nsamples = None,
                 batch_size = 1,
                 shuffle = False,
                 seed = None,
                 postprocessing_function=None,
                 stratify=None,
                 oversampling=True,
                 subsample_factor=None,
                 subsample_num=None,
                 batch_rate=1,
                 dtype = K.floatx(),
                 color_mode=None,
                 data_format = 'channels_last',
                 label_col = "label",
                 filename_col = "filename",
                 encode_label=None,
                 ):
        channels_axis = 3 if data_format == 'channels_last' else 1
        self.channels_axis = channels_axis
        self.dtype = dtype
        self.color_mode=color_mode
        #self.image_data_generator = image_data_generator
        self.transforms = []
        if image_data_generator is not None:
            self.transforms.append( image_data_generator.random_transform )
            self.transforms.append( image_data_generator.standardize )

        self.dataset = MemMapDataset(root_dir, csv_file, classes=classes, binary=binary,
                                     transform=transform, nsamples=nsamples,
                                     label_col = label_col,
                                     filename_col = filename_col,
                                     encode_label=encode_label,
                                     )
        self.classes = self.dataset.classes
        self.filenames = self.dataset.filenames

        super(MemMapIterator, self).__init__(len(self.dataset), batch_size, shuffle, seed,
                                             stratify=self.classes if stratify else None,
                                             oversampling=oversampling,
                                             subsample_factor=subsample_factor,
                                             subsample_num=subsample_num,
                                             postprocessing_function=postprocessing_function,
                                                )

    def _get_batches_of_transformed_samples(self, index_array):
        print("index_array", index_array)
        batch_x = np.zeros(tuple([len(index_array)] + list(self.dataset[0][0].shape)),
                           dtype=self.dtype)
        if len(batch_x.shape)==3:
            batch_x = batch_x.reshape(batch_x.shape + (1,))
        for i, j in enumerate(index_array):
            x = self.dataset[j][0]
            if len(x.shape)==2:
                x = x.reshape(x.shape + (1,))
            x = x.astype(self.dtype)
            for tt in self.transforms:
                x = tt(x)
            batch_x[i] = x
        if self.color_mode in (3,'rgb'):
            if len(batch_x.shape)==3:
                batch_x = np.stack([ batch_x ]*3, axis=-1)
            if batch_x.shape[self.channels_axis]==1:
                batch_x = np.concatenate([ batch_x ]*3, axis=3)
            #print("batch_x", batch_x.shape)
            #raise Exception("test!!!")
        if self.postprocessing_function is not None:
            batch_x = self.postprocessing_function(batch_x)
        if len(self.dataset[0])==1:
            return batch_x
        batch_y = np.asarray([self.dataset[j][1] for j in index_array])
        return batch_x, batch_y

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


def read_decode_coco(fname):
    with open(fname) as fh:
        coco = json.load(fh)
    return decode(coco)

class MemMapCocoDataset():
    def __init__(self, root_dir, csv_file, 
                 binary=True,
                 transform=None,
                 nsamples = None,
                 mmapcol = 'memmap',
                 cococol = 'coco',
                 ):
        import pandas as pd
        self.transform = transform
        self.table = pd.read_csv(csv_file)
        if nsamples:
            self.table = self.table[:nsamples]
        self.root_dir = root_dir
        self.mmapcol = mmapcol
        self.cococol = cococol

    def __len__(self):
        return len(self.table)
    
    def __getitem__(self, idx):
        item = self.table.iloc[idx]
        
        img_name = os.path.join(self.root_dir,
                                item[self.mmapcol])
        image = open_memmap(img_name, mode='r')

        coco_name = os.path.join(self.root_dir,
                                item[self.cococol])
        label = read_decode_coco(coco_name)

        if self.transform:
            sample = self.transform([image, label])
        else:
            sample = [image, label]

        return sample


def resize_inputs(xx, yy,
                mode='constant',
                target_size = [512, 512],
                constant_values_x = 255,
                constant_values_y = 0,
                add_const_to_label = 0,
              ):
    xx = (crop_pad_center(xx, target_size, pad_mode=mode, constant_values=constant_values_x))
    if add_const_to_label>0:
        yy += add_const_to_label
    yy = (crop_pad_center(yy, target_size, pad_mode=mode, constant_values=constant_values_y))
    if len(yy.shape) == 2:
        yy= yy[..., np.newaxis]
    return xx, yy

class MemMapCocoIterator(Iterator):
    def __init__(self, root_dir, csv_file,
                 image_data_generator = None,
                 binary=True,
                 mode='constant',
                 target_size = [512, 512],
                 constant_values_x = 255,
                 constant_values_y = 0,
                 nsamples = None,
                 batch_size = 1,
                 shuffle = False,
                 seed = None,
                 postprocessing_function=None,
                 stratify=None,
                 oversampling=True,
                 subsample_factor=None,
                 subsample_num=None,
                 batch_rate=1,
                 dtype = K.floatx(),
                 color_mode=None,
                 data_format = 'channels_last',
                 output_indices = False,
                 add_const_to_label=0,
                 ):
        self.output_indices = output_indices
        channels_axis = 3 if data_format == 'channels_last' else 1
        self.channels_axis = channels_axis
        self.dtype = dtype
        self.color_mode=color_mode
        #self.image_data_generator = image_data_generator
        #self.transforms = []
        if image_data_generator is not None:
            self.get_random_transform_specs = image_data_generator.get_random_transform_specs
            self.apply_geom_transform = image_data_generator.apply_transform
            self.intensity_transform = image_data_generator.standardize
            #self.transforms.append( image_data_generator.random_transform )
            #self.transforms.append( image_data_generator.standardize )
        if target_size is not None:
            transform = lambda x : resize_inputs(x[0], x[1],
                                                 mode=mode,
                                                 target_size = target_size,
                                                 constant_values_x = constant_values_x,
                                                 constant_values_y = constant_values_y,
                                                 add_const_to_label=add_const_to_label,
                                                 )
        self.constant_values_x = constant_values_x 
        self.constant_values_y = constant_values_y
        
        self.dataset = MemMapCocoDataset(root_dir, csv_file, binary=binary,
                                         transform=transform, nsamples=nsamples)

        super(MemMapCocoIterator, self).__init__(len(self.dataset), batch_size, shuffle, seed,
                                                 stratify=stratify, oversampling=oversampling,
                                                 subsample_factor=subsample_factor,
                                                 subsample_num=subsample_num,
                                                 postprocessing_function=postprocessing_function
                                                )

    def _get_batches_of_transformed_samples(self, index_array):
        #import ipdb
        #ipdb.set_trace()
        batch_x = np.zeros(tuple([len(index_array)] + list(self.dataset[0][0].shape)),
                           dtype=self.dtype)
        batch_y = np.zeros(tuple([len(index_array)] + list(self.dataset[0][1].shape)),
                           dtype=self.dtype)
        if len(batch_x.shape)==3:
            batch_x = batch_x.reshape(batch_x.shape + (1,))
        for i, j in enumerate(index_array):
            x, y = self.dataset[j]
            if len(x.shape)==2:
                x = x.reshape(x.shape + (1,))
            
            if hasattr(self, 'get_random_transform_specs'):
                transform_matrix, horizontal_flip, vertical_flip = self.get_random_transform_specs(x)
                x = self.apply_geom_transform(x, transform_matrix, horizontal_flip, vertical_flip,
                                        interp=cv2.INTER_CUBIC,
                                        #interp=cv2.INTER_NEAREST,
                                        borderMode = cv2.BORDER_CONSTANT,
                                        cval = [self.constant_values_x] *3,
                                        use_opencv=True)
                y = self.apply_geom_transform(y, transform_matrix, horizontal_flip, vertical_flip,
                                              interp=cv2.INTER_NEAREST,
                                              borderMode = cv2.BORDER_CONSTANT,
                                              cval=self.constant_values_y,
                                              use_opencv=True)
            if hasattr(self, 'intensity_transform'):
                x = self.intensity_transform(x)
            x = x.astype(self.dtype)
            batch_x[i] = x
            batch_y[i] = y
        if self.color_mode in (3,'rgb'):
            if len(batch_x.shape)==3:
                batch_x = np.stack([ batch_x ]*3, axis=-1)
            if batch_x.shape[self.channels_axis]==1:
                batch_x = np.concatenate([ batch_x ]*3, axis=3)
            #print("batch_x", batch_x.shape)
            #raise Exception("test!!!")
        if self.postprocessing_function is not None:
            batch_x = self.postprocessing_function(batch_x)
        if self.output_indices:
            if len(self.dataset[0])==1:
                return batch_x, index_array
            return batch_x, batch_y, index_array
        else:
            if len(self.dataset[0])==1:
                return batch_x
            return batch_x, batch_y

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)
