import numpy as np
import tensorflow as tf

norm_decay = input('Enter momentum: ')
norm_epsilon = input('Enter epsilon: ')

def batch_norm_layers(input,format,learning):
    layers = tf.layers.batch_normalization(inputs = input,momentum = norm_decay,epsilon = norm_epsilon,training = learning, axis = 1 if format == 'channels_first' else 3)
    return layers

def padded_input(input,kernelsize,format):
    pad_s = (kernelsize - 1)//2
    pad_e = kernelsize - 1 - pad_s
    if format == 'channels_first':
        return tf.pad(input,[[0,0],[0,0],[pad_s,pad_e],[pad_s,pad_e]])
    else:
        return tf.pad(input,[[0,0],[pad_s,pad_e],[pad_s,pad_e],[0,0]])

def conv_layer(input,stride,filter,kernel_size,format):
    if stride >1:
        input = padded_input(input,kernel_size,format)
        padding = 'VALID'
    else:
        padding = 'SAME'
    conv_output = tf.layers.conv2D(inputs = input,filters= filter,kernel_size = kernel_size,padding = padding,strides = stride,data_format = format)
