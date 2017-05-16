"""
A module to create placeholders and feeds for complex data structures
"""

import tensorflow as tf
import numpy as np

__EXPECTED_OUTPUT_TYPES__ = (np.ndarray, float, int, str, bool)


def create_placeholder_for(operation, name, counter=-1, is_variable=None):
    op = operation

    current_op_is_var = False

    if is_variable is None:
        is_variable = dict()

    t = type(op)
    placeholder = None
    if t is tf.Tensor:
        counter += 1
        placeholder = tf.placeholder(op.dtype, op.shape, '%s_%d' % (name, counter))
    elif t is tf.Variable:
        counter += 1
        placeholder=op
        is_variable[counter] = True
    elif t is list:
        placeholder = []
        for el in op:
            p, counter, is_variable = create_placeholder_for(el, name, counter, is_variable)
            placeholder.append(p)
    elif t is tuple:
        placeholder = ()
        for el in op:
            p, counter, is_variable = create_placeholder_for(el, name, counter, is_variable)
            placeholder += (p, )
    elif t is dict:
        placeholder = dict()
        for k, v in op.items():
            p, counter, is_variable = create_placeholder_for(v, name, counter, is_variable)
            placeholder[k] = p
    else:
        print("ERROR : create_placeholder_for : Unknown or unsupported type in the operation data structure : %s" % str(t))

    return placeholder, counter, is_variable


def create_feed_based_on(operation_output, name, is_variable=None, feed=None, counter=-1):
    o = operation_output

    t = type(o)

    if feed is None:
        feed = dict()

    if is_variable is None:
        is_variable = dict()

    if __is_value(o):
        counter += 1
        if not (counter in is_variable):
            feed['%s_%d:0' % (name, counter)] = o
    elif t is list:
        for el in o:
            feed, counter = create_feed_based_on(el, name, is_variable, feed, counter)
    elif t is tuple:
        for el in o:
            feed, counter = create_feed_based_on(el, name, is_variable, feed, counter)
    elif t is dict:
        for k, v in o.items():
            feed, counter = create_feed_based_on(v, name, is_variable, feed, counter)
    else:
        print("ERROR : create_feed_based_on : Unknown or unsupported type in the operation output data structure : %s" % str(t))

    return feed, counter


def __is_value(v):
    global __EXPECTED_OUTPUT_TYPES__

    return type(v) in __EXPECTED_OUTPUT_TYPES__

