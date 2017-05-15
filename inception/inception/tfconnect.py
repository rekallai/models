"""
A module to create placeholders and feeds for complex data structures
"""

import tensorflow as tf
import numpy as np

__EXPECTED_OUTPUT_TYPES__ = (np.ndarray, float, int, str, bool)


def create_placeholder_for(operation, name, counter=-1):
    op = operation

    t = type(op)
    placeholder = None
    if t is tf.Tensor:
        counter += 1
        placeholder = tf.placeholder(op.dtype, op.shape, '%s_%d' % (name, counter))
    elif t is list:
        placeholder = []
        for el in op:
            p, counter = create_placeholder_for(el, name, counter)
            placeholder.append(p)
    elif t is tuple:
        placeholder = ()
        for el in op:
            p, counter = create_placeholder_for(el, name, counter)
            placeholder += (p, )
    elif t is dict:
        placeholder = dict()
        for k, v in op.items():
            p, counter = create_placeholder_for(v, name, counter)
            placeholder[k] = p
    else:
        print("ERROR : Unknown or unsupported type in the operation data structure : " + t)

    return placeholder, counter


def create_feed_based_on(operation_output, name, feed=None, counter=-1):
    o = operation_output

    t = type(o)

    if feed is None:
        feed = dict()

    if __is_value(o):
        counter += 1
        feed['%s_%d' % (name, counter)] = o
    elif t is list:
        for el in o:
            feed, counter = create_feed_based_on(el, name, feed, counter)
    elif t is tuple:
        for el in o:
            feed, counter = create_feed_based_on(el, name, feed, counter)
    elif t is dict:
        for k, v in o.items():
            feed, counter = create_feed_based_on(v, name, feed, counter)
    else:
        print("ERROR : Unknown or unsupported type in the operation data structure : " + t)

    return feed, counter


def __is_value(v):
    global __EXPECTED_OUTPUT_TYPES__

    return type(v) in __EXPECTED_OUTPUT_TYPES__

