#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
get_ipython().run_line_magic('matplotlib', 'inline')

import math
import random
import time
import os

import tensorflow as tf


# In[16]:


class Model():
    
    def __init__(self, args):
        self.args = args
        self.graves_initializer = tf.truncated_normal_initializer(mean=0., stddev=.075, seed=None, dtype=tf.float32)
        self.window_b_initializer = tf.truncated_normal_initializer(mean=-3.0, stddev=.25, seed=None, dtype=tf.float32)
        self.learning_rate = 0.001
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        
        # Build an LSTM cell, each cell has rnn_size number of units
        cell_func = tf.contrib.rnn.LSTMCell
        print(self.args)
        with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
            self.cell0 = cell_func(self.args['rnn_size'], state_is_tuple=True, initializer=self.graves_initializer)
            self.cell1 = cell_func(self.args['rnn_size'], state_is_tuple=True, initializer=self.graves_initializer)
            self.cell2 = cell_func(self.args['rnn_size'], state_is_tuple=True, initializer=self.graves_initializer)
        
            # Placeholders for input and output data, each entry has tsteps points at a time
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, args['tsteps'], 3])
            self.output = tf.placeholder(dtype=tf.float32, shape=[None, args['tsteps'], 3])
        
            # Setting the states of memory cells in each LSTM cell.
            # batch_size is the number of training examples in a batch. Each training example is a set of tsteps number of
            # (x,y, <end_of_stroke>) tuples, i.e. a sequence of strokes till t time steps.
            self.istate_cell0 = self.cell0.zero_state(batch_size=args['batch_size'], dtype=tf.float32)
            self.istate_cell1 = self.cell1.zero_state(batch_size=args['batch_size'], dtype=tf.float32)
            self.istate_cell2 = self.cell2.zero_state(batch_size=args['batch_size'], dtype=tf.float32)


    def init_args():
    #     args.train = False
        args.rnn_size = 100 
        args.tsteps = 256 if args.train else 1
        args.batch_size = 32 if args.train else 1
        args.nmixtures = 8 # number of Gaussian mixtures in MDN
        args.alphabet = ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        args.tsteps_per_ascii = 25
        return args
    
    def build_computational_graph(self, inputs, cell, initial_cell_state, scope):
        # TODO():update scope
        output, cell_final_state = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, initial_cell_state, cell, loop_function=None, scope=scope)
        return output

    


# In[ ]:





# In[7]:





# In[17]:


m = Model({'rnn_size':400, 'tsteps': 4, 'batch_size' : 5})


# In[ ]:




