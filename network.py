import tensorflow as tf
import numpy as np
from utils import *


args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'num_channels': 512,
    'c': 0.5})


class NeuralNet(object):
    def __init__(self, board):
        self.input = tf.placeholder(shape=(None, len(board.board())))
        self.action_size = np.array(board.moves()).shape()
        self._build_model()
        self._build_loss()

    def _build_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            h1 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.layers.conv1d(self.input, args.num_channels, 3, padding='same')))
            h2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.layers.conv1d(h1, args.num_channels, 3, padding='same')))
            h3 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.layers.conv1d(h2, args.num_channels, 3, padding='same')))
            h4 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.layers.conv1d(h3, args.num_channels, 2, padding='same')))
            h = tf.reshape(h4, [-1])

            h = tf.layers.dropout(tf.nn.relu(tf.contrib.layers.batch_norm(tf.contrib.layers.fully_connected(h, 1024))), args.dropout)
            h = tf.layers.dropout(tf.nn.relu(tf.contrib.layers.batch_norm(tf.contrib.layers.fully_connected(h, 512))), args.dropout)

            self.pi = tf.contrib.layers.fully_connected(h, self.action_size)
            self.prob = tf.nn.softmax(self.pi)
            self.v = tf.tanh(tf.contrib.layers.fully_connected(h, 1))

    def _build_loss(self):
        self.target_pis = tf.placeholder(tf.float32, shape=[None, self.action_size])
        self.target_vs = tf.placeholder(tf.float32, shape=None)
        self.loss_pi = tf.losses.softmax_cross_entropy(self.target_pis, self.pi)
        self.loss_v = tf.losses.mean_squared_error(self.target_vs, tf.reshape(self.v, shape=[-1, ]))
        self.loss = self.loss_pi + self.loss_v
        self.optimize = tf.train.AdamOptimizer(args.lr).minimize(self.loss)
