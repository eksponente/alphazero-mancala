import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Dropout, Multiply, Softmax
from keras import backend as K


class NeuralNet(object):
    def __init__(self, board, num_channels, lr, dropout, epochs, batch_size, action_size=6):
        self.board_size = len(board.board())
        self.action_size = action_size
        self.num_channels = num_channels
        self.lr = lr
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        tf.reset_default_graph()
        self.sess = tf.Session()
        K.set_session(self.sess)
        with tf.name_scope("model") as scope:
            self._build_model_keras()
        self.init()

    def __call__(self, board, valid_moves):
        try:
            hh = np.array(board.board())
        except AttributeError:
            hh = np.array(board)
        if len(hh.shape) < 3:
            hh = np.array([hh])
            hh = np.expand_dims(hh, -1)
        valid_moves = np.array([valid_moves])
        return self.sess.run([self.prob, self.v], feed_dict={self.input: hh,
                                                             self.valid_moves_tensor: valid_moves})

    def init(self):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

    def save(self):
        name = 'model.h5'
        print("Saving model {}".format(name))
        self.model.save(name)

    def _build_model_keras(self):
        self.input_tensor = tf.placeholder(tf.float32, shape=(None, self.board_size, 1))
        self.valid_moves_tensor = tf.placeholder(tf.float32, shape=(None, self.action_size))
        self.input = Input(tensor=self.input_tensor)
        self.valid_moves_mask = Input(tensor=self.valid_moves_tensor)
        x = self.input
        x = Flatten()(x)

        x = Dropout(self.dropout)(Dense(2048, activation=tf.nn.relu)(x))
        x = Dropout(self.dropout)(Dense(1024, activation=tf.nn.relu)(x))
        x = Dropout(self.dropout)(Dense(512, activation=tf.nn.relu)(x))
        self.pi = Dense(self.action_size)(x)
        self.prob = Softmax()(Multiply()([self.pi, self.valid_moves_mask]))
        self.v = Dense(1)(x)
        self.model = Model(inputs=[self.input, self.valid_moves_mask], outputs=[self.prob, self.v])

        self.target_pis = tf.placeholder(tf.float32, shape=[None, self.action_size])
        self.target_vs = tf.placeholder(tf.float32, shape=None)
        self.loss_pi = tf.losses.softmax_cross_entropy(self.target_pis, self.prob)
        self.loss_v = tf.losses.mean_squared_error(self.target_vs, tf.reshape(self.v, shape=[-1, ]))
        self.loss = self.loss_pi + self.loss_v
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.optimize = self.optimizer.minimize(self.loss)


    def loss(self, pis, vs, boards, valid_moves):
        # TODO: add L2
        return self.sess.run([self.loss, self.optimize],
                             feed_dict={self.target_pis: pis,
                                        self.target_vs: vs,
                                        self.input: boards,
                                        self.valid_moves_tensor: valid_moves})


class ReplayBuffer:
    def __init__(self, maxlen, batch_size, action_size=6, board_size=14):
        self.episode = 0
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.pis = np.array([]).reshape((0, action_size))
        self.vs = np.array([]).reshape((0, 1))
        self.boards = np.array([]).reshape((0, board_size))
        self.valid_moves = np.array([]).reshape((0, action_size))

    def _drop_items(self):
        if self.pis.shape[0] >= self.maxlen:
            self.pis = self.pis[1:]
            self.vs = self.vs[1:]
            self.boards = self.boards[1:]
            self.valid_moves = self.valid_moves[1:]

    def add(self, board, action, pi, z_mult, valid_moves):
        self.pis = np.vstack([self.pis, pi])
        self.boards = np.vstack([self.boards, board])
        self.vs = np.vstack([self.vs, z_mult])
        self.valid_moves = np.vstack([self.valid_moves, valid_moves])
        self._drop_items()
        self.episode += 1

    def finish_episode(self, result):
        while self.episode > 0:
            self.vs[-self.episode] *= result
            self.episode -= 1

    def sample(self):
        idxs = np.random.choice(np.arange(self.pis.shape[0]), self.batch_size)
        return self.pis[idxs], self.vs[idxs], self.boards[idxs], self.valid_moves[idxs]
