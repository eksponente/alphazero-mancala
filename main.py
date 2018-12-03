from comet_ml import Experiment
import argparse
from mcts import MCTS
from network import NeuralNet, ReplayBuffer
from game import Game
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import time


parser = argparse.ArgumentParser()
parser.add_argument('--learning-rate', type=float, default=0.001)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--num-channels', type=int, default=512)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch-size', type=int, default=2048)
parser.add_argument('--replay-buffer-size', type=int, default=500000)
parser.add_argument('--mcts-rollouts', type=int, default=150)
parser.add_argument('--temp-decrese-moves', type=int, default=1)
parser.add_argument('--n-episodes-per-iteration', type=int, default=3)
parser.add_argument('--eval', action='store_true')
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES


def train_alphazero(lr, dropout, num_channels, epochs, batch_size, replay_buffer_size,
                    temp_decrese_moves, mcts_rollouts, n_episodes_per_iteration, eval):
    board = Game(player_turn=1)
    network = NeuralNet(board, num_channels, lr, dropout, epochs,
                        batch_size)
    # if evaluating, just run the evaluator
    if eval:
        network.load()
        evaluate_network(network, board)
        
    # set up the experiment
    experiment = Experiment(api_key=os.environ.get('ALPHAZERO_API_KEY'),
                            project_name=os.environ.get('ALPHAZERO_PROJECT_NAME'),
                            workspace=os.environ.get('ALPHAZERO_WORKSPACE'))
    experiment.log_multiple_params({
        'lr': lr,
        'dropout': dropout,
        'num_channels': num_channels,
        'epochs': epochs,
        'batch_size': batch_size,
        'replay_buffer_size': replay_buffer_size,
        'temp_decrese_moves': temp_decrese_moves,
        'mcts_rollouts': mcts_rollouts,
        'n_episodes_per_iteration': n_episodes_per_iteration
    })
    buf = ReplayBuffer(replay_buffer_size, batch_size)

    epoch = 0
    while True:
        epoch += 1
        print("Epoch {}, {}".format(epoch, time.clock()))
        for i in range(n_episodes_per_iteration):
            winner = execute_episode(network, buf, experiment)
            print("Finished episode {}, winner {}, time {}".format(i, winner, time.clock()))
        for _ in range(epochs):
            loss, entropy = train_network(network, buf, experiment)
        print("Training loss: {}, entropy: {}".format(loss, entropy))
        network.save()


def execute_episode(network, replay_buffer, experiment):
    examples = []
    board = Game(player_turn=1)
    mcts = MCTS(board.clone(), network)
    temp = 1.0
    i = 0
    while not board.over():
        i += 1
        if i >= experiment.get_parameter('temp_decrese_moves'):
            t = 10e-3
        # perform mcts search
        for i in range(experiment.get_parameter('mcts_rollouts')):
            mcts.search(mcts.root, board.clone())

        # choose the action
        N_total = np.sum(np.array(list(mcts.root.N.values())) ** (1 / temp))
        pi = np.zeros(6)
        for a in mcts.root.actions:
            pi[a] = mcts.root.N[a] ** (1 / temp) / N_total
        action = np.random.choice(np.arange(6), p=pi)
        # add the move to the replay buffer
        replay_buffer.add(board.board(), action, pi, mcts.root.v_mult, board.valid_moves())
        board.move(action)
        if board.over():
            replay_buffer.finish_episode(board.winner())
            return board.winner()
        mcts.root = mcts.root.children[action]


def train_network(network, replay_buffer, experiment):
    pis, vs, boards, valid_moves = replay_buffer.sample()
    loss, _, entropy = network.train(pis, vs, boards, valid_moves)
    experiment.log_metric("loss", loss)
    experiment.log_metric("entropy", entropy)
    return loss, entropy


def evaluate_network(network, board):
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    args = parser.parse_args()
    train_alphazero(args.learning_rate, args.dropout, args.num_channels, args.epochs, args.batch_size, args.replay_buffer_size, args.temp_decrese_moves, args.mcts_rollouts, args.n_episodes_per_iteration, args.eval)
