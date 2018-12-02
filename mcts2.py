import numpy as np
import time


class MCTS:
    def __init__(self, board, network):
        # state = root state
        p, v = network(board, board.valid_moves())
        self.network = network
        self.root = MCTSNode(board, network)
        self.board = board

    def search(self, node, board=None):
        if board is not None:
            self.board = board
        # check if game is finished
        if self.board.over():
            return self.board.winner()

        if node.leaf:
            p, v = self.network(node, node.valid_moves())
            node.expand()
            return v

        a = node.select_move()
        self.board.move(a)
        v = self.search(node.children[a])
        node.update(v)
        return v


class MCTSNode:
    def __init__(self, board, network, c=4):
        self.s = str(board)
        self._board = board
        self.c = c
        self.actions = np.nonzero(board.valid_moves())[0]
        self.network = network
        self.v_mult = board.turn_player()
        probabilities, _ = network(self._board.board(), self.valid_moves())
        self.P = {}
        self.N = {}
        self.Q = {}
        self.W = {}
        for a, p in zip(self.actions, probabilities[0]):
            self.P[a] = p
            self.N[a] = 0
            self.Q[a] = 0
            self.W[a] = 0
        self._ucb = {}
        self._update_ucb()
        self.children = {}
        self.leaf = True

    def __str__(self):
        return self.s

    def _update_ucb(self):
        ss = sum(self.P.values())
        for a in self.actions:
            self._ucb[a] = self.c * self.P[a] * (np.sqrt(ss)) / (1 + self.N[a]) + self.Q[a]

    def expand(self):
        if self.leaf:
            for a in self.actions:
                bb = self._board.clone()
                bb.move(a)
                self.children[a] = MCTSNode(bb, self.network)
        self.leaf = False

    def select_move(self):
        self.last_move = max(self._ucb, key=self._ucb.get)
        return self.last_move

    def update(self, v):
        self.N[self.last_move] += 1
        self.W[self.last_move] += v * self.v_mult
        self.Q[self.last_move] = self.W[self.last_move] / self.N[self.last_move]
        self._update_ucb()

    def board(self):
        return self._board.board()

    def valid_moves(self):
        return self._board.valid_moves()
