import numpy as np

class MCTS:
    def __init__(self, board, network):
        # state = root state
        p, v = network(board)
        self.network = network
        root = MCTSNode(board, network)
        self.board = board

    def search(self, node):
        # check if game is finished
        if self.board.over():
            return self.board.winner()

        if node.leaf:
            p, v = self.network(node)
            node.expand()
            return v
        # TODO: select move, propagate the value
        a = node.select_move()
        self.board.move(a) # TODO: should this be turning the board?
        node.update(v)


class MCTSNode:
    def __init__(self, board, network, mult=1):
        self.s = str(board)
        self.actions = np.nonzero(board.valid_moves())
        self.network = network
        self.v_mult = mult
        probabilities, _ = network(self.s)
        self.P = {}
        self.N = {}
        self.Q = {}
        self.W = {}
        for a, p in zip(self.actions, probabilities):
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
        for a in self.actions:
            self._ucb[a] = self.P[a] / (1 + self.N[a]) + self.Q[a]

    def expand(self):
        if self.leaf:
            for a in self.actions:
                bb, v_mult = self.board.clone_turn(a)
                self.children[a] = MCTSNode(bb, self.network, self.v_mult * v_mult)
        self.leaf = False

    def select_move(self):
        self.last_move = max(self._ucb, key=self._ucb.get)
        return self.last_move

    def update(self, v):
        self.N[self.last_move] += 1
        self.W[self.last_move] += v * self.v_mult
        self.Q[self.last_move] = self.W[self.last_move] / self.N[self.last_move]
