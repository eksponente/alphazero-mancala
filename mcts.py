import math


class MCTS():
    def __init__(self, board, network, c):
        self.board = board
        self.network = network
        self.Qs = {}
        self.Ps = {}
        self.Ns = {}

        self.Qsa = {}
        self.Psa = {}
        self.Nsa = {}
        self.n = 0
        self.c = c

    def ucb(self, state, action):
        u = self.Qsa[state][action] + self.c * self.Psa[state][action] * math.sqrt(sum(self.Nsa[state])) / (1 + self.Nsa[state][action])
        return u

    def search(self, board, same_player):
        # if this is the final position
        if board.score() != 0:
            return board.score()[0] * same_player

        s = str(board)

        # expansion: the state has never been visited before
        if s not in self.Psa:
            pi, v = self.network(board)
            mask = board.valid_moves()
            self.Psa[s] = [p * m for p, m in zip(pi, mask)]
            self.Psa[s] /= sum(self.Psa[s])
            self.Nsa[s] = [0] * len(mask)
            self.Qsa[s] = [0] * len(mask)
            return v * same_player

        # selection phase
        # choose the next turn by using the UCB algorithm
        us = []
        for a in self.Psa[s]:
            if board.illegal_move(a):
                us.append(float("-inf"))
            else:
                us.append(self.ucb(s, a))
        a = us.index(max(us))

        next_state, same_player = board.clone_turn(a)
        v = self.search(next_state, 1 if same_player else -1)

        self.Qsa[s][a] = (self.Nsa[s][a] * self.Qsa[s][a] + v) / (1 + self.Nsa[s][a])
        self.Nsa[s][a] += 1

        return v * same_player
