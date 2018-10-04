from mcts import MCTS
from network import NeuralNet, args
from game import Game


def new_board():
    return Game(player_turn=1)


network = NeuralNet(board)


def execute_episode():
    examples = []
    mcts = MCTS(board, network, args.c)
    board = new_board()
    while True:
        for _ in range(800):
            mcts.search(board.clone(), False)
        examples.append([mcts.Qsa[str(board)], board.turn_player(), None])
        a = random.choice(len(mcts.Qsa[str(board)]), p=mcts.Qsa[str(board)])
        board.move(a)
        if board.winner() == 0:
            
