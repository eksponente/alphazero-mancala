# alphazero-mancala
DeepMind's [AlphaGo Zero paper](https://deepmind.com/blog/alphago-zero-learning-scratch/) implementation for the game Mancala.
Experiments are at:
https://www.comet.ml/eksponente/alphazero

## Repository organization
* `game.py` - contains Mancala board implementation and the game logic
* `netowrk.py` - contains implementation of the Neural Network
* `mcts.py` - implementation of Monte Carlo Tree Search, which is guided by the Neural Network
* `main.py` - main file

### TODO:
- [x] Add L2 loss
- [ ] Implement evaluator
- [ ] Add documentation
- [ ] Add tensorboard support
- [ ] Add proper logging
- [ ] Describe results
