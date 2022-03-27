import time
from board import CELL_X, Board, play_games, play_random_move
from neuralnet import NetContext, NeuralNet
import torch
import numpy as np

from qneural import create_qneural_player, get_q_values, play_training_games
from tensorboardX import SummaryWriter

policy_net = NeuralNet()
target_net = NeuralNet()
sgd = torch.optim.SGD(policy_net.parameters(), lr=0.1)
adam = torch.optim.Adam(policy_net.parameters(), lr=0.01)
loss = torch.nn.MSELoss()
net_context = NetContext(policy_net, target_net, sgd, loss)

with torch.no_grad():
    board = Board(np.array([1, -1, -1, 0, 1, 1, 0, 0, -1]))
    q_values = get_q_values(board, net_context.target_net)
    print(f"Before training q_values = {q_values}")

total_games = 100
discount_factor = 1.0
epsilon = 0.7
o_strategies = None
results = []

writer = SummaryWriter(logdir='logs', comment=f"dqn logs")


def eval_games(i):
    with torch.no_grad():
        play_qneural_move = create_qneural_player(net_context)
        stats = play_games(100, play_qneural_move, play_random_move)
        results.append(stats)

        writer.add_scalar("score/x wins", int(stats[0]), i)
        writer.add_scalar("score/o wins", int(stats[1]), i)
        writer.add_scalar("score/draws", int(stats[2]), i)


eval_games(0)
for i in range(1, 100):
    start = time.time()
    # train for 1000 games

    # self play
    play_qneural_move = create_qneural_player(net_context)

    play_training_games(net_context, CELL_X, 10000,
                        discount_factor, epsilon, None, [play_random_move])

    # play 100 games for evaluation
    eval_games(i)
    print(f'took {time.time()-start}')

for result in results:
    print(f'x:{result[0]} o:{result[1]} d:{result[2]}')

with torch.no_grad():
    board = Board(np.array([1, -1, -1, 0, 1, 1, 0, 0, -1]))
    q_values = get_q_values(board, net_context.target_net)
    print(f"After training q_values = {q_values}")
