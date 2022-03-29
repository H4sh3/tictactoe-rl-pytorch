import itertools
import torch
from board import Board, play_game, play_games
from neuralnet import NetContext, NeuralNet
from qneural import create_qneural_player, create_qneural_player_from_model



policy_net = NeuralNet()
target_net = NeuralNet()

checkpoint = torch.load('policy_net_v2.pt')
policy_net.load_state_dict(checkpoint)

checkpoint = torch.load('target_net_v2.pt')
target_net.load_state_dict(checkpoint)

sgd = torch.optim.SGD(policy_net.parameters(), lr=0.1)
adam = torch.optim.Adam(policy_net.parameters(), lr=0.01)
loss = torch.nn.MSELoss()
net_context = NetContext(policy_net, target_net, sgd, loss)

play_qneural_move = create_qneural_player_from_model(net_context.target_net)

def human_move(board):
    print('input 0-9 ?')

    move_index = int(input())
    while move_index in board.get_illegal_move_indexes():
        print("illegal move!")
        move_index = int(input())
    return board.play_move(move_index)


board = Board()
player_strategies = itertools.cycle([human_move,play_qneural_move])

while not board.is_gameover():
    play = next(player_strategies)
    board = play(board)
    board.print_board()
print("game over")