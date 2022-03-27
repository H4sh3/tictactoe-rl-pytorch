from torch import nn

INPUT_SIZE = 9
OUTPUT_SIZE = 9
hidden_layer = int((INPUT_SIZE/3)*2)+OUTPUT_SIZE


class NeuralNet(nn.Module):
    def __init__(self):
        # super().__init__()
        super(NeuralNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, OUTPUT_SIZE),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.net.forward(input)


class NetContext:
    def __init__(self, policy_net, target_net, optimizer, loss_function):
        self.policy_net = policy_net

        self.target_net = target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net = self.target_net.eval()

        self.optimizer = optimizer
        self.loss_function = loss_function
