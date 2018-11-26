import torch
import torch.nn as nn


class NN(nn.Module):
    def __init__(self, num_of_layers=5, output_func='sigmoid', activation_function='relu', input_size=(257*9),
                 output_size=257, hidden_cells=1000):
        super(NN, self).__init__()

        layers = []
        activation_functions = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'leakyrelu': nn.LeakyReLU
        }
        # TODO: if we want we can change the hidden_cells for a list with the size for every layer
        active_layer = None
        layers.append(nn.Linear(in_features=input_size, out_features=hidden_cells))
        layers.append(nn.BatchNorm1d(hidden_cells))
        for _ in range(num_of_layers-2):
            layers.append(nn.Linear(in_features=hidden_cells, out_features=hidden_cells))
            layers.append(nn.BatchNorm1d(hidden_cells))
            layers.append(activation_functions[activation_function](inplace=True))
        layers.append(nn.Linear(in_features=hidden_cells, out_features=output_size))
        try:
            if output_func.lower() == 'softmax':
                active_layer = nn.Softmax()
            elif output_func.lower() == 'sigmoid':
                active_layer = nn.Sigmoid()
            else:
                raise NameError('Activation not supported')
        except NameError:
            print(output_func, 'not support, Please change')

        self.nn = nn.Sequential(*layers)
        self.activation = active_layer

    def forward(self, x):
        out = self.nn(x)
        out2 = self.activation(out)
        return out2
