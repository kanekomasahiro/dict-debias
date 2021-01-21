import torch
import torch.nn as nn
from torch import cuda


class GradReverse(torch.autograd.Function):

    def __init__(self, lambd):
        self.lambd = lambd


    def forward(self, x):
        return x.view_as(x)


    def backward(self, grad_output):
        return (grad_output * -self.lambd)


class Encoder(nn.Module):

    def __init__(self, emb_size, hidden_size, dropout_rate=0.):
        super(Encoder, self).__init__()
        self.hidden_layer = nn.Linear(emb_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.tanh = nn.Tanh()


    def forward(self, emb):
        hidden = self.tanh(self.hidden_layer(self.dropout(emb)))

        return hidden


class Decoder(nn.Module):

    def __init__(self, hidden_size, emb_size, dropout_rate=0., grad_reverse_flag=False):
        super(Decoder, self).__init__()
        self.output_layer = nn.Linear(hidden_size, emb_size)
        self.grad_reverse_flag = grad_reverse_flag
        self.dropout = nn.Dropout(p=dropout_rate)
        self.tanh = nn.Tanh()

        if self.grad_reverse_flag:
            self.grad_reverse = GradReverse(-1)


    def forward(self, hidden):
        if self.grad_reverse_flag:
            hidden = self.grad_reverse(hidden)
        output = self.tanh(self.output_layer(self.dropout(hidden)))

        return output
