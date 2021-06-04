import torch
import torch.nn as nn
import torch.nn.functional as F
class RNNL(nn.Module):
    def __init__(self, input_x_dim, input_y_dim, hidden_dim=64, output_dim=1, residual='none'):
        super(RNNL, self).__init__()
        # self.fcx = nn.Linear(input_x_dim, hidden_dim)   
        # self.fcy = nn.Linear(input_y_dim, hidden_dim)
        self.fcz = nn.Linear(input_x_dim*2, hidden_dim)
        # rnn block mapping train_X to loss 
        self.rnn1 = nn.RNN(hidden_dim, input_x_dim)
        self.rnn2 = nn.RNN(input_x_dim, input_x_dim)
        self.fc1 = nn.Linear(input_x_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # cross entropy loss for learning the residual
        self.residual = residual

        if self.residual == 'residual':
            self.cross_entropy = nn.CrossEntropyLoss()
        self.input_x_dim, self.input_y_dim = input_x_dim, input_y_dim
        nn.init.orthogonal_(self.fc1.weight, gain=1)
        nn.init.orthogonal_(self.fc2.weight, gain=1)
        nn.init.orthogonal_(self.fcz.weight, gain=1)
    def forward(self, x, y):
        if self.residual == 'residual':
            ce = self.cross_entropy(x, y)
        # x = x.unsqueeze(0)
        x = x.type(torch.float).cuda()
        y = y.type(torch.float).cuda()
        y = y.reshape(-1, 1)
        y = y.repeat(1, x.shape[1])
        # Concat and then go thru linear/maybe attention
        z = torch.cat((x,y), dim=1)
        z_emb = self.fcz(z)
        # use y as initial hidden state
        h = y.unsqueeze(0)
        out1, hn1 = self.rnn1(z_emb.unsqueeze(0), h)
        out2, hn2 = self.rnn2(out1, hn1)
        out = self.fc1(out1.squeeze(0)) + z_emb
        # Residual connection
        if self.residual == 'residual':
            out += ce 
        # breakpoint()
        out = F.relu(out)
        # Global average pooling
        out = self.fc2(out.squeeze(0))
        return torch.mean(out).cuda()