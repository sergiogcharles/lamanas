import torch
import torch.nn as nn
import torch.nn.functional as F
class RNNL(nn.Module):
    def __init__(self, input_x_dim, input_y_dim, hidden_dim=64, output_dim=1, num_layers=5, residual='none'):
        super(RNNL, self).__init__()
        # self.fcx = nn.Linear(input_x_dim, hidden_dim)   
        # self.fcy = nn.Linear(input_y_dim, hidden_dim)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.residual = residual
        self.fcz = nn.Linear(input_x_dim + 1, hidden_dim)
        # rnn block mapping train_X to loss 
        # self.rnn1 = nn.RNN(hidden_dim, input_x_dim)
        # self.rnn2 = nn.RNN(input_x_dim, input_x_dim)
        # self.fc1 = nn.Linear(input_x_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, output_dim)
        # # cross entropy loss for learning the residual
        # self.residual = residual

        if self.residual == 'residual':
            self.cross_entropy = nn.CrossEntropyLoss()
        # self.input_x_dim, self.input_y_dim, self.hidden_dim = input_x_dim, input_y_dim, hidden_dim
        # nn.init.orthogonal_(self.fc1.weight, gain=1)
        # nn.init.orthogonal_(self.fc2.weight, gain=1)
        # nn.init.orthogonal_(self.fcz.weight, gain=1)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers)
        self.fc1 = nn.Linear(hidden_dim, 32)
        nn.init.orthogonal_(self.fc1.weight, gain=1)
        self.fc2 = nn.Linear(32, output_dim)
        nn.init.orthogonal_(self.fc2.weight, gain=1)

    def forward(self, x, y):
        if self.residual == 'residual':
            ce = self.cross_entropy(x, y)
        # x = x.unsqueeze(0)
        x = x.type(torch.float).cuda()
        y = y.type(torch.float).cuda()
        y = y.reshape(-1, 1)
        # y = y.repeat(1, x.shape[1])
        # Concat and then go thru linear/maybe attention
        z = torch.cat((x,y), dim=1)
        z_emb = self.fcz(z)
        z_emb = z_emb.unsqueeze(0)
        z_emb = F.relu(z_emb)

        # print('input', z_emb.shape)
        self.lstm.flatten_parameters()
        # make z_emb go from (N, H) to (1, N, H) to align with (time_steps, batch_size, in_size)
        # Initial hidden/cell state h_0=c_0 of shape (num_layers, batch, hidden_size)
        h = y.unsqueeze(0)
        # print(h.shape)
        h = h.repeat(self.num_layers, 1, self.hidden_dim)
        # print('h shape', h.shape)

        (h_0, c_0) = (h, h)
        out_seq, _ = self.lstm(z_emb, (h_0, c_0))
        out = out_seq.squeeze(0)
        out = self.fc1(out)

        # Residual connection
        if self.residual == 'residual':
            out += ce 
        out = F.relu(out)

        out = self.fc2(out)

        # print('seq', output_seq.shape)
        # out = self.fc1(output_seq)
        # out = output_seq[-1].cuda()
        # print('out', out.shape)
        return torch.mean(out)

        # model = nn.LSTM(in_size, classes_no, 2)
        # input_seq = Variable(torch.randn(time_steps, batch_size, in_size))
        # output_seq, _ = model(input_seq)
        # last_output = output_seq[-1]


        # print("z", z.shape, "z_emb", z_emb.shape, "x", x.shape, "y", y.shape)
        # use y as initial hidden state
        # h = y.unsqueeze(0)
        # h = y.unsqueeze(0)
        # # print(h.shape)
        # h = h.repeat(1, 1, h.shape[1])
        # # print(h.shape)
        # # breakpoint()

        # # self.rnn1.flatten_parameters()
        # # self.rnn2.flatten_parameters()

        # print('squeeze', z_emb.unsqueeze(0).shape)

        # # seq length of 1
        # out1, hn1 = self.rnn1(z_emb.unsqueeze(0), h)
        # out2, hn2 = self.rnn2(out1, hn1)
        # out = self.fc1(out2.squeeze(0)) + z_emb
        # # Residual connection
        # if self.residual == 'residual':
        #     out += ce 
        # # breakpoint()
        # out = F.relu(out)
        # # Global average pooling
        # out = self.fc2(out.squeeze(0))
        # return torch.mean(out).cuda()