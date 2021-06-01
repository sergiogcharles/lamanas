import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNL(nn.Module):
    def __init__(self, input_x_dim, input_y_dim, hidden_dim=64, output_dim=1):
        super(RNNL, self).__init__()

        self.fcx = nn.Linear(input_x_dim, hidden_dim)   
        self.fcy = nn.Linear(input_y_dim, hidden_dim)

        # rnn block mapping train_X to loss 
        self.rnn1 = nn.RNN(input_x_dim, input_x_dim)
        self.rnn2 = nn.RNN(input_x_dim, input_x_dim)
        self.fc1 = nn.Linear(input_x_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # cross entropy loss for learning the residual
        self.cross_entropy = nn.CrossEntropyLoss()


    def forward(self, x, y):
        ce = self.cross_entropy(x, y)
        x = x.unsqueeze(0)
        x = x.type(torch.float).cuda()
        y = y.type(torch.float).cuda()

        y = y.reshape(-1, 1)
        y = y.repeat(1, x.shape[2])

        # print("y shape", y.shape)
        # print("x shape", x.shape)
        x_emb = self.fcx(x)
        y_emb = self.fcy(y)

        # use y as initial hidden state
        h = y.unsqueeze(0)

        out1, hn1 = self.rnn1(x, h)
        out2, hn2 = self.rnn2(out1, hn1)
        
        out = self.fc1(out1.squeeze(0)) + x_emb + y_emb 
        # print(out.shape)
        # out = torch.sum(out, axis=1).mean()
        # breakpoint()
        # Residual connection
        out += ce 
        # out + ce + x_emb + y_emb
        # print(ce.shape)
        # breakpoint()
        out = F.relu(out)
        
        out = self.fc2(out.squeeze(0))
        # print(out.shape)
        # breakpoint()
        # out = F.relu(out)
        # out = self.fc3(out)
        
        # out = torch.abs(out)
        # Average over minibatch N
        # print(f"Loss: {torch.mean(out)}")
        return torch.mean(out).cuda()