import torch
import torch.nn as nn
import torch.nn.functional as F

# From: https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/
class NNL(nn.Module):
    def __init__(self, input_x_dim, input_y_dim, hidden_dim=64, output_dim=1, residual='none'):
        super(NNL, self).__init__()
        #  Linear layer 1
        self.fc1 = nn.Linear(input_x_dim + 1, hidden_dim)
        nn.init.orthogonal_(self.fc1.weight, gain=1)
        # self.fcy = nn.Linear(input_y_dim, hidden_dim) 
        # Linear layer 2
        self.fc2 = nn.Linear(hidden_dim, output_dim)  
        nn.init.orthogonal_(self.fc2.weight, gain=1)
        
        # Linear layer 3
        # self.fc3 = nn.Linear(output_dim, output_dim) 
        # self.batchnorm = nn.BatchNorm1d(output_dim) 
        self.residual = residual

        if self.residual == 'residual':
            self.cross_entropy = nn.CrossEntropyLoss()
            
        self.activation = nn.ELU()

    def forward(self, x, y):
        if self.residual == 'residual':
            ce = self.cross_entropy(x, y)

        # return ce
        
        x = x.type(torch.float).cuda()
        y = y.type(torch.float).reshape(-1, 1).cuda()        
        z = torch.cat((x,y), dim=1)

        # FCC layer 1
        z_emb = self.fc1(z)

        # Residual connection
        out = z_emb
        if self.residual == 'residual':
            out += ce
        out = self.activation(out)
        # FCC layer 2
        out = self.fc2(out)

        # Average over minibatch N
        return torch.mean(out).cuda()
