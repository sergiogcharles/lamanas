import torch
import torch.nn as nn
import torch.nn.functional as F

# From: https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/
class NNL(nn.Module):
    def __init__(self, input_x_dim, input_y_dim, hidden_dim=64, output_dim=1):
        super(NNL, self).__init__()
        #  Linear layer 1
        self.fcx = nn.Linear(input_x_dim + 1, hidden_dim)
        nn.init.orthogonal_(self.fcx.weight, gain=1)
        # self.fcy = nn.Linear(input_y_dim, hidden_dim) 
        # Linear layer 2
        self.fc2 = nn.Linear(hidden_dim, output_dim)  
        nn.init.orthogonal_(self.fc2.weight, gain=1)
        
        # Linear layer 3
        # self.fc3 = nn.Linear(output_dim, output_dim) 
        # self.batchnorm = nn.BatchNorm1d(output_dim) 

        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, y):
        ce = self.cross_entropy(x, y)
        return ce
        # x = x.type(torch.float).cuda()
        # y = y.type(torch.float).reshape(-1, 1).cuda()        
        # z = torch.cat((x,y), dim=1)

        # z_emb = self.fcx(z)

        # # Residual connection
        # out = z_emb + ce
        # # + y_emb + ce
        # out = self.fc2(out)

        # # Average over minibatch N
        # return torch.mean(out).cuda()