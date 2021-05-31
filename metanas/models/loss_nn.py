import torch
import torch.nn as nn
import torch.nn.functional as F

# From: https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/
class LossNN(nn.Module):
    def __init__(self, input_x_dim, input_y_dim, hidden_dim=64, output_dim=1):
        super(LossNN, self).__init__()
        #  Linear layer 1
        self.fcx = nn.Linear(input_x_dim, hidden_dim)
        self.fcy = nn.Linear(input_y_dim, hidden_dim) 
        # Linear layer 2
        self.fc2 = nn.Linear(hidden_dim, output_dim)  

        # Linear layer 3
        # self.fc3 = nn.Linear(output_dim, output_dim)  

        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, y):
        ce = self.cross_entropy(x, y)

        x = x.type(torch.float).cuda()
        y = y.type(torch.float).cuda()
        
        y = y.reshape(-1, 1)
        y = y.repeat(1, x.shape[1])

        # print("y shape", y.shape)
        x_emb = self.fcx(x)
        y_emb = self.fcy(y)

        # Residual connection
        out = x_emb + y_emb + ce
        out = F.relu(out)
        out = self.fc2(out)
        # out = F.relu(out)
        # out = self.fc3(out)
        
        # out = torch.abs(out)
        # Average over minibatch N
        # print(f"Loss: {torch.mean(out)}")
        return torch.mean(out).cuda()