import torch
import torch.nn as nn
import torch.nn.functional as F

# From: https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/
class NNL(nn.Module):
    def __init__(self, input_x_dim, input_y_dim, hidden_dim=64, output_dim=1):
        super(NNL, self).__init__()
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
        
        print(y.shape)
        y = y.reshape(-1, 1)
        y = y.repeat(1, x.shape[1])
        # print(y.shape)
        if y.shape[0] == 10:
            breakpoint()
        # print("y shape", y.shape)
        x_emb = self.fcx(x)
        y_emb = self.fcy(y)
        # print(x_emb.shape, y_emb.shape)
        # Residual connection
        out = x_emb + y_emb + ce
        # print(ce.shape, out.shape)
        # print(out.shape)
        out = F.relu(out)
        out = self.fc2(out)
        # print(out.shape)
        # breakpoint()
        # out = F.relu(out)
        # out = self.fc3(out)
        
        # out = torch.abs(out)
        # Average over minibatch N
        # print(f"Loss: {torch.mean(out)}")
        return torch.mean(out).cuda()