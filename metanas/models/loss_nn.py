import torch
import torch.nn as nn
import torch.nn.functional as F

# From: https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/
class NNL(nn.Module):
    def __init__(self, input_x_dim, input_y_dim, hidden_dim=64, output_dim=1):
        super(NNL, self).__init__()
        #  Linear layer 1
        self.fcx = nn.Linear(input_x_dim, hidden_dim)
        # self.fcy = nn.Linear(input_y_dim, hidden_dim) 
        # Linear layer 2
        self.fc2 = nn.Linear(hidden_dim, output_dim)  
        
        # Linear layer 3
        # self.fc3 = nn.Linear(output_dim, output_dim) 
        # self.batchnorm = nn.BatchNorm1d(output_dim) 

        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, y):
        ce = self.cross_entropy(x, y)

        x = x.type(torch.float).cuda()
        # print(x.shape)
        y = y.type(torch.float).reshape(1, -1).cuda()
        # print(y.shape)
        z = torch.cat((x,y), dim=0)
        # print("z dim", z.shape)
        # print('stack ', z.shape)
        
        # y = y.reshape(-1, 1)
        # y = y.repeat(1, x.shape[1])
        
        # print("y shape", y.shape)
        z_emb = self.fcx(z)
        # y_emb = self.fcy(y)

        # TODO
        # Suggestions: stack on first few pretrained layers of VGG
        # Global average pooling

        # Concat and then go thru linear/maybe attention
        # torch.cat((x_emb, y_emb))


        # Residual connection
        out = z_emb + ce
        # + y_emb + ce
        out = self.fc2(out)
        # out = self.fc3(out)
        # Batch norm
        # out = self.batchnorm(out)
        # out = 1 + torch.sigmoid(out)

        # out = torch.abs(out)
        # Average over minibatch N
        # print(f"Loss: {torch.mean(out)}")
        return torch.mean(out).cuda()