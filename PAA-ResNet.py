import torch.nn as nn
import torchvision

from modules import *
from utils import *
import torch.autograd as autograd
import torch.nn.functional as F

class AA(nn.Module):

    def __init__(self, input_size,output_size, dropout=0, num_layers=1):
        super(AIA_Transformer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.k1 = Parameter(torch.ones(1))
        self.k2 = Parameter(torch.ones(1))

        self.input = nn.Sequential(
            nn.Conv2d(input_size, input_size // 2, kernel_size=1),
            nn.PReLU()
        )

        # dual-path RNN
        self.row_trans = nn.ModuleList([])
        self.col_trans = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_layers):
            self.row_trans.append(TransformerEncoderLayer(d_model=input_size//2, nhead=4, dropout=dropout))
            self.col_trans.append(TransformerEncoderLayer(d_model=input_size//2, nhead=4, dropout=dropout))
            self.row_norm.append(nn.GroupNorm(1, input_size//2, eps=1e-8))
            self.col_norm.append(nn.GroupNorm(1, input_size//2, eps=1e-8))

        # output layer
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv2d(input_size//2, output_size, 1)
                                    )

    def forward(self, input):
        #  input --- [b,  c,  num_frames, frame_size]  --- [b, c, dim2, dim1]
        b, c, dim2, dim1 = input.shape
        output_list = []
        output = self.input(input)
        for i in range(len(self.row_trans)):
            row_input = output.permute(3, 0, 2, 1).contiguous().view(dim1, b*dim2, -1)  # [F, B*T, c]
            row_output = self.row_trans[i](row_input)  # [F, B*T, c]
            row_output = row_output.view(dim1, b, dim2, -1).permute(1, 3, 2, 0).contiguous()  # [B, C, T, F]
            row_output = self.row_norm[i](row_output)  # [B, C, T, F]

            col_input = output.permute(2, 0, 3, 1).contiguous().view(dim2, b*dim1, -1)
            col_output = self.col_trans[i](col_input)
            col_output = col_output.view(dim2, b, dim1, -1).permute(1, 3, 0, 2).contiguous()
            col_output = self.col_norm[i](col_output)
            output = output + self.k1*row_output + self.k2*col_output
            output_i = self.output(output)
            output.append(output_i)
        del row_input, row_output, col_input, col_output

        return output

class PAA_ResNet(nn.Module):

    def __init__(self, args, bn_adv_flag=False, bn_adv_momentum = 0.01):
        super(PAA_ResNet, self).__init__()

        self.args = args
        self.bn_adv_flag = bn_adv_flag
        self.bn_adv_momentum = bn_adv_momentum
        self.inplanes = 64
        self.bn_adv_flag = bn_adv_flag
        self.bn_adv_momentum = bn_adv_momentum

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        if bn_adv_flag:
            self.bn1_adv = nn.BatchNorm2d(64, momentum=self.bn_adv_momentum)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], bn_adv_flag=self.bn_adv_flag,
                                       bn_adv_momentum=bn_adv_momentum)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, bn_adv_flag=self.bn_adv_flag,
                                       bn_adv_momentum=bn_adv_momentum)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, bn_adv_flag=self.bn_adv_flag,
                                       bn_adv_momentum=bn_adv_momentum)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, bn_adv_flag=self.bn_adv_flag,
                                       bn_adv_momentum=bn_adv_momentum)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, low_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.attention = AA(input_size,output_size, dropout=0, num_layers=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, adv=False):
        x = self.conv1(x)
        if adv and self.bn_adv_flag:
            out = self.bn1_adv(x)
        else:
            out = self.bn1(x)

        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, adv=adv)
        x = self.attention1(x)
        x = self.layer2(x, adv=adv)
        x = self.attention1(x)
        x = self.layer3(x, adv=adv)
        x = self.attention1(x)
        x = self.layer4(x, adv=adv)
        x = self.attention1(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        z = self.projector(x)

        if self.args.normalize:
            z = nn.functional.normalize(z, dim=1)
        return h, z
