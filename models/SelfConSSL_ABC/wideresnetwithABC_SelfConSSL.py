import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .Sub_layers import *



def mish(x):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function (https://arxiv.org/abs/1908.08681)"""
    return x * torch.tanh(F.softplus(x))


class PSBatchNorm2d(nn.BatchNorm2d):
    """How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/2001.11216)"""

    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x):
        return super().forward(x) + self.alpha


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.in_planes, self.out_planes= in_planes, out_planes
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class Normalize(nn.Module):
    """ Ln normalization copied from
    https://github.com/salesforce/CoMatch
    """
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class WideResNet_SelfConSSL(nn.Module):
    def __init__(self,
                 num_classes,
                 depth=28,
                 widen_factor=2,
                 drop_rate=0.0,
                 proj=False,
                 proj_after=False,
                 low_dim=64,
                 selfcon_arch='resnet',
                 selfcon_pos=[False,False],selfcon_size='same'):
        super(WideResNet_SelfConSSL, self).__init__()
        # prepare self values
        self.blockexpansion = 1
        self.widen_factor = widen_factor
        self.depth = depth
        self.drop_rate = drop_rate
        # if use projection head
        self.proj = proj
        # if use the output of projection head for classification
        self.proj_after = proj_after
        self.low_dim = low_dim
        self.selfcon_size = selfcon_size
        self.channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        self.selfcon_arch = selfcon_arch
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        self.blocks_num = n
        # (28 - 4) / 6 = 24 / 6 = 4
        self.block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, self.channels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(
            n, self.channels[0], self.channels[1], self.block, 1, drop_rate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(
            n, self.channels[1], self.channels[2], self.block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(
            n, self.channels[2], self.channels[3], self.block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(self.channels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.selfcon_layer = nn.ModuleList([self._make_sub_layer(idx, pos) for idx, pos in enumerate(selfcon_pos)])
        print(f'self.selfcon_layer = {self.selfcon_layer}')
        # if proj after means we classify after projection head
        # so we must change the in channel to low_dim of laster fc
        if self.proj_after:
            self.fc = nn.Linear(self.low_dim, num_classes)
            self.rot = nn.Linear(self.low_dim, 4)
            self.fc2=nn.Linear(self.low_dim, num_classes)   
        else:
            self.fc = nn.Linear(self.channels[3], num_classes)
            self.rot = nn.Linear(self.channels[3], 4)
            self.fc2 = nn.Linear(self.channels[3], num_classes)
        self.channels = self.channels[3]

        # projection head
        if self.proj:
            self.l2norm = Normalize(2)

            self.mlp1 = nn.Linear(64 * self.widen_factor, 64 * self.widen_factor)
            self.relu_mlp = nn.LeakyReLU(inplace=True, negative_slope=0.1)
            self.mlp2 = nn.Linear(64 * self.widen_factor, self.low_dim)
        heads = []
        for pos in selfcon_pos:
            if pos:
                heads.append(nn.Sequential(
                    nn.Linear(64 * self.widen_factor, 64 * self.widen_factor),
                    # nn.LeakyReLU(inplace=True, negative_slope=0.1)
                    nn.ReLU(inplace=True),
                    nn.Linear(64 * self.widen_factor, self.low_dim)
                ))
            else:
                heads.append(None)
        self.sub_heads = nn.ModuleList(heads)
        self.init_wegihts()

    # init_wegihts
    def init_wegihts(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def _make_sub_layer(self, idx, pos):
        channels = self.channels
        strides = [1, 2, 2, 2]
        if self.selfcon_size == 'same':
            num_blocks = [int(self.blocks_num) for _ in range(4)]
        elif self.selfcon_size == 'small':
            num_blocks = [int(n/2) for n in self.blocks_num]
        elif self.selfcon_size == 'large':
            num_blocks = [int(n*2) for n in self.blocks_num]
        elif self.selfcon_size == 'fc':
            pass
        else:
            raise NotImplemented
        
        if not pos:
            return None
        else:
            if self.selfcon_size == 'fc':
                return nn.Linear(channels[idx] * self.blockexpansion, channels[-1] * self.blockexpansion)
            else:
                if self.selfcon_arch == 'resnet':
                    # selfcon layer do not share any parameters
                    layers = []
                    for i in range(idx+2, 4):
                        in_planes = channels[i-1] * self.block.expansion
                        # print(f'in_planes = {in_planes} // out_planes = {channels[i]}')
                        layers.append(resnet_sub_layer(self.block, in_planes, channels[i], num_blocks[i], strides[i]))
                elif self.selfcon_arch == 'vgg':
                    raise NotImplemented
                elif self.selfcon_arch == 'efficientnet':
                    raise NotImplemented

                return nn.Sequential(*layers)

    def forward(self, x):
        sub_out = []
        feat = self.conv1(x)
        feat = self.block1(feat)
        if self.selfcon_layer[0] is not None:
            if self.selfcon_size != 'fc':
                # print(f'feat shape = {feat.shape}')
                out = self.selfcon_layer[0](feat)
                out = torch.flatten(F.adaptive_avg_pool2d(out,1), 1)
            else:
                out = torch.flatten(F.adaptive_avg_pool2d(feat,1), 1)
                out = self.selfcon_layer[0](out)
            sub_out.append(out)

        feat = self.block2(feat)
        
        if self.selfcon_layer[1] is not None:
            if self.selfcon_size != 'fc':
                out = self.selfcon_layer[1](feat)
                out = torch.flatten(F.adaptive_avg_pool2d(out,1), 1)
            else:
                out = torch.flatten(F.adaptive_avg_pool2d(feat,1), 1)
                out = self.selfcon_layer[1](out)
            sub_out.append(out)

        feat = self.block3(feat)

        feat = self.relu(self.bn1(feat))
        feat = F.adaptive_avg_pool2d(feat, 1)
        feat = feat.view(-1, self.channels)


        return feat, sub_out

    
    def classify(self,out): # balanced classifier
        return self.fc(out)
    def rotclassify(self,out):
        return self.rot(out)
    def classify2(self,out): # target classifier (using for inference)
        return self.fc2(out)
    def mlp(self,feature,sub_feature1=None,sub_feature2=None):
        sub_out = []
        
        if self.selfcon_layer[0] is not None and sub_feature1 is not None:
            # if self.selfcon_size != 'fc':
            #     # print(f'feat shape = {feat.shape}')
            #     out = self.selfcon_layer[0](sub_feature1)
            #     out = torch.flatten(F.adaptive_avg_pool2d(out,1), 1)
            # else:
            #     out = torch.flatten(F.adaptive_avg_pool2d(sub_feature1,1), 1)
            #     out = self.selfcon_layer[0](out)
            out = self.sub_heads[0](sub_feature1)
            sub_out.append(self.l2norm(out))
            
        if self.selfcon_layer[1] is not None:
            if sub_feature2 is not None: 
                feat = sub_feature2
            else:
                feat = sub_feature1
            # if self.selfcon_size != 'fc':
            #     out = self.selfcon_layer[1](feat)
            #     out = torch.flatten(F.adaptive_avg_pool2d(out,1), 1)
            # else:
            #     out = torch.flatten(F.adaptive_avg_pool2d(feat,1), 1)
            #     out = self.selfcon_layer[1](out)
            out = self.sub_heads[1](feat)
            sub_out.append(self.l2norm(out))
        
        
        pfeat = self.mlp1(feature)
        pfeat = self.relu_mlp(pfeat)
        pfeat = self.mlp2(pfeat)
        pfeat = self.l2norm(pfeat)

            # if projection after classifiy, we classify last

        return pfeat, sub_out
        


def build_wideresnet_SelfConSSL(dropout=0.,
                     depth=28,
                     widen_factor=2,
                     num_classes=10,
                     proj=True,
                     low_dim=64,
                     selfcon_pos=[False,False],selfcon_size='same',
                     **kwargs):
    
    return WideResNet_SelfConSSL(depth=depth,
                      widen_factor=widen_factor,
                      drop_rate=dropout,
                      num_classes=num_classes,
                      proj=proj,
                      low_dim=low_dim,
                      selfcon_pos=selfcon_pos,selfcon_size=selfcon_size,
                      **kwargs)

