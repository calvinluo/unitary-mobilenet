'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1, capacity=2):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        # Initialize the v1 and v2 vectors, as well as the permutations
        self.v1, self.v2, self.ind = tunable_param(in_planes, capacity)
        self.v1 = nn.Parameter(self.v1)
        self.v2 = nn.Parameter(self.v2)
        self.capacity = capacity
        self.concat = True if out_planes // in_planes > 1 else False
        if out_planes // in_planes > 1:
            self.v1_1, self.v2_1, self.ind = tunable_param(in_planes, capacity)
            self.v1_1 = nn.Parameter(self.v1_1)
            self.v2_1 = nn.Parameter(self.v2_1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # Apply our pointwise transformation, represented as a unitary matrix
        rotated = self.loop(out.permute(0,3,2,1), self.v1, self.v2).permute(0,3,2,1)
        # If we wish to increase the channel size, concatenate another block
        if self.concat:
            # Concatenating with another unitary matrix transformation approach
            rotated_1 = self.loop(out.permute(0,3,2,1), self.v1_1, self.v2_1).permute(0,3,2,1)
            rotated = torch.cat([rotated, rotated_1], dim=1)
            # Concatenating with residual block approach
            # rotated = torch.cat([rotated, out], dim=1)
        out = rotated
        # Performing bn and relu on rotated matrix hinders performance, as it is redundant with previous bn + relu transformation
        # out = F.relu(self.bn2(rotated))
        return out

    def loop(self, x, v1, v2):
        """
        Adapted from EUNN-tensorflow: https://github.com/jingli9111/EUNN-tensorflow
        Applies the efficient representation of the decomposition to x: Fx = v1*x + v2*permute(x)
        """
        for i in range(self.capacity):
            diag = x * v1[i,:]
            off = x * v2[i,:]
            x = diag + off[:,:,:,torch.LongTensor(self.ind[i])]
        return x

class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=10, capacity=2):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32, capacity=capacity)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes, capacity):
        if capacity > in_planes:
            raise ValueError("Do not set capacity larger than hidden size, it is redundant")
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride, capacity))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def generate_index_tunable(s, L):
    """
    Adapted from EUNN-tensorflow: https://github.com/jingli9111/EUNN-tensorflow
    generate the index lists for eunn to prepare weight matrices 
    and perform efficient rotations
    This function works for tunable case
    """
    ind1 = list(range(s))
    ind2 = list(range(s))

    for i in range(s):
        if i%2 == 1:
            ind1[i] = ind1[i] - 1
            if i == s -1:
                continue
            else:
                ind2[i] = ind2[i] + 1
        else:
            ind1[i] = ind1[i] + 1
            if i == 0:
                continue
            else:
                ind2[i] = ind2[i] - 1

    ind_exe = [ind1, ind2] * int(L/2)

    ind3 = []
    ind4 = []

    for i in range(int(s/2)):
        ind3.append(i)
        ind3.append(i + int(s/2))

    ind4.append(0)
    for i in range(int(s/2) - 1):
        ind4.append(i + 1)
        ind4.append(i + int(s/2))
    ind4.append(s - 1)

    ind_param = [ind3, ind4]

    return ind_exe, ind_param

def tunable_param(num_units, capacity):
    """
    Adapted from EUNN-tensorflow: https://github.com/jingli9111/EUNN-tensorflow
    Adjusted to simply support orthogonal matrices, as Pytorch has no complex number support
    """
    capacity_A = capacity // 2
    capacity_B = capacity - capacity_A
    phase_init = nn.init.uniform_

    theta_A = phase_init(torch.Tensor(capacity_A, num_units//2), a=-3.14, b=3.14)
    cos_theta_A = torch.cos(theta_A)
    sin_theta_A = torch.sin(theta_A)

    cos_list_A = torch.cat([cos_theta_A, cos_theta_A], 1)
    sin_list_A = torch.cat([sin_theta_A, -sin_theta_A], 1)

    theta_B = phase_init(torch.Tensor(capacity_B, num_units//2-1), a=-3.14, b=3.14)
    cos_theta_B = torch.cos(theta_B)
    sin_theta_B = torch.sin(theta_B)

    cos_list_B = torch.cat([torch.ones(capacity_B, 1), cos_theta_B, cos_theta_B, torch.ones(capacity_B, 1)], 1)
    sin_list_B = torch.cat([torch.zeros(capacity_B, 1), sin_theta_B, -sin_theta_B, torch.zeros(capacity_B, 1)], 1)

    ind_exe, [index_A, index_B] = generate_index_tunable(num_units, capacity)

    diag_list_A = torch.index_select(cos_list_A, 1, torch.LongTensor(index_A))
    off_list_A = torch.index_select(sin_list_A, 1, torch.LongTensor(index_A))
    diag_list_B = torch.index_select(cos_list_B, 1, torch.LongTensor(index_B))
    off_list_B = torch.index_select(sin_list_B, 1, torch.LongTensor(index_B))

    v1 = torch.reshape(torch.cat([diag_list_A, diag_list_B], 1), [capacity, num_units])
    v2 = torch.reshape(torch.cat([off_list_A, off_list_B], 1), [capacity, num_units])

    return v1, v2, ind_exe

def test():
    net = MobileNet()
    x = torch.randn(1,3,32,32)
    y = net(x)

# test()
