import torch
from torch import Tensor
import torch.nn as nn
import math
from torch.nn import functional as F_torch
from thop import profile


class Taylor(nn.Module):
    def __init__(self, order_num: int):
        super(Taylor, self).__init__()
        self.F = Fnet()
        self.P = MGC()
        self.order_num = order_num
        self.up = nn.Upsample(scale_factor=4)
        highorderblock_list = []
        for i in range(order_num):

            highorderblock_list.append(Qth())
        self.highorderblock_list = nn.ModuleList(highorderblock_list)
        #self.highorderblock_list = Qth()

    def forward(self, Input1: Tensor, Input2: Tensor):
        feature_head = self.P(Input1)

        Input2 = self.up(Input2)
        zero_term = self.F(Input2)
        out_term, pre_term = zero_term, zero_term
        # a = []
        # b = []
        # a.append(out_term)
        for order_id in range(self.order_num):
            update_term = self.highorderblock_list[order_id](pre_term, Input2, feature_head, self.order_num) + order_id * pre_term
            # update_term = self.highorderblock_list(pre_term, Input2, feature_head,self.order_num) + order_id * pre_term
            pre_term = update_term
            out_term = out_term + update_term / math.factorial(order_id+1)
            # a.append(pre_term)
            # a.append(out_term)
            # b.append(update_term / math.factorial(order_id+1))
        return out_term


def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer


class FFTConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(FFTConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        self.conv_fft_1 = nn.Conv2d(out_size, out_size, 1, 1, 0)
        self.relu_fft_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_fft_2 = nn.Conv2d(out_size, out_size, 1, 1, 0)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1)
        )
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=3, stride=1, padding=1)


    def forward(self, x, enc=None, dec=None):
        H, W = x.shape[-2:]
        x_fft =torch.fft.fft2(x, dim=(-2, -1))
        x_amp = torch.abs(x_fft)
        x_phase = torch.angle(x_fft)

        x_amp1 = self.conv_1(x_amp)
        x_amp11 = self.relu_1(x_amp1)
        x_amp111 = self.conv2(x_amp11)
        # print(x_amp.shape)
        # print(torch.exp(1j*x_phase).shape)
        x_fft_res = torch.fft.ifft2(x_amp111*torch.exp(1j*x_phase), dim=(-2, -1))
        x_fft_res = x_fft_res.real
        out = x + x_fft_res
        out = self.conv4(out)

        x_phase1 = self.conv_1(x_phase)
        x_phase11 = self.relu_1(x_phase1)
        x_phase111 = self.conv2(x_phase11)
        x_fft_res1 = torch.fft.ifft2(x_amp * torch.exp(1j * x_phase111), dim=(-2, -1))
        x_fft_res1 = x_fft_res1.real
        out1 = x + x_fft_res1
        out1 = self.conv4(out1)
        out1 = out + out1
        out1 = self.conv5(out1)

        return out1


class Fnet(nn.Module):
    def __init__(self):
        super(Fnet, self).__init__()
        self.conv1 = nn.Sequential(
             nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=1, padding=1),
             nn.LeakyReLU(0.1)
        )
        self.conv2 = nn.Sequential(
             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
             nn.LeakyReLU(0.1)
        )
        self.conv_local = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1)
        )
        self.fft = FFTConvBlock(in_size=32, out_size=32, relu_slope=0.2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=3, stride=1, padding=1)

    def forward(self, a):
        l1 = self.conv_local(a)
        x1 = self.conv1(a)
        x2 = self.fft(x1)
        x = a + x2 + l1
        return x



class SAB(nn.Module):
    def __init__(self):
        super(SAB, self).__init__()
        # self.conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0)
        # self.up = nn.Upsample(scale_factor=16)
        self.conv4 = nn.Sequential(
             nn.Conv2d(in_channels=48, out_channels=24, kernel_size=3, stride=1, padding=1),
             nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, input1, input2):
        z = torch.cat((input1, input2), dim=1)
        z = self.conv4(z)
        z = torch.sigmoid(z)
        return z


class SAB1(nn.Module):
    def __init__(self):
        super(SAB1, self).__init__()
        # self.conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0)
        # self.up = nn.Upsample(scale_factor=16)
        self.conv4 = nn.Sequential(
             nn.Conv2d(in_channels=48, out_channels=24, kernel_size=3, stride=1, padding=1),
             nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, input1, input2):
        z = torch.cat((input1, input2), dim=1)
        z = self.conv4(z)
        z = torch.sigmoid(z)
        return z


class GConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size1=1, kernel_size2=3, kernel_size3=5, groups=3):
        super(GConv2d, self).__init__()
        self.groups = groups
        self.conv1 = nn.Conv2d(in_channels=in_channels // groups, out_channels=out_channels // groups,
                               kernel_size=kernel_size1, padding=kernel_size1 // 2)
        self.conv2 = nn.Conv2d(in_channels=in_channels // groups, out_channels=out_channels // groups,
                               kernel_size=kernel_size2, padding=kernel_size2 // 2)
        self.conv3 = nn.Conv2d(in_channels=in_channels // groups, out_channels=out_channels // groups,
                               kernel_size=kernel_size3, padding=kernel_size3 // 2)
        self.sgb = SGB()
        self.scab = RCAB(kernel=3, pad=1)
        self.scab3 = RCAB(kernel=3, pad=1)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1))

    def forward(self, x):
        x1, x2, x3 = torch.chunk(x, self.groups, 1)
        y1 = self.conv1(x1)
        y2 = self.conv2(x2)
        y3 = self.conv3(x3)
        z1 = self.sgb(y1, y2, y3)
        z2 = self.sgb(y2, y1, y3)
        z3 = self.sgb(y1, y3, y2)
        z = torch.cat((z1, z2, z3), dim=1)
        z = self.scab(z)
        z = self.conv4(z)
        return z



class SGB(nn.Module):
    def __init__(self):
        super(SGB, self).__init__()
        self.conv = nn.Conv2d(in_channels=24, out_channels=8, kernel_size=3, stride=1, padding=1)

    def forward(self, x, y, z):
        x1 = torch.cat((x, y, z), dim=1)
        x1 = self.conv(x1)
        x1 = x + x1 + y
        return x1


class RCAB(nn.Module):
    def __init__(self, kernel, pad):
        super(RCAB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=kernel, stride=1, padding=pad),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=kernel, stride=1, padding=pad),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(256)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=kernel, stride=1, padding=pad),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=kernel, stride=1, padding=pad),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv1(x)
        y1 = self.avg_pool(y)
        y2 = self.conv2(y1)
        y3 = y * y2
        z = y3 + x
        return z


class MGC(nn.Module):
    def __init__(self):
        super(MGC, self).__init__()
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1)
        )
        self.conv = GConv2d(in_channels=24, out_channels=24, kernel_size1=1, kernel_size2=3, kernel_size3=5)
        self.conv0 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Sequential(
             nn.Conv2d(in_channels=24, out_channels=32, kernel_size=1, stride=1, padding=0),
             nn.LeakyReLU(0.1)
         )

    def forward(self, Input2):
        Fp = self.conv5(Input2)
        Fp = self.conv(Fp)
        return Fp


class MGC1(nn.Module):
    def __init__(self):
        super(MGC1, self).__init__()
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1)
        )
        self.conv = GConv2d(in_channels=24, out_channels=24, kernel_size1=1, kernel_size2=3, kernel_size3=5)
        self.conv0 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Sequential(
             nn.Conv2d(in_channels=24, out_channels=32, kernel_size=1, stride=1, padding=0),
             nn.LeakyReLU(0.1)
         )

    def forward(self, Input2):
        Fp = self.conv5(Input2)
        Fp = self.conv(Fp)
        return Fp


class _ScaleAwareConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            padding: str,
            bias: bool = False,
            num_experts: int = 4,
    ) -> None:
        super(_ScaleAwareConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.num_experts = num_experts

        # Use fc layers to generate routing weights
        self.routing = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(True),
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.Linear(32, num_experts),
            nn.Softmax(1)
        )

        # Initialize experts
        weight_pool = []
        for i in range(num_experts):
            weight_pool.append(nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)))
            nn.init.kaiming_uniform_(weight_pool[i], math.sqrt(5))
        self.weight_pool = nn.Parameter(torch.stack(weight_pool, 0))

        if bias:
            self.bias_pool = nn.Parameter(torch.Tensor(num_experts, out_channels))
            # Calculate fan_in
            dimensions = self.weight_pool.dim()
            if dimensions < 2:
                raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

            num_input_feature_maps = self.weight_pool.size(1)
            receptive_field_size = 1
            if self.weight_pool.dim() > 2:
                # math.prod is not always available, accumulate the product manually
                # we could use functools.reduce but that is not supported by TorchScript
                for s in self.weight_pool.shape[2:]:
                    receptive_field_size *= s
            fan_in = num_input_feature_maps * receptive_field_size
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_pool, -bound, bound)

    def forward(self, x: Tensor, order_number) -> Tensor:
        device = x.device
        # Use fc layers to generate routing weights
        order_number /= torch.ones(1, 1).to(device)
        routing_weights = self.routing(order_number).view(self.num_experts, 1, 1)

        # Fuse experts
        fused_weight = (self.weight_pool.view(self.num_experts, -1, 1) * routing_weights).sum(0)
        fused_weight = fused_weight.view(-1, self.in_channels, self.kernel_size, self.kernel_size)

        if self.bias:
            fused_bias = torch.mm(routing_weights, self.bias_pool).view(-1)
        else:
            fused_bias = None

        out = F_torch.conv2d(x, fused_weight, fused_bias, self.stride, self.padding)

        return out


class Qth(nn.Module):
    def __init__(self, inchannels:int = 24, outchannels: int = 24):
        super(Qth, self).__init__()
        self.MGC = MGC()
        self.MGC1 = MGC1()
        self.SAB = SAB()
        self.SAB1 = SAB1()
        self.conv = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.down = nn.MaxPool2d(4)
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1)
        )
        self.conv8 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.adaption = _ScaleAwareConv(inchannels, outchannels, 3, 1, (1,1))

    def forward(self, pre_term: Tensor, Input2: Tensor, feature_head: Tensor, order_number):
        pre_term = self.conv(pre_term)
        x = self.MGC1(pre_term)
        x = self.adaption(x, order_number)
        Input2 = self.conv(Input2)
        x2 = self.MGC1(Input2)
        feature_head_1 = self.adaption(feature_head, order_number)
        x3 = self.SAB(feature_head_1, x2)
        x3 = x3 * feature_head_1
        x3 = x3 + feature_head
        x4 = self.SAB(x, x2)
        x4 = x4 * x
        x5 = x4 + x2 + x3
        x5 = self.conv7(x5)
        return x5


if  __name__ == "__main__":
    dc = Taylor(order_num=5)
    A = torch.FloatTensor(size=(1, 1, 256, 256)).normal_(0, 1)
    B = torch.FloatTensor(size=(1, 4, 64, 64)).normal_(0, 1)
    out = dc(A, B)
    flops, params = profile(dc,(A,B))
    print('flops:', flops, 'params:', params)
    # print()