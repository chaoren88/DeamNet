import torch
import torch.nn as nn


class ConvLayer1(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer1, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, stride= stride)

        nn.init.xavier_normal_(self.conv2d.weight.data)

    def forward(self, x):
        # out = self.reflection_pad(x)
        # out = self.conv2d(out)
        return self.conv2d(x)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = (kernel_size-1)//2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride),
            nn.ReLU()
        )
        nn.init.xavier_normal_(self.block[0].weight.data)

    def forward(self, x):
        return self.block(x)


class line(nn.Module):
    def __init__(self):
        super(line, self).__init__()
        self.delta = nn.Parameter(torch.randn(1, 1))

    def forward(self, x ,y ):
        return torch.mul((1-self.delta), x) + torch.mul(self.delta, y)


class Encoding_block(nn.Module):
    def __init__(self, base_filter, n_convblock):
        super(Encoding_block, self).__init__()
        self.n_convblock = n_convblock
        modules_body = []
        for i in range(self.n_convblock-1):
            modules_body.append(ConvLayer(base_filter, base_filter, 3, stride=1))
        modules_body.append(ConvLayer(base_filter, base_filter, 3, stride=2))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        for i in range(self.n_convblock-1):
            x = self.body[i](x)
        ecode = x
        x = self.body[self.n_convblock-1](x)
        return ecode, x


class UpsampleConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        self.conv2d = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, scale_factor=self.upsample)
        out = self.conv2d(x_in)
        return out


class upsample1(nn.Module):
    def __init__(self, base_filter):
        super(upsample1, self).__init__()
        self.conv1 = ConvLayer(base_filter, base_filter, 3, stride=1)
        self.ConvTranspose = UpsampleConvLayer(base_filter, base_filter, kernel_size=3, stride=1, upsample=2)
        self.cat = ConvLayer1(base_filter*2, base_filter, kernel_size=1, stride=1)

    def forward(self, x, y):
        y = self.ConvTranspose(y)
        x = self.conv1(x)
        return self.cat(torch.cat((x, y), dim=1))


class Decoding_block2(nn.Module):
    def __init__(self, base_filter, n_convblock):
        super(Decoding_block2, self).__init__()
        self.n_convblock = n_convblock
        self.upsample = upsample1(base_filter)
        modules_body = []
        for i in range(self.n_convblock-1):
            modules_body.append(ConvLayer(base_filter, base_filter, 3, stride=1))
        modules_body.append(ConvLayer(base_filter, base_filter, 3, stride=1))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x, y):
        x = self.upsample(x, y)
        for i in range(self.n_convblock):
            x = self.body[i](x)
        return x

#Corresponds to DEAM Module in NLO Sub-network
class Attention_unet(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Attention_unet, self).__init__()
        self.conv_du = nn.Sequential(
                ConvLayer1(in_channels=channel, out_channels=channel // reduction, kernel_size=3, stride=1),
                nn.ReLU(inplace=True),
                ConvLayer1(in_channels=channel // reduction, out_channels=channel, kernel_size=3, stride=1),
                nn.Sigmoid()
        )
        self.cat = ConvLayer1(in_channels=channel * 2, out_channels=channel, kernel_size=1, stride=1)
        self.C = ConvLayer1(in_channels=channel, out_channels=channel, kernel_size=3, stride=1)
        self.ConvTranspose = UpsampleConvLayer(channel, channel, kernel_size=3, stride=1, upsample=2)#up-sampling

    def forward(self, x, g):
        up_g = self.ConvTranspose(g)
        weight = self.conv_du(self.cat(torch.cat([self.C(x), up_g], 1)))
        rich_x = torch.mul((1 - weight), up_g) + torch.mul(weight, x)
        return rich_x

#Corresponds to NLO Sub-network
class ziwangluo1(nn.Module):
    def __init__(self, base_filter, n_convblock_in, n_convblock_out):
        super(ziwangluo1, self).__init__()
        self.conv_dila1 = ConvLayer1(64, 64, 3, 1)
        self.conv_dila2 = ConvLayer1(64, 64, 5, 1)
        self.conv_dila3 = ConvLayer1(64, 64, 7, 1)

        self.cat1 = torch.nn.Conv2d(in_channels=64 * 3, out_channels=64, kernel_size=1, stride=1, padding=0,
                                    dilation=1, bias=True)
        nn.init.xavier_normal_(self.cat1.weight.data)
        self.e3 = Encoding_block(base_filter, n_convblock_in)
        self.e2 = Encoding_block(base_filter, n_convblock_in)
        self.e1 = Encoding_block(base_filter, n_convblock_in)
        self.e0 = Encoding_block(base_filter, n_convblock_in)


        self.attention3 = Attention_unet(base_filter)
        self.attention2 = Attention_unet(base_filter)
        self.attention1 = Attention_unet(base_filter)
        self.attention0 = Attention_unet(base_filter)
        self.mid = nn.Sequential(ConvLayer(base_filter, base_filter, 3, 1),
                                 ConvLayer(base_filter, base_filter, 3, 1))
        self.de3 = Decoding_block2(base_filter, n_convblock_out)
        self.de2 = Decoding_block2(base_filter, n_convblock_out)
        self.de1 = Decoding_block2(base_filter, n_convblock_out)
        self.de0 = Decoding_block2(base_filter, n_convblock_out)

        self.final = ConvLayer1(base_filter, base_filter, 3, stride=1)

    def forward(self, x):
        _input = x
        encode0, down0 = self.e0(x)
        encode1, down1 = self.e1(down0)
        encode2, down2 = self.e2(down1)
        encode3, down3 = self.e3(down2)

        # media_end = self.Encoding_block_end(down3)
        media_end = self.mid(down3)

        g_conv3 = self.attention3(encode3, media_end)
        up3 = self.de3(g_conv3, media_end)
        g_conv2 = self.attention2(encode2, up3)
        up2 = self.de2(g_conv2, up3)

        g_conv1 = self.attention1(encode1, up2)
        up1 = self.de1(g_conv1, up2)

        g_conv0 = self.attention0(encode0, up1)
        up0 = self.de0(g_conv0, up1)

        final = self.final(up0)

        return _input+final


class line(nn.Module):
    def __init__(self):
        super(line, self).__init__()
        self.delta = nn.Parameter(torch.randn(1, 1))

    def forward(self, x, y):
        return torch.mul((1 - self.delta), x) + torch.mul(self.delta, y)


class SCA(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCA, self).__init__()
        self.conv_du = nn.Sequential(
                ConvLayer1(in_channels=channel, out_channels=channel // reduction, kernel_size=3, stride=1),
                nn.ReLU(inplace=True),
                ConvLayer1(in_channels=channel // reduction, out_channels=channel, kernel_size=3, stride=1),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv_du(x)
        return y


class Weight(nn.Module):
    def __init__(self, channel):
        super(Weight, self).__init__()
        self.cat =ConvLayer1(in_channels=channel*2, out_channels=channel, kernel_size=1, stride=1)
        self.C = ConvLayer1(in_channels=channel, out_channels=channel, kernel_size=3, stride=1)
        self.weight = SCA(channel)

    def forward(self, x, y):
        delta = self.weight(self.cat(torch.cat([self.C(y), x], 1)))
        return delta


class transform_function(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(transform_function, self).__init__()
        self.ext = ConvLayer1(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1)
        self.pre = torch.nn.Sequential(
            ConvLayer1(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            ConvLayer1(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1),

        )

    def forward(self, x):
        y = self.ext(x)
        return y+self.pre(y)


class Inverse_transform_function(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Inverse_transform_function, self).__init__()
        self.ext = ConvLayer1(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1)
        self.pre = torch.nn.Sequential(
            ConvLayer1(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            ConvLayer1(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1),
        )

    def forward(self, x):
        x = self.pre(x)+x
        x = self.ext(x)
        return x


class Deam(nn.Module):
    def __init__(self, Isreal):
        super(Deam, self).__init__()
        if Isreal:
            self.transform_function = transform_function(3, 64)
            self.inverse_transform_function = Inverse_transform_function(64, 3)
        else:
            self.transform_function = transform_function(1, 64)
            self.inverse_transform_function = Inverse_transform_function(64, 1)

        self.line11 = Weight(64)
        self.line22 = Weight(64)
        self.line33 = Weight(64)
        self.line44 = Weight(64)

        self.net2 = ziwangluo1(64, 3, 2)

    def forward(self, x):
        x = self.transform_function(x)
        y = x

        # Corresponds to NLO Sub-network
        x1 = self.net2(y)
        # Corresponds to DEAM Module
        delta_1 = self.line11(x1, y)
        x1 = torch.mul((1 - delta_1), x1) + torch.mul(delta_1, y)

        x2 = self.net2(x1)
        delta_2 = self.line22(x2, y)
        x2 = torch.mul((1 - delta_2), x2) + torch.mul(delta_2, y)

        x3 = self.net2(x2)
        delta_3 = self.line33(x3, y)
        x3 = torch.mul((1 - delta_3), x3) + torch.mul(delta_3, y)

        x4 = self.net2(x3)
        delta_4 = self.line44(x4, y)
        x4 = torch.mul((1 - delta_4), x4) + torch.mul(delta_4, y)
        x4 = self.inverse_transform_function(x4)
        return x4


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

