
import jittor as jt
from jittor import init
from jittor import nn

from jrealesrgan.registry import ARCH_REGISTRY

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.gauss_(m.weight, 0.0, 0.02)
        init.gauss_(m.bias, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        init.gauss_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0.0)

class DenseResidualBlock(nn.Module):
    def __init__(self, filters, grow_filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, out_features, non_linearity=True):
            layers = [nn.Conv(in_features, out_features, 3, stride=1, padding=1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU(scale=0.2)]
            return nn.Sequential(*layers)
        self.b1 = block(in_features=(filters), out_features=grow_filters)
        self.b2 = block(in_features=(1 * grow_filters + filters), out_features=grow_filters)
        self.b3 = block(in_features=(2 * grow_filters + filters), out_features=grow_filters)
        self.b4 = block(in_features=(3 * grow_filters + filters), out_features=grow_filters)
        self.b5 = block(in_features=(4 * grow_filters + filters), out_features=filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]
        
    def execute(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = jt.contrib.concat([inputs, out], dim=1)
        return (out * self.res_scale + x)

class ResidualInResidualDenseBlock(nn.Module):

    def __init__(self, filters, grow_filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(DenseResidualBlock(filters, grow_filters), DenseResidualBlock(filters, grow_filters), DenseResidualBlock(filters, grow_filters))

    def execute(self, x):
        return (self.dense_blocks(x) * self.res_scale + x)

@ARCH_REGISTRY.register()
class RRDBNet(nn.Module):

    def __init__(self, num_in_ch, num_out_ch, num_feat=64, num_grow_ch=32, num_block=16, num_upsample=2):
        super(RRDBNet, self).__init__()
        self.conv1 = nn.Conv(num_in_ch, num_feat, 3, stride=1, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv2 = nn.Conv(num_feat, num_feat, 3, stride=1, padding=1)
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [jt.make_module(nn.interpolate, 6)(scale_factor=2, mode='nearest'), nn.Conv(num_feat, num_feat, 3, stride=1, padding=1), nn.LeakyReLU(scale=0.2)]
        self.upsampling = nn.Sequential(*upsample_layers)
        self.conv3 = nn.Sequential(nn.Conv(num_feat, num_feat, 3, stride=1, padding=1), nn.LeakyReLU(scale=0.2), nn.Conv(num_feat, num_out_ch, 3, stride=1, padding=1))
        for m in self.modules():
            weights_init_normal(m)

    def execute(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = out1 + out2
        out = self.upsampling(out)
        out = self.conv3(out)
        return out