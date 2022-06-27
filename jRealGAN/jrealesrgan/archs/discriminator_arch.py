
import jittor as jt
from jittor import init
from jittor import nn

from jrealesrgan.registry import ARCH_REGISTRY

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1, eps=1e-12):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        self.eps = eps
        if not self._made_params():
            self._make_params()

    def l2normalize(self, v):
        return v / (v.norm() + self.eps)

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name)

        height = w.shape[0]
        for _ in range(self.power_iterations):
            v.assign(self.l2normalize((w.view(height,-1).t() * u.unsqueeze(0)).sum(-1)))
            u.assign(self.l2normalize((w.view(height,-1) * v.unsqueeze(0)).sum(-1)))
        sigma = (u * (w.view(height,-1) * v.unsqueeze(0)).sum(-1)).sum()
        getattr(self.module, self.name).assign(w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name)
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.shape[0]
        width = w.view(height, -1).shape[1]

        u = jt.empty([height], dtype=w.dtype).gauss_(0, 1)
        v = jt.empty([width], dtype=w.dtype).gauss_(0, 1)
        u = self.l2normalize(u)
        v = self.l2normalize(v)

        setattr(self.module, self.name + "_u", u.stop_grad())
        setattr(self.module, self.name + "_v", v.stop_grad())

    def execute(self, *args):
        self._update_u_v()
        return self.module.execute(*args)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.gauss_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            init.gauss_(m.bias, 0.0, 0.02)

@ARCH_REGISTRY.register()
class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = SpectralNorm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

        # initialization(added)
        for m in self.modules():
            weights_init_normal(m)

    def execute(self, x):
        # downsample
        x0 = nn.leaky_relu(self.conv0(x), scale=0.2)
        x1 = nn.leaky_relu(self.conv1(x0), scale=0.2)
        x2 = nn.leaky_relu(self.conv2(x1), scale=0.2)
        x3 = nn.leaky_relu(self.conv3(x2), scale=0.2)

        # upsample
        x3 = nn.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = nn.leaky_relu(self.conv4(x3), scale=0.2)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = nn.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = nn.leaky_relu(self.conv5(x4), scale=0.2)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = nn.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = nn.leaky_relu(self.conv6(x5), scale=0.2)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = nn.leaky_relu(self.conv7(x6), scale=0.2)
        out = nn.leaky_relu(self.conv8(out), scale=0.2)
        out = self.conv9(out)

        return out
