import cv2
import torch
import torch.nn as nn
from net.restormer import TransformerBlock as Restormer
import torch.nn.functional as F
from einops import rearrange
import numbers

#Restormer
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        # point-wise convolution 1x1
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        # depth-wise convolution groups=in_channels
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        # 1x1
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x): # (b,c,h,w)
        # point-wise convolution
        x = self.project_in(x)
        # depth-wise convolution
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2 # (b,hidden_features,h,w)
        x = self.project_out(x) # (b,c,h,w)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1)) # (num_heads,1,1)

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        # depth-wise groups=in_channels
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x): # x: (b,dim,h,w)
        b,c,h,w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)

        # (b,dim,h,w)->(b,num_head,c,h*w)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1) #  (b,num_head,c,c)
        out = (attn @ v)
        # reshape: (b,num_head,c,h*w)->(b,num_head*c,h,w)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # 1x1conv: (b,dim,h,w)
        out = self.project_out(out) # dim=c*num_head
        return out # (b,c,h,w)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x): # (b,c,h,w)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x # (b,c,h,w)

class TransformerBlock_Base(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_Base, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x): # (b,c,h,w)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x # (b,c,h,w)
#Restormer End

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

class LinearMappingLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearMappingLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )
    def forward(self, x):
        return self.bottleneckBlock(x)

class DetailNode(nn.Module):
    def __init__(self):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=16, oup=16, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=16, oup=16, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=16, oup=16, expand_ratio=2)
        self.shffleconv = nn.Conv2d(32, 32, kernel_size=1,
                                    stride=1, padding=0, bias=True)
    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2
    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2

class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=3):
        super(DetailFeatureExtraction, self).__init__()
        INNmodules = [DetailNode() for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)
    def forward(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)

class Restormer_Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 dim=32,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(Restormer_Encoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.baseFeature = TransformerBlock_Base(dim=dim, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.detailFeature = DetailFeatureExtraction()

    def forward(self, inp_img):
        inp_img = inp_img.permute(0, 3, 1, 2)
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        base_feature = self.baseFeature(out_enc_level1)
        detail_feature = self.detailFeature(out_enc_level1)
        return base_feature, detail_feature

class Restormer_Decoder(nn.Module):
    def __init__(
        self,
        out_channels=1,
        dim=16,
        num_blocks=[4, 4],
        heads=[8, 8, 8],
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type="WithBias",
    ):

        super(Restormer_Decoder, self).__init__()
        self.reduce_channel = nn.Conv2d(
            2*dim, int(dim), kernel_size=1, bias=bias
        )
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=dim,num_heads=heads[1],ffn_expansion_factor=ffn_expansion_factor,bias=bias,LayerNorm_type=LayerNorm_type,)for i in range(num_blocks[1])])
        self.output = nn.Sequential(
            nn.Conv2d(
                int(dim), int(dim) // 2, kernel_size=3, stride=1, padding=1, bias=bias
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                int(dim) // 2,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            ),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, base_feature, detail_feature):
        out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1)
        out_enc_level0 = self.reduce_channel(out_enc_level0)
        out_enc_level1 = self.encoder_level2(out_enc_level0)
        '''if inp_img is not None:
            out_enc_level1 = self.output(out_enc_level1) + inp_img
        else:'''
        out_enc_level1 = self.output(out_enc_level1)
        Fuse_Image = self.sigmoid(out_enc_level1)
        return Fuse_Image

class BaseFuseLayer(nn.Module):
    def __init__(
        self,
        dim=32,
        num_heads=8,
        ffn_expansion_factor=1.0,
        bias=False,
    ):
        super(BaseFuseLayer, self).__init__()
        self.norm1 = LayerNorm(dim, "WithBias")
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, "WithBias")
        self.mlp = FeedForward(dim,ffn_expansion_factor,bias)
    def forward(self, Basic_feature):
        Basic_feature = Basic_feature + self.attn(self.norm1(Basic_feature))
        Basic_feature = Basic_feature + self.mlp(self.norm2(Basic_feature))
        return Basic_feature


class imagefeature2textfeature2(nn.Module):
    def __init__(self, in_channel, mid_channel, hidden_dim):
        super(imagefeature2textfeature2, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=mid_channel, kernel_size=1)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x, [160, 224], mode='nearest')
        x = x.contiguous().view(x.size(0), x.size().numel() // x.size(0) // self.hidden_dim, self.hidden_dim)
        return x
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, query, key, value):
        query = query.transpose(0, 1).float()
        key = key.transpose(0, 1).float()
        value = value.transpose(0, 1).float()

        attn_output, _ = self.multihead_attn(query, key, value)
        attn_output = attn_output.transpose(0, 1)

        return attn_output

class CrossAttentionFusion(nn.Module):
    def __init__(
            self,
            input_channel,
            image2text_dim,
            hidden_dim=256,
            restormerdim=32,
            restormerhead=4,
            ffn_expansion_factor=4,
            bias=False,
            LayerNorm_type='WithBias',
            pooling='avg',
            normalization='l1'
    ):
        super(CrossAttentionFusion, self).__init__()
        self.imagef2textf1 = imagefeature2textfeature2(restormerdim, image2text_dim, hidden_dim)
        self.cross_attention = CrossAttention(embed_dim=hidden_dim, num_heads=8)
        self.image2text_dim = image2text_dim
        self.conv2 = nn.Conv2d(image2text_dim, restormerdim, kernel_size=1)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(2 * restormerdim, restormerdim, kernel_size=1)
        self.prelu3 = nn.PReLU()
    def forward(self, imageA, imageB):
        b, _, H, W = imageA.shape
        #imageA_resize = F.interpolate(imageA, size=(128, 128), mode='bilinear', align_corners=False)
        #imageB_resize = F.interpolate(imageB, size=(128, 128), mode='bilinear', align_corners=False)
        imagetotextA = self.imagef2textf1(imageA)
        imagetotextB = self.imagef2textf1(imageB)
        ca = self.cross_attention(imagetotextB, imagetotextA, imagetotextA)  # (B,L,512)
        ca = torch.nn.functional.adaptive_avg_pool1d(ca.permute(0, 2, 1), 1).permute(0, 2, 1)
        ca = F.normalize(ca, p=1, dim=2)
        ca = (imagetotextA * ca).view(imageA.shape[0], self.image2text_dim, 160, 224)
        ca = F.interpolate(ca, [H, W], mode='nearest')
        image_sideout = imageA
        image_sideout = F.interpolate(image_sideout, [H, W], mode='nearest')
        ca = self.prelu3(self.conv3(torch.cat((F.interpolate(imageA, [H, W], mode='nearest'), self.prelu2(self.conv2(ca)) + image_sideout), 1)))
        return ca

class Net(nn.Module):
    def __init__(
            self,
            mid_channel=32,
            decoder_num_heads=8,
            bias=False,
            LayerNorm_type='WithBias',
            out_channel = 1,
            ffn_factor=4
    ):
        super().__init__()
        self.Restormer_Encoder = Restormer_Encoder()
        self.restormer1 = Restormer(2 * mid_channel, decoder_num_heads, ffn_factor, bias, LayerNorm_type)
        self.restormer2 = Restormer(mid_channel, decoder_num_heads, ffn_factor, bias, LayerNorm_type)
        self.restormer3 = Restormer(mid_channel, decoder_num_heads, ffn_factor, bias, LayerNorm_type)
        self.conv1 = nn.Conv2d(2 * mid_channel, mid_channel, kernel_size=1)
        self.conv2 = nn.Conv2d(mid_channel, out_channel, kernel_size=1)
        self.softmax = nn.Sigmoid()
        self.Detail1 = nn.Sequential(
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.Detail2 = nn.Sequential(
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.Detail3 = nn.Sequential(
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.Base1 = TransformerBlock(dim=mid_channel, num_heads=decoder_num_heads, ffn_expansion_factor=2, bias=bias,
                                      LayerNorm_type=LayerNorm_type)
        self.Base2 = TransformerBlock(dim=mid_channel, num_heads=decoder_num_heads, ffn_expansion_factor=2, bias=bias,
                                      LayerNorm_type=LayerNorm_type)
        self.Base3 = TransformerBlock(dim=mid_channel, num_heads=decoder_num_heads, ffn_expansion_factor=2, bias=bias,
                                      LayerNorm_type=LayerNorm_type)
        self.decoder = nn.Sequential(
            nn.Conv2d(2 * mid_channel, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.BaseFuseLayer = BaseFuseLayer()
        self.DetailFuseLayer = DetailFeatureExtraction()
        self.CrossAttentionFusion = CrossAttentionFusion(input_channel=mid_channel, image2text_dim=32)
    def forward(self, imageA, imageB):
        Basic_Image_Feature_A, Detail_Image_Feature_A = self.Restormer_Encoder(imageA)  # (B,C,H,W)
        x1 = Basic_Image_Feature_A
        x2 = Detail_Image_Feature_A
        Basic_Image_Feature_B, Detail_Image_Feature_B = self.Restormer_Encoder(imageB)
        x3 = Basic_Image_Feature_B
        x4 = Detail_Image_Feature_B
        Basic_Feature_A = self.Base1(Basic_Image_Feature_A)
        x5 = Basic_Feature_A
        Basic_Feature_A = self.Base2(Basic_Feature_A)
        x6 = Basic_Feature_A
        Basic_Feature_A = self.Base3(Basic_Feature_A)
        x7 = Basic_Feature_A
        Detail_Feature_A = self.Detail1(Detail_Image_Feature_A)
        x8 = Detail_Feature_A
        Detail_Feature_A = self.Detail2(Detail_Feature_A)
        x9 = Detail_Feature_A
        Detail_Feature_A = self.Detail3(Detail_Feature_A)
        x10 = Detail_Feature_A
        Basic_Feature_B = self.Base1(Basic_Image_Feature_B)
        x11 = Basic_Feature_B
        Basic_Feature_B = self.Base2(Basic_Feature_B)
        x12 = Basic_Feature_B
        Basic_Feature_B = self.Base3(Basic_Feature_B)
        x13 = Basic_Feature_B
        Detail_Feature_B = self.Detail1(Detail_Image_Feature_B)
        x14 = Detail_Feature_B
        Detail_Feature_B = self.Detail2(Detail_Feature_B)
        x15 = Detail_Feature_B
        Detail_Feature_B = self.Detail3(Detail_Feature_B)
        x16 = Detail_Feature_B


        Basic_Feature = Basic_Feature_A + Basic_Feature_B
        Detail_Feature = Detail_Feature_A + Detail_Feature_B

        Basic_Feature = self.BaseFuseLayer(Basic_Feature)
        Detail_Feature = self.DetailFuseLayer(Detail_Feature)
        Fuse_Feature_A = self.CrossAttentionFusion(Basic_Feature, Detail_Feature)
        x17 = Fuse_Feature_A
        Fuse_Feature_B = self.CrossAttentionFusion(Detail_Feature, Basic_Feature)
        x18 = Fuse_Feature_B

        fusionfeature = torch.cat((Fuse_Feature_A, Fuse_Feature_B), dim=1)
        fusionfeature = self.restormer1(fusionfeature)
        fusionfeature = self.conv1(fusionfeature)
        fusionfeature = self.restormer2(fusionfeature)
        fusionfeature = self.restormer3(fusionfeature)
        fusionfeature = self.conv2(fusionfeature)
        Fuse_image = self.softmax(fusionfeature)
        intermediate_outputs = (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18)
        return Fuse_image, Basic_Feature_A, Basic_Feature_B, Detail_Feature_A, Detail_Feature_B, intermediate_outputs