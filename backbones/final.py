import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from typing import Optional, Callable
import numpy as np

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x




class FourierTransformLayer(nn.Module):
    """傅里叶变换增强层
    
    用于在Transformer架构前对图像进行傅里叶变换增强，通过掩码增强高频信息
    提升模型对细节特征的捕捉能力
    """
    def __init__(self, radius_ratio, smoothness, img_size):
        """初始化傅里叶变换增强层
        
        Args:
            radius_ratio: 半径比率，控制掩码中心区域大小
            smoothness: 平滑系数，控制掩码的平滑过渡
            img_size: 输入图像尺寸
        """
        super(FourierTransformLayer, self).__init__()
        self.radius_ratio = radius_ratio  # 半径比率
        self.smoothness = smoothness      # 平滑系数
        # 可学习参数，用于调整不同通道的增强效果
        self.learnable_param = nn.Parameter(torch.ones(3, img_size, img_size))

    def create_smooth_mask(self, h, w):
        """创建平滑掩码
        
        生成一个中心平滑、边缘逐渐增强的掩码，用于增强傅里叶域中的高频信息
        
        Args:
            h: 图像高度
            w: 图像宽度
            
        Returns:
            平滑掩码，形状为 [3, h, w]
        """
        # 初始化全1掩码
        mask = torch.ones((h, w), dtype=torch.float32).cuda()
        # 计算图像中心点
        center_h, center_w = h // 2, w // 2
        # 计算掩码中心区域半径
        radius_h, radius_w = int(h * self.radius_ratio), int(w * self.radius_ratio)

        # 遍历图像每个像素点
        for i in range(h):
            for j in range(w):
                # 计算当前点到中心点的距离
                dist_h = abs(i - center_h)
                dist_w = abs(j - center_w)

                # 中心点附近使用平滑系数
                if dist_h**2 + dist_w**2 <= 1:
                    mask[i, j] = self.smoothness
                else:
                    # 归一化距离到[0,1]范围
                    dist_h_norm = dist_h / (h / 2)
                    dist_w_norm = dist_w / (w / 2)
                    # 计算欧几里得距离
                    distance = np.sqrt(dist_h_norm ** 2 + dist_w_norm ** 2)
                    # 边缘区域逐渐增强，确保最小值为1.0
                    mask[i, j] = max(1.0, distance/self.smoothness)

        # 扩展掩码到3通道
        return mask.unsqueeze(0).expand(3, -1, -1)

    def forward(self, x):
        """前向传播
        
        对输入图像进行傅里叶变换，应用掩码增强高频信息，然后进行逆变换重建图像
        
        Args:
            x: 输入图像张量，形状为 [B, C, H, W]
            
        Returns:
            增强后的图像张量，形状为 [B, C, H, W]
        """
        # 1. 对图像进行傅里叶变换
        img_fft = torch.fft.fft2(x)                # 二维傅里叶变换
        img_fft_shifted = torch.fft.fftshift(img_fft).cuda()  # 频域中心化

        # 2. 应用平滑掩码增强高频信息
        h, w = img_fft_shifted.shape[-2:]          # 获取图像尺寸
        smooth_mask = self.create_smooth_mask(h, w)  # 创建平滑掩码
        # 应用掩码和可学习参数
        img_filtered = img_fft_shifted * smooth_mask * self.learnable_param
        
        # 3. 逆傅里叶变换重建图像
        img_enhanced = torch.fft.ifftshift(img_filtered)  # 频域去中心化
        img_reconstructed = torch.fft.ifft2(img_enhanced).real  # 逆傅里叶变换，取实部
        
        return img_reconstructed

 
class Attention(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 qkv_bias: bool = False,
                 qk_scale: Optional[None] = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        
        with torch.cuda.amp.autocast(True):
            batch_size, num_token, embed_dim = x.shape
            #qkv is [3,batch_size,num_heads,num_token, embed_dim//num_heads]
            qkv = self.qkv(x).reshape(
                batch_size, num_token, 3, self.num_heads, embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        with torch.cuda.amp.autocast(False):
            q, k, v = qkv[0].float(), qkv[1].float(), qkv[2].float()
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(batch_size, num_token, embed_dim)
        with torch.cuda.amp.autocast(True):
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 num_patches: int,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_scale: Optional[None] = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 act_layer: Callable = nn.ReLU6,
                 norm_layer: str = "ln", 
                 patch_n: int = 144):
        super().__init__()

        if norm_layer == "bn":
            self.norm1 = VITBatchNorm(num_features=num_patches)
            self.norm2 = VITBatchNorm(num_features=num_patches)
        elif norm_layer == "ln":
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.extra_gflops = (num_heads * patch_n * (dim//num_heads)*patch_n * 2) / (1000**3)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        with torch.cuda.amp.autocast(True):
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=108, patch_size=9, in_channels=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        assert height == self.img_size[0] and width == self.img_size[1], \
            f"Input image size ({height}*{width}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size: int = 112,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 num_classes: int = 1000,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_scale: Optional[None] = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 hybrid_backbone: Optional[None] = None,
                 norm_layer: str = "ln",
                 mask_ratio = 0.1,
                 using_checkpoint = False,
                 ):
        super().__init__()
        self.num_classes = num_classes
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim

        if hybrid_backbone is not None:
            raise ValueError
        else:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        self.mask_ratio = mask_ratio
        self.using_checkpoint = using_checkpoint
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        patch_n = (img_size//patch_size)**2
        self.blocks = nn.ModuleList(
            [
                Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                      num_patches=num_patches, patch_n=patch_n)
                for i in range(depth)]
        )
        self.extra_gflops = 0.0
        for _block in self.blocks:
            self.extra_gflops += _block.extra_gflops

        if norm_layer == "ln":
            self.norm = nn.LayerNorm(embed_dim)
        elif norm_layer == "bn":
            self.norm = VITBatchNorm(self.num_patches)

        # features head
        self.feature = nn.Sequential(
            nn.Linear(in_features=embed_dim * num_patches, out_features=embed_dim, bias=False),
            nn.BatchNorm1d(num_features=embed_dim, eps=2e-5),
            nn.Linear(in_features=embed_dim, out_features=num_classes, bias=False),
            nn.BatchNorm1d(num_features=num_classes, eps=2e-5)
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.fourier_transform_layer = FourierTransformLayer(0.9, 0.8, img_size)  

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head
    
    def random_masking(self, x, mask_ratio=0.1):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.size()  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_features(self, x):
        #if self.training:
        x = self.fourier_transform_layer(x)  
        
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        if self.training and self.mask_ratio > 0:
            x, _, ids_restore = self.random_masking(x)


        for func in self.blocks:
            if self.using_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(func, x)
            else:
                x = func(x)
        x = self.norm(x.float())
        
        if self.training and self.mask_ratio > 0:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
            x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = x_
        return torch.reshape(x, (B, self.num_patches * self.embed_dim))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.feature(x)
        return x