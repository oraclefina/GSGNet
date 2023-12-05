import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from timm.models.layers import trunc_normal_

class BilinearUpsample(nn.Module):
    def __init__(self,up_scale,align_corners=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=up_scale,mode='bilinear',align_corners=align_corners)
    def forward(self,x):
        return self.up(x)

class ConvBNRelu(nn.Module):
    def __init__(self,in_channels,out_channels,kernel,stride,padding) -> None:
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                kernel_size=kernel,stride=stride,padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DWConvPE(nn.Module):
    def __init__(self,in_channels,kernel_size=7,stride=1,padding=3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,groups=in_channels)
    def forward(self,x):
        return self.conv(x)
    

    
class CompressChannel3d(nn.Module):
    def __init__(self,n_heads, head_dim, compress_ratio=1, k=7) -> None:
        super().__init__()
        assert compress_ratio >= 0  and compress_ratio <= 1.0

        self.n_heads = n_heads
        self.padding = (k - 1) // 2
 
        c_g = np.floor(compress_ratio * head_dim+1)
        c_g = int(np.clip(c_g,1,head_dim))
        self.embedding = nn.Parameter(torch.randn([c_g,head_dim,3,k,k]),requires_grad=True)
        trunc_normal_(self.embedding,std=0.02)
            
    
    def forward(self,x):
        x = rearrange(x, 'b nh d h w-> b d nh h w')
        x = F.conv3d(x,self.embedding,stride=(1,1,1),padding=(1,self.padding,self.padding))
        x = rearrange(x, 'b d nh h w-> b nh d h w')
        return x

class LayerNorm2d(nn.Module):
    def __init__(self,dim,eps=1e-5) -> None:
        super(LayerNorm2d,self).__init__()
        self.norm = nn.LayerNorm(dim,eps=eps)

    def forward(self,x):
        x = x.permute(0,2,3,1)
        x = self.norm(x)
        x = x.permute(0,3,1,2)
        return x
    
class AdaptiveDownsample2d(nn.Module):
    def __init__(self,output_size,pool_type='bilinear') -> None:
        super().__init__()
        if pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(output_size=output_size)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(output_size=output_size)
        else:
            self.pool = nn.Upsample(size=output_size,mode='bilinear',align_corners=False)
    
    def forward(self,x):
        return self.pool(x)
    
class ConvFFN(nn.Module):
    def __init__(self,dim,dim_out=None,hidden_dim=None,expansion_factor=4.,mlp_drop=0.):
        super().__init__()
        hidden_dim = int(expansion_factor * dim) if hidden_dim is None else hidden_dim
        dim_out = dim if dim_out is None else dim_out
        self.net = nn.Sequential(
            nn.Conv2d(dim,hidden_dim,1,1),
            nn.GELU(),
            nn.Conv2d(hidden_dim,dim_out,1,1),
            nn.Dropout(mlp_drop)
        )
    def forward(self,x):
        return self.net(x)
    
class ChannelSqueezeSpatialAttention(nn.Module):
    def __init__(self,dim_q, dim_kv=None, n_heads=4, depth=None, qk_scale=None, qkv_bias=False, k=7, \
            attn_drop=0., proj_drop=0., proj=False, keep_ratio=0.25):
        super().__init__()
        depth = dim_q if depth is None else depth
        self.n_heads = n_heads
        head_dim = depth // n_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.padding = (k-1)//2

        # projection
        self.query = nn.Conv2d(dim_q,depth,1,bias=qkv_bias)
        self.key = nn.Conv2d(dim_kv,depth,1,bias=qkv_bias)
        self.value = nn.Conv2d(dim_kv,depth,1, bias=qkv_bias)
        self.proj = nn.Conv2d(depth,dim_q,1) if proj else nn.Identity()
        # dropout
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # 3D Conv
        self.embedding_q = CompressChannel3d(n_heads=n_heads, head_dim=head_dim, compress_ratio=keep_ratio)
        self.embedding_k = CompressChannel3d(n_heads=n_heads, head_dim=head_dim, compress_ratio=keep_ratio)

    def forward(self,xq,xkv):
        B,C,H,W = xq.size()
        _,c,h,w = xkv.size()

        # projection
        q = self.query(xq)
        k = self.key(xkv)
        v = self.value(xkv)

        # Reshape
        q = rearrange(q, 'b (nh d) h w -> b nh d h w', nh=self.n_heads)
        k = rearrange(k, 'b (nh d) h w -> b nh d h w', nh=self.n_heads)
        v = rearrange(v, 'b (nh d) h w -> b nh (h w) d', nh=self.n_heads)

        q = self.embedding_q(q)
        q = rearrange(q,'b nh d h w -> b nh (h w) d')
        k = self.embedding_k(k)
        k = rearrange(k,'b nh d h w -> b nh d (h w)')

        # attention operation
        sim = (q @ k) * self.scale
        sim = sim.softmax(dim=-1)
        sim = self.attn_drop(sim)
        
        rec = sim @ v
        rec = rearrange(rec, 'b nh (h w) d -> b (nh d) h w', h=H, w=W)
        rec = self.proj(rec)
        rec = self.proj_drop(rec)

        return rec
       
        
class SpatialReductionCSSA(nn.Module):
    def __init__(self,dim, spatial_size, n_heads=4, depth=None, qk_scale=None, qkv_bias=False, k=7, \
            attn_drop=0., proj_drop=0., proj=False, reduction='bilinear',keep_ratio=0.25) -> None:
        super().__init__()
        self.cssa = ChannelSqueezeSpatialAttention(dim, dim, n_heads, depth, qk_scale, qkv_bias, k, \
            attn_drop, proj_drop, proj, keep_ratio)
        self.spatial_reduction = AdaptiveDownsample2d(output_size=spatial_size,pool_type=reduction)

    def forward(self,x):
        xq = x 
        xkv = self.spatial_reduction(x)
        attn = self.cssa(xq,xkv) 

        return attn
    
class CrossCSSA(nn.Module):
    def __init__(self,dim_q, dim_kv, n_heads=4, depth=None, qk_scale=None, qkv_bias=False, k=7, \
            attn_drop=0., proj_drop=0., proj=False,keep_ratio=0.25) -> None:
        super().__init__()
        self.pe_q = DWConvPE(dim_q)
        self.pe_kv = DWConvPE(dim_kv)
        self.norm_q = LayerNorm2d(dim_q)
        self.norm_kv = LayerNorm2d(dim_kv)

        self.cssa = ChannelSqueezeSpatialAttention(dim_q, dim_kv, n_heads, depth, qk_scale, qkv_bias, k, \
            attn_drop, proj_drop, proj,keep_ratio)

    def forward(self,xl,xh):
        xq = xl + self.pe_q(xl)
        xkv = xh + self.pe_kv(xh)
        attn = xq + self.cssa(self.norm_q(xq),self.norm_kv(xkv)) 

        return attn
        
class SemanticEmbeddingBranch(nn.Module):
    def __init__(self,dim,kernel_size=(7,7)):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.padding = (kernel_size[0]-1) // 2

        self.kernels = AdaptiveDownsample2d(output_size=kernel_size,pool_type='max')
        self.norm = LayerNorm2d(dim)

    def forward(self,x,xk):
        B,C,H,W = x.size()

        kernels = self.kernels(xk).view(B,C,-1)
        kernels = F.softmax(kernels,dim=-1).view(B,C,self.kernel_size[0],self.kernel_size[1])
        conv_out = F.conv2d(x.reshape(1,-1,H,W),kernels.reshape(-1,1,self.kernel_size[0],self.kernel_size[1]),
        padding=self.padding,groups=B*C)
        conv_out = conv_out.reshape(B,C,H,W)
        conv_out = F.relu(self.norm(conv_out))+x

        return conv_out

class SpatialWeightedConcat(nn.Module):
    def __init__(self,dim,dim_h,up_scale,kernel_size=(7,7)) -> None:
        super().__init__()
        self.up = BilinearUpsample(up_scale=up_scale) if up_scale > 1 else nn.Identity()
        self.conv_b1 = nn.Conv2d(dim_h,dim,1,1)
        self.kf = SemanticEmbeddingBranch(dim=dim,kernel_size=kernel_size)
        self.conv_b2 = nn.Conv2d(dim_h,1,3,1,1)
        self.norm = LayerNorm2d(dim=dim)
        self.to_out = nn.Conv2d(dim*2,dim,1,1)

    def forward(self,x,xh):
        _,_,H,W = x.size()
        # branch 1
        x_pool = self.conv_b1(xh)
        x1 = self.kf(x,x_pool)
        # branch 2
        s_mask = torch.sigmoid(self.conv_b2(xh))
        x2 = x * self.up(s_mask)
        x2 = self.norm(x2)

        x = torch.cat([x1,x2],dim=1)
        x = self.to_out(x)
        return x



    


    

