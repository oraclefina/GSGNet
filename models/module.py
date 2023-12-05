import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath
from models.ops import *


class ContextSelfBlock(nn.Module):
    def __init__(self,dim, spatial_size=(3,3), n_heads=4, depth=None, qk_scale=None, qkv_bias=False, k=7, \
            keep_ratio=0.25,attn_drop=0., proj_drop=0., proj=False, reduction=['bilinear','avg'], expansion_factor=4., mlp_drop=0.) -> None:
        super().__init__()
        self.pe = DWConvPE(dim)
        self.norm1 = LayerNorm2d(dim)
        self.norm2 = LayerNorm2d(dim)

        self.attn1 = SpatialReductionCSSA(dim, spatial_size, n_heads, depth, qk_scale, qkv_bias, k, \
            attn_drop, proj_drop, proj, reduction[0],keep_ratio)
        self.attn2 = SpatialReductionCSSA(dim, spatial_size, n_heads, depth, qk_scale, qkv_bias, k, \
            attn_drop, proj_drop, proj, reduction[1],keep_ratio)
        
        self.norm_trans = LayerNorm2d(dim*2)
        self.transition = nn.Conv2d(dim*2,dim,1)
        self.norm = LayerNorm2d(dim)
        self.ffn = ConvFFN(dim=dim,expansion_factor=expansion_factor,mlp_drop=mlp_drop)
    def forward(self,x):
        # add PE
        x = x + self.pe(x)
        # inception-like
        sa1 = x + self.attn1(self.norm1(x))
        sa2 = x + self.attn2(self.norm2(x))

        sa = torch.cat([sa1,sa2],dim=1)
        sa = self.norm_trans(sa)
        sa = self.transition(sa)
        out = sa + self.ffn(self.norm(sa))

        return out


class LocalGlobalFusionBlock(nn.Module):
    '''LGFB'''
    def __init__(self,dim_q, dim_kv, factor, n_heads=4, depth=None, qk_scale=None, qkv_bias=False, k=7, \
            attn_drop=0., proj_drop=0., proj=False, drop_path=0.,keep_ratio=0.25):
        super().__init__()
        self.attn = CrossCSSA(dim_q, dim_kv, n_heads, depth, qk_scale, qkv_bias, k, \
            attn_drop, proj_drop, proj,keep_ratio)
        

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm = LayerNorm2d(dim_q*2)
        self.ffn = ConvFFN(dim=dim_q*2)

        self.sw = SpatialWeightedConcat(dim_q,dim_kv,factor,kernel_size=(k,k))
        self.to_out = nn.Conv2d(dim_q*2,dim_q*2,1,1)

    def forward(self,xl,xh):
        # channel split
        x1 = xl[:,xl.size()[1]//2:,:,:]
        x2 = xl[:,:xl.size()[1]//2,:,:]
        # local branch 
        x2 = self.sw(x2,xh)
        # global branch 
        x1 = self.attn(x1,xh)
        # concat
        x = torch.cat([x1,x2],dim=1)
        x = x + self.drop_path(self.ffn(self.norm(x)))
        out = self.to_out(x)

        return out
    

class LocalGlobalFusionBlockL(nn.Module):
    '''LGFB'''
    def __init__(self,dim_q, dim_kv, factor, n_heads=4, depth=None, qk_scale=None, qkv_bias=False, k=7, \
            attn_drop=0., proj_drop=0., proj=False, drop_path=0.,scale_factor=0.5,keep_ratio=0.25):
        super().__init__()

        self.pool = nn.Upsample(scale_factor=scale_factor,mode='bilinear',align_corners=False)

        self.attn = CrossCSSA(dim_q, dim_kv, n_heads, depth, qk_scale, qkv_bias, k, \
            attn_drop, proj_drop, proj,keep_ratio)
        

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm = LayerNorm2d(dim_q*2)
        self.ffn = ConvFFN(dim=dim_q*2)

        self.sw = SpatialWeightedConcat(dim_q,dim_kv,factor,kernel_size=(k,k))
        self.upconv = nn.Sequential(
            BilinearUpsample(up_scale=int(1/scale_factor)),
            nn.Conv2d(dim_q*2,dim_q*2,3,1,1),
        )

    def forward(self,xl,xh):
        xl = self.pool(xl)
        # channel split
        x1 = xl[:,xl.size()[1]//2:,:,:]
        x2 = xl[:,:xl.size()[1]//2,:,:]
        # local branch 
        x2 = self.sw(x2,xh)
        # global branch 
        x1 = self.attn(x1,xh)
        # concat
        x = torch.cat([x1,x2],dim=1)
        x = x + self.drop_path(self.ffn(self.norm(x)))
        # up
        out = self.upconv(x)

        return out
    
