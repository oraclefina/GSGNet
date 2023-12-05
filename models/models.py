import torch 
import torch.nn as nn
import timm
from models.module import *
from models.ops import *
from fvcore.nn import FlopCountAnalysis, flop_count_table


class decoder(nn.Module):
  def __init__(self,backbone_dim=[2208,2112,768,384,96],embed_dim=[192,192,192,192]) -> None:
    super().__init__()
    # top layer
    self.toplayer = nn.Conv2d(backbone_dim[0],embed_dim[0],1,1,0)
    # lateral layers
    self.latlayer1 = nn.Conv2d(backbone_dim[1],embed_dim[1],1,1,0)
    self.latlayer2 = nn.Conv2d(backbone_dim[2],embed_dim[2],1,1,0)
    self.latlayer3 = nn.Conv2d(backbone_dim[3],embed_dim[3],1,1,0)
    # upsample
    self.up = BilinearUpsample(2)
    self.up4 = BilinearUpsample(4)
    self.up8 = BilinearUpsample(8)
    # feature enhance and fuse modules
    self.csb = ContextSelfBlock(dim=embed_dim[0])
    self.lgfb_1 = LocalGlobalFusionBlock(dim_q=embed_dim[1]//2,dim_kv=embed_dim[1],factor=2)
    self.lgfb_2 = LocalGlobalFusionBlock(dim_q=embed_dim[2]//2,dim_kv=embed_dim[2],factor=4)
    self.lgfb_3 = LocalGlobalFusionBlockL(dim_q=embed_dim[3]//2,dim_kv=embed_dim[3],factor=2,scale_factor=0.25)


    #smooth layer
    self.smooth0 = ConvBNRelu(embed_dim[0],128,3,1,1)
    self.smooth1 = ConvBNRelu(embed_dim[1],128,3,1,1)
    self.smooth2 = ConvBNRelu(embed_dim[2],128,3,1,1)
    self.smooth3 = ConvBNRelu(embed_dim[3],128,3,1,1)
    # read out
    self.Conv7 = nn.Sequential(
        nn.Conv2d(4*128,128,1,1,0),
        nn.Conv2d(128,128,7,1,3,groups=128),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128,64,1,1,0),
        nn.Conv2d(64,64,7,1,3,groups=64),
        nn.BatchNorm2d(64),
        nn.ReLU(),
    )
    self.out = nn.Sequential(
        nn.ConvTranspose2d(64,64,kernel_size=4,stride=2,padding=1,groups=64),
        ConvBNRelu(64,64,3,1,1),
        nn.ConvTranspose2d(64,64,kernel_size=4,stride=2,padding=1,groups=64),
        nn.Conv2d(64,1,3,1,1),
        nn.Sigmoid(),
    )

    self.sideout1 = nn.Sequential(
       ConvBNRelu(128,64,3,1,1),
       ConvBNRelu(64,32,3,1,1),
       nn.Conv2d(32,1,3,1,1),
     )
    self.sideout2 = nn.Sequential(
       ConvBNRelu(128,64,3,1,1),
       ConvBNRelu(64,32,3,1,1),
       nn.Conv2d(32,1,3,1,1),
    )

  def forward(self,o1,o2,o3,o4):
    # transition 
    o1 = self.toplayer(o1)
    o2 = self.latlayer1(o2)
    o3 = self.latlayer2(o3)
    o4 = self.latlayer3(o4)
    # top down
    x1 = self.csb(o1)
    x2 = self.lgfb_1(o2,x1)
    x3 = self.lgfb_2(o3,x1)
    x4 = self.lgfb_3(o4,x1)
    # smooth
    x1 = self.smooth0(x1)
    x2 = self.smooth1(x2)
    x3 = self.smooth2(x3)
    x4 = self.smooth3(x4)
    x = torch.cat([self.up8(x1),self.up4(x2),self.up(x3),x4],dim=1)
    # readout
    x = self.Conv7(x)
    x = self.out(x)
    x = x.squeeze(1)

    supervised_1 = self.sideout1(x3).squeeze(1)
    supervised_2 = self.sideout2(x4).squeeze(1)
    
    return x,supervised_1,supervised_2

class GSGNet(nn.Module):
  def __init__(self,backbone_dim=[2208,2112,768,384,96],embed_dim=[192,192,192,192],backbone_type='densenet161'):
    super().__init__()
    # backbone
    self.backbone = timm.create_model(backbone_type,pretrained=True,features_only=True)
    # top layer
    self.decoder = decoder(backbone_dim,embed_dim)
    
  def forward(self,input):
    # encoder
    o = self.backbone(input)
    o1,o2,o3,o4 = o[-1],o[-2],o[-3],o[-4]
    # transition 
    x,supervised_1,supervised_2 = self.decoder(o1,o2,o3,o4)
    
    return x,supervised_1,supervised_2

class GSGNet_T(nn.Module):
  def __init__(self,backbone_dim=[2208,2112,768,384,96],embed_dim=[192,192,192,192],setting=0.25,backbone_type='densenet161'):
    super().__init__()
    # backbone
    self.backbone = timm.create_model(backbone_type,pretrained=True,features_only=True)
    # top layer
    self.toplayer = nn.Conv2d(backbone_dim[0],embed_dim[0],1,1,0)
    # lateral layers
    self.latlayer1 = nn.Conv2d(backbone_dim[1],embed_dim[1],1,1,0)
    self.latlayer2 = nn.Conv2d(backbone_dim[2],embed_dim[2],1,1,0)
    self.latlayer3 = nn.Conv2d(backbone_dim[3],embed_dim[3],1,1,0)
    # upsample
    self.up = BilinearUpsample(2)
    self.up4 = BilinearUpsample(4)
    self.up8 = BilinearUpsample(8)
    # feature enhance and fuse modules
    self.csb = ContextSelfBlock(dim=embed_dim[0],keep_ratio=setting, n_heads=4)
    self.lgfb_1 = LocalGlobalFusionBlock(dim_q=embed_dim[1]//2,dim_kv=embed_dim[1],factor=2,keep_ratio=setting,n_heads=4)
    self.lgfb_2 = LocalGlobalFusionBlock(dim_q=embed_dim[2]//2,dim_kv=embed_dim[2],factor=4,keep_ratio=setting,n_heads=4)
    self.lgfb_3 = LocalGlobalFusionBlockL(dim_q=embed_dim[3]//2,dim_kv=embed_dim[3],factor=2,scale_factor=0.25,keep_ratio=setting,n_heads=4)


    #smooth layer
    self.smooth0 = ConvBNRelu(embed_dim[0],128,3,1,1)
    self.smooth1 = ConvBNRelu(embed_dim[1],128,3,1,1)
    self.smooth2 = ConvBNRelu(embed_dim[2],128,3,1,1)
    self.smooth3 = ConvBNRelu(embed_dim[3],128,3,1,1)
    # read out
    self.Conv7 = nn.Sequential(
        nn.Conv2d(4*128,128,1,1,0),
        nn.Conv2d(128,128,7,1,3,groups=128),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128,64,1,1,0),
        nn.Conv2d(64,64,7,1,3,groups=64),
        nn.BatchNorm2d(64),
        nn.ReLU(),
    )
    self.out = nn.Sequential(
        nn.ConvTranspose2d(64,64,kernel_size=4,stride=2,padding=1,groups=64),
        ConvBNRelu(64,64,3,1,1),
        nn.ConvTranspose2d(64,64,kernel_size=4,stride=2,padding=1,groups=64),
        nn.Conv2d(64,1,3,1,1),
        nn.Sigmoid(),
    )

    self.sideout1 = nn.Sequential(
       ConvBNRelu(128,64,3,1,1),
       ConvBNRelu(64,32,3,1,1),
       nn.Conv2d(32,1,3,1,1),
     )
    self.sideout2 = nn.Sequential(
       ConvBNRelu(128,64,3,1,1),
       ConvBNRelu(64,32,3,1,1),
       nn.Conv2d(32,1,3,1,1),
    )

    
  def forward(self,input):
    # encoder
    o = self.backbone(input)
    o1,o2,o3,o4 = o[-1],o[-2],o[-3],o[-4]
    # transition 
    o1 = self.toplayer(o1)
    o2 = self.latlayer1(o2)
    o3 = self.latlayer2(o3)
    o4 = self.latlayer3(o4)
    # top down
    x1 = self.csb(o1)
    x2 = self.lgfb_1(o2,x1)
    x3 = self.lgfb_2(o3,x1)
    x4 = self.lgfb_3(o4,x1)
    # smooth
    x1 = self.smooth0(x1)
    x2 = self.smooth1(x2)
    x3 = self.smooth2(x3)
    x4 = self.smooth3(x4)
    x = torch.cat([self.up8(x1),self.up4(x2),self.up(x3),x4],dim=1)
    # readout
    x = self.Conv7(x)
    x = self.out(x)
    x = x.squeeze(1)

    supervised_1 = self.sideout1(x3).squeeze(1)
    supervised_2 = self.sideout2(x4).squeeze(1)
    
    return x,supervised_1,supervised_2
  

if __name__ == "__main__":
  model = GSGNet()
  dummy_input = torch.randn(1,3,352,352)
  print(flop_count_table(FlopCountAnalysis(model,dummy_input)))
