import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .connect import xcorr_depthwise


class ARN(nn.Module):
    """
    Attention Retrieval Network in Ocean+
    """

    def __init__(self, inchannels=256, outchannels=256):
        super(ARN, self).__init__()
        self.s_embed = nn.Conv2d(inchannels, outchannels, 1)  # embedding for search feature
        self.t_embed = nn.Conv2d(inchannels, outchannels, 1)  # embeeding for template feature

    def forward(self, xf, zf, zf_mask):
        # xf: [B, C, H, W]
        # zf: [B, C, H, W]
        # zf_mask: [B, H, W]
        # pdb.set_trace()
        xf = self.s_embed(xf)
        zf = self.t_embed(zf)

        B, C, Hx, Wx = xf.size()
        B, C, Hz, Wz = zf.size()

        xf = xf.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        xf = xf.view(B, -1, C)  # [B, H*W, C]
        zf = zf.view(B, C, -1)  # [B, C, H*W]

        att = torch.matmul(xf, zf)  # [HW, HW]
        att = att / math.sqrt(C)
        att = F.softmax(att, dim=-1)  # [HW, HW]
        zf_mask = nn.Upsample(size=(Hz, Wz), mode='bilinear', align_corners=True)(zf_mask.unsqueeze(1))
        # zf_mask = (zf_mask > 0.5).float()
        zf_mask = zf_mask.view(B, -1, 1)

        arn = torch.matmul(att, zf_mask)  # [B, H*W]
        arn = arn.view(B, Hx, Hx).unsqueeze(1)
        return arn

class Graph_Attention_Union(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Graph_Attention_Union, self).__init__()

        # search region nodes linear transformation
        self.support = nn.Conv2d(in_channel, in_channel, 1, 1)

        # target template nodes linear transformation
        self.query = nn.Conv2d(in_channel, in_channel, 1, 1)

        # linear transformation for message passing
        self.g = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )

        # aggregated feature
        self.fi = nn.Sequential(
            nn.Conv2d(in_channel*2, out_channel, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, zf, xf):
        # linear transformation
        xf_trans = self.query(xf)
        zf_trans = self.support(zf)

        # linear transformation for message passing
        xf_g = self.g(xf)
        zf_g = self.g(zf)

        # calculate similarity
        shape_x = xf_trans.shape
        shape_z = zf_trans.shape

        zf_trans_plain = zf_trans.view(-1, shape_z[1], shape_z[2] * shape_z[3])
        zf_g_plain = zf_g.view(-1, shape_z[1], shape_z[2] * shape_z[3]).permute(0, 2, 1)
        xf_trans_plain = xf_trans.view(-1, shape_x[1], shape_x[2] * shape_x[3]).permute(0, 2, 1)

        similar = torch.matmul(xf_trans_plain, zf_trans_plain)
        similar = F.softmax(similar, dim=2)

        embedding = torch.matmul(similar, zf_g_plain).permute(0, 2, 1)
        embedding = embedding.view(-1, shape_x[1], shape_x[2], shape_x[3])

        # aggregated feature
        output = torch.cat([embedding, xf_g], 1)
        output = self.fi(output)
        return output


class EGNet(nn.Module):
    def __init__(self):
        super(EGNet, self).__init__()
        # BACKBONE
        self.b1 = nn.Sequential(nn.Conv2d(64, 64, 1), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv2d(256, 64, 1), nn.ReLU())
        self.b3 = nn.Sequential(nn.Conv2d(512, 64, 1), nn.ReLU())
        self.b4 = nn.Sequential(nn.Conv2d(256, 64, 1), nn.ReLU())
        self.c = nn.Sequential(nn.Conv2d(256, 64, 1), nn.ReLU())

        # REFINE
        self.rb1 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.rb2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.rb3 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.rb4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())

        # multi output
        self.e1 = nn.Sequential(nn.Conv2d(64, 1, 3, padding=1))
        self.e2 = nn.Sequential(nn.Conv2d(64, 1, 3, padding=1))
        self.m2 = nn.Sequential(nn.Conv2d(64, 1, 3, padding=1))
        self.m3 = nn.Sequential(nn.Conv2d(64, 1, 3, padding=1))
        self.m4 = nn.Sequential(nn.Conv2d(64, 1, 3, padding=1))

        # multi edge feature and mask feature refine layer
        self.reb2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.eb2 = nn.Sequential(nn.Conv2d(64, 1, 3, padding=1))
        self.reb3 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.eb3 = nn.Sequential(nn.Conv2d(64, 1, 3, padding=1))
        self.reb4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.eb4 = nn.Sequential(nn.Conv2d(64, 1, 3, padding=1))

        # high featuer conduct edge
        self.trans = nn.Sequential(nn.Conv2d(64, 64, 1), nn.ReLU())

        # final refine
        self.f1 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.f2 = nn.Sequential(nn.Conv2d(64, 1, 3, padding=1))

        self.relu = nn.ReLU()

        # ## mish
        # self.ms1 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1))
        # self.ms2 = nn.Sequential(nn.Conv2d(64, 1, 3, padding=1))


        ##新增网络阶段，原；来EGNet上面的使用边缘特征输出mask的之路会先经过一个1*1的降通道
        self.transb2 = nn.Sequential(nn.Conv2d(64, 64, 1), nn.ReLU())
        self.transb3 = nn.Sequential(nn.Conv2d(64, 64, 1), nn.ReLU())
        self.transb4 = nn.Sequential(nn.Conv2d(64, 64, 1), nn.ReLU())
        ##-----

        # ##mask attention
        # self.mask_att = nn.Sequential(nn.Conv2d(1, 64, 1), nn.ReLU())
        # self.edge_att = nn.Sequential(nn.Conv2d(1, 64, 1), nn.ReLU())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)

    def forward(self, features, output_size=None):
        b1, b2, b3, b4, corr = features
        output_mask = []
        output_edge = []

        b4_size = b4.size()[2:]
        b3_size = b3.size()[2:]
        b2_size = b2.size()[2:]
        b1_size = b1.size()[2:]
        if output_size is None: output_size = (255, 255)

        # fuse
        corr = self.c(corr)
        corr = nn.Upsample(size=b4_size, mode='bilinear', align_corners=True)(corr)

        b4 = self.b4(b4)
        rb4 = self.rb4(b4+corr)

        b3 = self.b3(b3)
        rb3 = self.rb3(b3+rb4)

        b2 = self.b2(b2)
        rb2 = self.rb2(b2+nn.Upsample(size=b2_size, mode='bilinear', align_corners=True)(rb3))


        b1 = self.b1(b1)
        ef = self.rb1(b1+nn.Upsample(size=b1_size, mode='bilinear', align_corners=True)(self.trans(rb4)))


        # mask output
        edge1 = nn.Upsample(size=output_size, mode='bilinear', align_corners=True)(self.e1(ef))
        output_edge.append(edge1)
        edge2 = nn.Upsample(size=output_size, mode='bilinear', align_corners=True)(self.e2(rb4))
        output_edge.append(edge2)
        
        m2 = nn.Upsample(size=output_size, mode='bilinear', align_corners=True)(self.m2(rb2))

        output_mask.append(m2)
        m3 = nn.Upsample(size=output_size, mode='bilinear', align_corners=True)(self.m3(rb3))

        output_mask.append(m3)
        m4 = nn.Upsample(size=output_size, mode='bilinear', align_corners=True)(self.m4(rb4))

        output_mask.append(m4)

        # ##新增网络
        # rb2 = self.transb2(rb2)
        # rb3 = self.transb2(rb3)
        # rb4 = self.transb2(rb4)
        # ##---

        # edge feature mask output
        rb2 = nn.Upsample(size=b1_size, mode='bilinear', align_corners=True)(rb2)

        rb3 = nn.Upsample(size=b1_size, mode='bilinear', align_corners=True)(rb3)

        rb4 = nn.Upsample(size=b1_size, mode='bilinear', align_corners=True)(rb4)


        rem2 = self.reb2(rb2+ef)
        rem3 = self.reb3(rb3+ef)
        rem4 = self.reb4(rb4+ef)

        em2 = nn.Upsample(size=output_size, mode='bilinear', align_corners=True)(self.eb2(rem2))

        output_mask.append(em2)
        em3 = nn.Upsample(size=output_size, mode='bilinear', align_corners=True)(self.eb3(rem3))

        output_mask.append(em3)
        em4 = nn.Upsample(size=output_size, mode='bilinear', align_corners=True)(self.eb4(rem4))

        output_mask.append(em4)

        ## final mask
        mf = self.relu(self.relu(rem2+rem3)+rem4)
        # mf = self.relu(self.relu(rem3 + rem4) + rem2)
        # mf = rem2 + rem3 + rem4
        mf = self.f1(mf)
        mf = nn.Upsample(size=output_size, mode='bilinear', align_corners=True)(self.f2(mf))


        # ##funal mask use mish af
        # mf = self.mish(self.mish(rem2+rem3)+rem4)
        # mf = self.ms1(mf)
        # mf = self.mish(mf)
        # mf = nn.Upsample(size=output_size, mode='bilinear', align_corners=True)(self.ms2(mf))


        output_mask.append(mf)

        return output_edge, output_mask

    def mish(self,input):
        return input*torch.tanh(F.softplus(input))

class MMS(nn.Module):
    def __init__(self):
        super(MMS, self).__init__()

        self.sequential = ARN(256, 64)  # transduction attention
        # self.att = ARN(256, 64)
        self.egnet = EGNet()

    def forward(self, features, input_size=None, zf_ori=None, template_mask=None):
        b1, b2, b3, b4, corr = features

        arn = self.sequential(b4, zf_ori, template_mask)  # [B, H, W]
        # arn = self.att(b4, zf_ori, template_mask)
        arn = torch.clamp(arn, 0, 1)
        b4 = b4 + arn

        features = [b1, b2, b3, b4, corr]
        pred_edge, pred_mask_set = self.egnet(features)


        return pred_edge, pred_mask_set
