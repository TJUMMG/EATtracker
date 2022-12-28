import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2
import pytorch_ssim
import pytorch_iou

ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)
#
class OceanPlus_(nn.Module):
    def __init__(self):
        super(OceanPlus_, self).__init__()
        self.features = None
        self.connect_model = None
        self.mask_model = None
        self.zf = None
        self.criterion = nn.BCEWithLogitsLoss()
        self.neck = None
        self.search_size = 255
        self.score_size = 25
        self.batch = 32 if self.training else 1
        self.lambda_u = 0.1
        self.lambda_s = 0.2

        # self.grids()

    def feature_extractor(self, x, online=False):
        return self.features(x, online=online)

    def extract_for_online(self, x):
        xf = self.feature_extractor(x, online=True)
        return xf

    def connector(self, template_feature, search_feature):
        pred_score = self.connect_model(template_feature, search_feature)
        return pred_score

    def update_roi_template(self, target_pos, target_sz,
                            score):  ##target_pos-当前帧预测的目标物体与255*255中心之间的偏移量，target_sz-上一帧目标物体的大小，score-当前帧预测的socer map上面的最大值
        """
        :param target_pos:  pos in search (not the original)
        :param target_sz:  size in target size
        :param score:
        :return:
        """

        lambda_u = self.lambda_u * float(score)
        lambda_s = self.lambda_s
        N, C, H, W = self.search_size
        stride = 8
        assert N == 1, "not supported"
        l = W // 2
        x = range(-l, l + 1)
        y = range(-l, l + 1)

        hc_z = (target_sz[1] + 0.3 * sum(target_sz)) / stride  ##换算到25*25的score map上面的尺度
        wc_z = (target_sz[0] + 0.3 * sum(target_sz)) / stride
        grid_x = np.linspace(- wc_z / 2, wc_z / 2, 17)
        grid_y = np.linspace(- hc_z / 2, hc_z / 2, 17)
        grid_x = grid_x[5:-5] + target_pos[0] / stride
        grid_y = grid_y[5:-5] + target_pos[1] / stride
        x_offset = grid_x / l
        y_offset = grid_y / l

        grid = np.reshape(np.transpose([np.tile(x_offset, len(y_offset)), np.repeat(y_offset, len(x_offset))]),
                          (len(grid_y), len(grid_x), 2))
        grid = torch.from_numpy(grid).unsqueeze(0).cuda()  ##(1,7,7,2)

        zmap = nn.functional.grid_sample(self.xf.double(), grid).float()  ##双线性插值的图像大小变换(貌似是一种warp操作)，(1,256,7,7)
        # cls_kernel = self.rpn.cls.make_kernel(zmap)
        self.MA_kernel = (1 - lambda_u) * self.MA_kernel + lambda_u * zmap  ##(1,256,7,7),MA_kernel-初始模板帧的特征
        self.zf_update = self.zf * lambda_s + self.MA_kernel * (1.0 - lambda_s)  ##一种对模板的线性更新策略

    def template(self, z, template_mask=None):  ##(1,3,127,127), (1,127,127)
        _, self.zf = self.feature_extractor(z)  ##backbone提取特征(1,1024,15,15)

        if self.neck is not None:  ##adjustlayer调整通道数
            self.zf_ori, self.zf = self.neck(self.zf, crop=True)  ##(1,256,15,15), (1,256,7,7)

        self.template_mask = template_mask.float()
        self.MA_kernel = self.zf.detach()
        self.zf_update = None

    def track(self, x):  ##(1,3,255,255)

        features_stages, xf = self.feature_extractor(
            x)  ##搜索帧经过backbone得到四个阶段的输出 b1-(1,64,125,125), b2-(1,256,63,63), b3-(1,512,31,31)  xf-(1,1024,31,31)

        if self.neck is not None:
            xf = self.neck(xf, crop=False)  ##搜索帧经过调整层，降通道-(1,256,31,31)

        features_stages.append(xf)
        bbox_pred, cls_pred, cls_feature, reg_feature = self.connect_model(xf, self.zf,
                                                                           update=self.zf_update)  ##(1,4,25,25), (1,1,25,25), (1,256,25,25), (1,256,25,25)

        features_stages.append(cls_feature)
        pred_edge_set, pred_mask_set= self.mask_model(features_stages, input_size=x.size()[2:], zf_ori=self.zf_ori,
                                       template_mask=self.template_mask)  ##(1,2,255,255) 由三个阶段预测的mask分别*0.33加权而得
        self.search_size = xf.size()  ##(1,256,31,31)
        self.xf = xf.detach()

        return cls_pred, bbox_pred, pred_mask_set[-1]

    ##wyt
    def forward(self, template, search, tem_mask, sea_mask, sea_contour, sea_edge, sea_edge_w):
        _, zf = self.feature_extractor(template)
        features_stages, xf = self.feature_extractor(search)

        if self.neck is not None:
            zf_or, zf = self.neck(zf, crop=True)
            xf = self.neck(xf, crop=False)

        features_stages.append(xf)
        bbox_pred, cls_pred, cls_feature, reg_feature = self.connect_model(xf,
                                                                           zf)  ##(1,4,25,25), (1,1,25,25), (1,256,25,25), (1,256,25,25)

        features_stages.append(cls_feature)
        pred_edge_set, pred_mask_set = self.mask_model(features_stages, input_size=search.size()[2:], zf_ori=zf_or,
                                                   template_mask=tem_mask)  ##(1,2,255,255) 由三个阶段预测的mask分别*0.33加权而

        sea_mask = sea_mask.unsqueeze(dim=1)
        sea_edge = sea_edge.unsqueeze(dim=1)
        sea_edge_w = sea_edge_w.unsqueeze(dim=1)

        # ##将生成的mask以及gt进行上菜样到原来的2倍
        # for i in range(len(pred_mask_set)):
        #     pred_mask_set[i] = F.interpolate(pred_mask_set[i], 255*2, mode='bilinear', align_corners=True)
        # sea_mask = F.interpolate(sea_mask, 255*2, mode='bilinear', align_corners=True)


        edge_loss, mask_loss = self.BCEL_loss(pred_edge_set, pred_mask_set, sea_edge, sea_edge_w, sea_mask)

        IOU_SSIM_loss = 0
        for i in range(len(pred_mask_set)):
            pred_mask = F.sigmoid(pred_mask_set[i])
            IOU_SSIM_loss += self.IOU_SSIM(pred_mask, sea_mask)

        # for i in range(3):
        #     pred_mask = F.sigmoid(pred_mask_set[i])
        #     IOU_SSIM_loss += self.IOU_SSIM(pred_mask, sea_mask)
        # pred_mask = F.sigmoid(pred_mask_set[-1])
        # IOU_SSIM_loss += self.IOU_SSIM(pred_mask, sea_mask)


        return edge_loss, mask_loss+IOU_SSIM_loss
        # return edge_loss, mask_loss



    def BCEL_loss(self,pred_edge_set, pred_mask_set, gt_edge, edge_weight, gt_mask):
        ##edge loss
        edge_loss = 0
        ##mask loss
        mask_loss = 0

        # ##新增预测mask生成的edge的loss
        # mask_edge_loss = 0
        for j in range(len(pred_edge_set)):
            edge_loss += F.binary_cross_entropy_with_logits(pred_edge_set[j], gt_edge, edge_weight)

        for i in range(len(pred_mask_set)):
            mask_loss += F.binary_cross_entropy_with_logits(pred_mask_set[i], gt_mask)
            # mask_edge_loss += F.binary_cross_entropy_with_logits(pred_mask_set[i]*gt_edge, gt_edge, edge_weight)

        return edge_loss, mask_loss

    def IOU_SSIM(self,pred,target):
        ssim_out = 1 - ssim_loss(pred, target)
        iou_out = iou_loss(pred, target)

        loss = ssim_out + iou_out

        return loss














