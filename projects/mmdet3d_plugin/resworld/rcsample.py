# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.checkpoint import checkpoint

from mmdet.models.backbones.resnet import BasicBlock
from mmdet3d.models.builder import NECKS
from time import time

class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target, weight=None, use_sigmoid=True):
        if use_sigmoid:
            pred_sigmoid = pred.sigmoid()
        else:
            pred_sigmoid = pred
        target = target.type_as(pred)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * pt.pow(self.gamma)
        if weight is not None:
            focal_weight = focal_weight * weight
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none') * focal_weight
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

class _ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):

    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(
            inplanes,
            mid_channels,
            1,
            padding=0,
            dilation=dilations[0],
            BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(
            int(mid_channels * 5), inplanes, 1, bias=False)
        self.bn1 = BatchNorm(inplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(
            x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

class DepthNet(nn.Module):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 context_channels,
                 depth_channels,
                 use_dcn=True,
                 use_aspp=True,
                 with_cp=False,
                 stereo=False,
                 bias=0.0,
                 aspp_mid_channels=-1):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(
            mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm1d(27)
        self.depth_mlp = Mlp(27, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        depth_conv_input_channels = mid_channels
        downsample = None

        depth_conv_list = [BasicBlock(depth_conv_input_channels, mid_channels,
                                      downsample=downsample),
                           BasicBlock(mid_channels, mid_channels),
                           BasicBlock(mid_channels, mid_channels)]
        if use_aspp:
            if aspp_mid_channels<0:
                aspp_mid_channels = mid_channels
            depth_conv_list.append(ASPP(mid_channels, aspp_mid_channels))
        if use_dcn:
            depth_conv_list.append(
                build_conv_layer(
                    cfg=dict(
                        type='DCN',
                        in_channels=mid_channels,
                        out_channels=mid_channels,
                        kernel_size=3,
                        padding=1,
                        groups=4,
                        im2col_step=128,
                    )))
        self.depth_conv = nn.Sequential(*depth_conv_list)
        self.depth_head = nn.Conv2d(mid_channels, depth_channels, kernel_size=1, stride=1, padding=0)
        self.with_cp = with_cp
        self.depth_channels = depth_channels

    def forward(self, x, mlp_input, stereo_metas=None):
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)

        if self.with_cp:
            depth_feat = checkpoint(self.depth_conv, depth)
        else:
            depth_feat = self.depth_conv(depth)
        depth = self.depth_head(depth_feat)
        return depth, context, depth_feat


@NECKS.register_module()
class RCSample(BaseModule):

    def __init__(self, grid_config, input_size, scale_num, ins_channels, 
                 out_channels, downsamples, accelerate=False,
                 loss_depth_weight=3.0, with_cp=False, depthnet_cfg=dict(), **kwargs):
        super(RCSample, self).__init__(**kwargs)
        self.loss_depth_weight = loss_depth_weight
        self.ins_channels = ins_channels
        self.out_channels = out_channels
        self.downsamples = downsamples
        self.with_cp=with_cp
        self.grid_config = grid_config
        self.input_size = input_size
        self.create_grid_infos(**grid_config)
        d = torch.arange(*grid_config['depth'], dtype=torch.float)
        self.D = d.shape[0]
        self.depthnet_cfg = depthnet_cfg
        self.depth_net = DepthNet(self.ins_channels[0], self.ins_channels[0],
                                    self.out_channels, self.D, with_cp=with_cp, **depthnet_cfg)
        self.scale_num = scale_num
        self.context_convs = nn.ModuleList([nn.Conv2d(self.ins_channels[i], self.out_channels, kernel_size=1) 
                         for i in range(1, self.scale_num)])
        self.depth_upconvs = nn.ModuleList([nn.Conv2d(self.ins_channels[i-1], self.ins_channels[i], kernel_size=1) 
                       for i in range(1, self.scale_num)])
        self.depth_convs = nn.ModuleList([nn.Conv2d(self.ins_channels[i], self.D, kernel_size=3, padding=1) 
                       for i in range(1, self.scale_num)])
        self.accelerate = accelerate
        if self.accelerate:
            self.pre_bev_coor = None

    def get_mlp_input(self, sensor2ego, ego2global, intrin, post_rot, post_tran, bda):
        B, N, _, _ = sensor2ego.shape
        bda = bda.view(B, 1, 3, 3).repeat(1, N, 1, 1)
        mlp_input = torch.stack([
            intrin[:, :, 0, 0],
            intrin[:, :, 1, 1],
            intrin[:, :, 0, 2],
            intrin[:, :, 1, 2],
            post_rot[:, :, 0, 0],
            post_rot[:, :, 0, 1],
            post_tran[:, :, 0],
            post_rot[:, :, 1, 0],
            post_rot[:, :, 1, 1],
            post_tran[:, :, 1],
            bda[:, :, 0, 0],
            bda[:, :, 0, 1],
            bda[:, :, 1, 0],
            bda[:, :, 1, 1],
            bda[:, :, 2, 2],], dim=-1)
        sensor2ego = sensor2ego[:,:,:3,:].reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        return mlp_input

    def create_grid_infos(self, x, y, z, **kwargs):
        self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [x, y, z]])
        self.grid_upper_bound = torch.Tensor([cfg[1] for cfg in [x, y, z]])
        self.grid_interval = torch.Tensor([cfg[2] for cfg in [x, y, z]])
        self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2]
                                       for cfg in [x, y, z]])
        x_coor = torch.tensor(list(range(self.grid_size[0].int()))) * x[2] + x[0] + 0.5 * x[2]
        y_coor = torch.tensor(list(range(self.grid_size[1].int()))) * y[2] + y[0] + 0.5 * y[2]
        y_coor, x_coor = torch.meshgrid(x_coor, y_coor)
        # x_coor, y_coor = torch.meshgrid(x_coor, y_coor)
        bev_coor = torch.stack([x_coor, y_coor, torch.zeros_like(x_coor)], dim=-1)
        self.bev_coor = bev_coor

    @force_fp32()
    def get_sample_coor(self, coor, sensor2ego, ego2global, cam2imgs, post_rots, post_trans, bda):
        B, N, _, _ = sensor2ego.shape
        coor = coor.unsqueeze(-1)
        coor = torch.inverse(bda).view(B, 1, 1, 3, 3).matmul(coor)
        coor = coor.unsqueeze(1).expand(-1, N, -1, -1, -1, -1)
        coor = coor - sensor2ego[:,:,:3, 3].view(B, N, 1, 1, 3, 1)
        coor = cam2imgs.matmul(torch.inverse(sensor2ego[:,:,:3,:3])).view(B, N, 1, 1, 3, 3).matmul(coor)
        coor[..., :2, :] /= coor[..., 2:3, :]
        coor = post_rots.view(B, N, 1, 1, 3, 3).matmul(coor)
        coor = coor + post_trans.view(B, N, 1, 1, 3, 1)
        coor = coor.squeeze(-1)
        coor[..., :2] /= self.downsamples[-1]
        # coor[..., 2] /= self.grid_config['depth'][-1]
        coor[..., 2] = (coor[..., 2] - self.grid_config['depth'][0]) / self.grid_config['depth'][-1]
        coor = torch.stack([coor[..., 0], coor[..., 2]], dim=-1)
        return coor
    
    def get_downsampled_gt_depth(self, gt_depths, downsample):
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // downsample,
                                   downsample, W // downsample,
                                   downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, downsample * downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // downsample,
                                   W // downsample)

        gt_depths = (
            gt_depths -
            (self.grid_config['depth'][0] -
             self.grid_config['depth'][2])) / self.grid_config['depth'][2]
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
                                                                           1:]
        return gt_depths.float()

    @force_fp32()
    def get_depth_loss(self, gt_depths, pred_depths):
        depth_loss = 0
        for i in range(self.scale_num):
            depth_labels = self.get_downsampled_gt_depth(gt_depths, self.downsamples[i])
            depth_preds = pred_depths[i].permute(0, 2, 3,
                                            1).contiguous().view(-1, self.D)
            fg_mask = torch.max(depth_labels, dim=1).values > 0.0
            depth_labels = depth_labels[fg_mask]
            depth_preds = depth_preds[fg_mask]
            with autocast(enabled=False):
                depth_loss += F.binary_cross_entropy(
                    depth_preds,
                    depth_labels,
                    reduction='none',
                ).sum() / max(1.0, fg_mask.sum()) * self.loss_depth_weight[i]
        return depth_loss

    @autocast(False)
    def forward(self, input):
        (x, sensor2ego, ego2global, cam2imgs, post_rots, post_trans, bda,
         mlp_input) = input[:8]

        depths = []
        # B, N, C, H, W = x[0].shape
        # input = x[0].view(B * N, C, H, W).float()
        B, N, C, H, W = x.shape
        input = x.view(B * N, C, H, W).float()
        depth, context, depth_feat = self.depth_net(input, mlp_input)
        depths.append(depth.softmax(dim=1))
        for i in range(1, self.scale_num):
            B, N, C, H, W = x[i].shape
            img_feat = x[i].view(B * N, C, H, W).float()
            context = F.interpolate(context, scale_factor=2, mode='bilinear')
            context += self.context_convs[i - 1](img_feat)
            depth_feat = F.interpolate(depth_feat, scale_factor=2, mode='bilinear')
            depth_feat = self.depth_upconvs[i - 1](depth_feat) + img_feat
            depth = self.depth_convs[i - 1](depth_feat)
            depths.append(depth.softmax(dim=1))

        frustum_feat = torch.matmul(depths[-1].permute(0, 3, 1, 2), 
                                    context.permute(0, 3, 2, 1)).permute(0, 3, 2, 1).contiguous()
        _, C, D, W = frustum_feat.shape
        if not self.accelerate or self.pre_bev_coor is None:
            bev_coor = self.bev_coor.clone().to(frustum_feat).unsqueeze(0).expand(B, -1, -1, -1)
            _, h, w, _ = bev_coor.shape
            bev_coor = self.get_sample_coor(bev_coor, sensor2ego, ego2global, cam2imgs, post_rots, post_trans, bda)

            norm_bev_coor = bev_coor.view(B * N, h, w, 2)
            norm_bev_coor[..., 0] = norm_bev_coor[..., 0] / W * 2 -1
            norm_bev_coor[..., 1] = norm_bev_coor[..., 1] / D * 2 -1
            if self.accelerate:
                self.pre_bev_coor = norm_bev_coor
        else:
            norm_bev_coor =  self.pre_bev_coor
        _, h, w, _ = norm_bev_coor.shape
        bev_feat = F.grid_sample(frustum_feat.view(B * N, C, D, W), norm_bev_coor)
        bev_feat = bev_feat.view(B, N, self.out_channels, h, w).sum(1)
        return bev_feat, depths