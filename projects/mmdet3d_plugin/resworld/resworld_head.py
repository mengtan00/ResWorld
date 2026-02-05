import copy
from math import pi, cos, sin
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS, build_loss 
from mmdet.models.dense_heads import DETRHead
from mmcv.runner import force_fp32, auto_fp16
from mmcv.runner import BaseModule
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.cnn import Linear, bias_init_with_prob
from torch.cuda.amp.autocast_mode import autocast
from mmdet.models.backbones.resnet import BasicBlock

from .tokenlearner import *
from .rcsample import Mlp, SELayer
from mmcv.cnn.bricks.conv_module import ConvModule
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding

class MLN(nn.Module):
    ''' 
    from "https://github.com/exiawsh/StreamPETR"
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim=256, use_ln=True):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.use_ln = use_ln

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        if self.use_ln:
            self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.init_weight()

    def init_weight(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        if self.use_ln:
            x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta

        return out

class SELayerMLP(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.mlp_reduce = nn.Linear(channels, channels)
        self.act1 = act_layer()
        self.mlp_expand = nn.Linear(channels, channels)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.mlp_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.mlp_expand(x_se)
        return x * self.gate(x_se)

@HEADS.register_module()
class ResWorldHead(BaseModule):
    def __init__(self,
                #  *args,
                 grid_config,
                 num_frames=3,
                 embed_dims=256,
                 in_channels=256,
                 num_reg_fcs=2,
                 positional_encoding=None,
                 bev_h=30,
                 bev_w=30,
                 fut_ts=6,
                 fut_mode=6,
                 num_scenes=16,
                 latent_decoder=None,
                 res_latent_decoder=None,
                 way_decoder=None,
                 ego_fut_mode=3,
                 loss_plan_reg=dict(type='L1Loss', loss_weight=0.25),
                 ego_lcf_feat_idx=None,
                 valid_fut_ts=6,
                 **kwargs):
        super(ResWorldHead, self).__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.embed_dims = embed_dims
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.latent_decoder = latent_decoder
        self.res_latent_decoder = res_latent_decoder
        self.way_decoder = way_decoder
        self.positional_encoding = positional_encoding
        self.ego_fut_mode = ego_fut_mode
        self.ego_lcf_feat_idx = ego_lcf_feat_idx
        self.valid_fut_ts = valid_fut_ts
        self.num_scenes = num_scenes
        self.num_frames = num_frames
        self.grid_min = torch.tensor([grid_config['x'][0], grid_config['y'][0]])
        self.grid_max = torch.tensor([grid_config['x'][1], grid_config['y'][1]])
        self.grid_size = torch.tensor([grid_config['x'][2], grid_config['y'][2]])

        self._init_layers()
        self.loss_plan_reg = build_loss(loss_plan_reg)
        self.loss_plan_reg_init = build_loss(loss_plan_reg)
                
    def _init_layers(self):
        ego_fut_decoder = []
        ego_fut_dec_in_dim = self.embed_dims + len(self.ego_lcf_feat_idx) \
            if self.ego_lcf_feat_idx is not None else self.embed_dims
        for _ in range(self.num_reg_fcs):
            ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, ego_fut_dec_in_dim))
            ego_fut_decoder.append(nn.ReLU())
        ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, 2))
        self.ego_fut_decoder = nn.Sequential(*ego_fut_decoder)
        init_ego_fut_decoder = []
        for _ in range(self.num_reg_fcs):
            init_ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, ego_fut_dec_in_dim))
            init_ego_fut_decoder.append(nn.ReLU())
        init_ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, 2))
        self.init_ego_fut_decoder = nn.Sequential(*init_ego_fut_decoder)

        self.navi_embedding = nn.Embedding(3, self.embed_dims)
        self.navi_se = SELayerMLP(self.embed_dims)
        self.canbus_mlp = Mlp(18, self.embed_dims, self.embed_dims)
        self.canbus_se = SELayerMLP(self.embed_dims)
        self.bev_fusion_conv = ConvModule(self.in_channels * self.num_frames, self.in_channels, 
                                          kernel_size=3, padding=1)

        self.way_point = nn.Embedding(self.ego_fut_mode*self.fut_ts, self.embed_dims * 2)
        self.tokenlearner = TokenLearner(self.num_scenes, self.embed_dims * 2)
        self.res_tokenlearner = TokenLearner(self.num_scenes, self.embed_dims * 2)
        self.tokenfuser = TokenFuser(self.num_scenes, 256)

        self.latent_decoder = build_transformer_layer_sequence(self.latent_decoder)
        self.way_decoder = build_transformer_layer_sequence(self.way_decoder)
        self.res_latent_decoder = build_transformer_layer_sequence(self.res_latent_decoder)
        self.col_attn = MultiScaleDeformableAttention(self.embed_dims,
                                            num_points=8, num_levels=1)
        self.action_mln = MLN(6*2)
        self.positional_encoding = build_positional_encoding(
            self.positional_encoding)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        if self.latent_decoder is not None:
            for p in self.latent_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p) 
        if self.way_decoder is not None:
            for p in self.way_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        if self.res_latent_decoder is not None:
            for p in self.res_latent_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p) 

    @force_fp32(apply_to=('bev_feats'))
    def forward(self,
                bev_inputs,
                img_metas,
                prev_bev=None,
                only_bev=False,
                ego_his_trajs=None,
                ego_lcf_feat=None,
                cmd=None
            ):
        
        bev_feats, can_bus_infos = bev_inputs
        bt, c, h, w = bev_feats.shape
        bs = bt // self.num_frames
        dtype = bev_feats[0].dtype
        device = bev_feats[0].device
        can_bus_infos = self.canbus_mlp(can_bus_infos.permute(1, 0, 2)).view(bt, 1, self.in_channels)
        bev_embed = self.canbus_se(bev_feats.view(bt, c, h*w).permute(0, 2, 1), can_bus_infos)
        bev_embed_single = bev_embed.clone()
        bev_embed = bev_embed.permute(0, 2, 1).view(self.num_frames, bs, c, h, w).permute(1, 0, 2, 3, 4)
        bev_embed = self.bev_fusion_conv(bev_embed.reshape(bs, self.num_frames * c, h, w))
        bev_feats = bev_feats.view(self.num_frames, bs, c, h, w)


        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_feats.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        pos_embd = bev_pos.flatten(2).permute(0, 2, 1)
        bev_embed = bev_embed.reshape(bs, c, h * w).permute(0, 2, 1)
        # res_embed = res_embed.reshape(bs, c, h * w).permute(0, 2, 1)

        navi_embed = []
        for bidx in range(bs):
            cmd_idx = torch.nonzero(cmd[bidx, 0, 0])[0, 0]
            navi_embed.append(self.navi_embedding.weight[cmd_idx][None, None])
        navi_embed = torch.cat(navi_embed, dim=0)

        bev_navi_embed = self.navi_se(bev_embed, navi_embed)

        bev_query = torch.cat((bev_navi_embed, pos_embd), -1)

        learned_latent_query, selected = self.tokenlearner(bev_query)
        _, res_selected = self.res_tokenlearner(bev_query)
        bev_embed_single = torch.cat((bev_embed_single, pos_embd.repeat(self.num_frames,1,1)), -1)
        bev_embed_single = torch.einsum('bsi,bic->bsc', res_selected.repeat(self.num_frames,1,1), bev_embed_single)\
                            .view(self.num_frames, bs, learned_latent_query.shape[1],learned_latent_query.shape[2])
        res_latent_query_all = bev_embed_single[:-1] - bev_embed_single[1:]

        learned_latent_query=learned_latent_query.permute(1, 0, 2)
        latent_query, latent_pos = torch.split(
            learned_latent_query, self.embed_dims, dim=2)

        latent_query = self.latent_decoder(
                query=latent_query,
                key=latent_query,
                value=latent_query,
                query_pos=latent_pos,
                key_pos=latent_pos)

        way_point = self.way_point.weight.to(dtype)
        wp_pos, way_point = torch.split(
            way_point, self.embed_dims, dim=1)

        wp_pos = wp_pos.unsqueeze(0).expand(bs, -1, -1)
        way_point = way_point.unsqueeze(0).expand(bs, -1, -1)
        wp_pos = wp_pos.permute(1, 0, 2)
        way_point = way_point.permute(1, 0, 2)

        way_point = self.way_decoder(
                query=way_point,
                key=latent_query,
                value=latent_query,
                query_pos=wp_pos,
                key_pos=latent_pos)
        init_ego_trajs = self.init_ego_fut_decoder(way_point)
        init_ego_trajs = init_ego_trajs.permute(1, 0, 2). view(bs, 
                                                    self.ego_fut_mode, self.fut_ts, 2)
        init_ego_coords = init_ego_trajs.cumsum(dim=2).view(bs, -1, 2)
        init_ego_coords = (init_ego_coords - self.grid_min.to(device).view(1, 1, 2)) / \
                            self.grid_size.to(device).view(1, 1, 2) / 200
        init_wp_vector = []
        for bidx in range(bs):
            cmd_idx = torch.nonzero(cmd[bidx, 0, 0])[0, 0]
            init_wp_vector.append(init_ego_trajs[bidx, cmd_idx, ...].reshape(1, 1, 12))  
        init_wp_vector = torch.cat(init_wp_vector, dim=1)

        reference_points = init_ego_coords.unsqueeze(2)
        spatial_shapes = torch.tensor([self.bev_w, self.bev_h]).view(1, 2).to(device)
        level_start_index = torch.tensor([0]).to(device)

        res_latent_query=res_latent_query_all[0].permute(1, 0, 2)
        res_latent_query, res_latent_pos = torch.split(
            res_latent_query, self.embed_dims, dim=2)
        res_latent_query = self.action_mln(res_latent_query, init_wp_vector)
        res_latent_query = self.res_latent_decoder(
                query=res_latent_query,
                key=res_latent_query,
                value=res_latent_query,
                query_pos=res_latent_pos,
                key_pos=res_latent_pos)
        
        pred_bev = self.tokenfuser(res_latent_query.permute(1, 0, 2), bev_navi_embed) + bev_navi_embed

        way_point = self.col_attn(
                query=way_point,
                key=pred_bev.permute(1, 0, 2),
                value=pred_bev.permute(1, 0, 2),
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index) 

        outputs_ego_trajs = self.ego_fut_decoder(way_point)
        outputs_ego_trajs = outputs_ego_trajs.permute(1, 0, 2). view(bs, 
                                                      self.ego_fut_mode, self.fut_ts, 2)

        wp_vector = []
        for bidx in range(bs):
            cmd_idx = torch.nonzero(cmd[bidx, 0, 0])[0, 0]
            wp_vector.append(outputs_ego_trajs[bidx, cmd_idx, ...].reshape(1, 1, 12))  
        wp_vector = torch.cat(wp_vector, dim=1)

        outs = {
            'bev_embed': bev_embed,
            'pred_bev': pred_bev,
            'scene_query': latent_query,
            'wp_vector': wp_vector,
            # 'act_query': act_query,
            # 'act_pos': act_pos,
            'ego_fut_preds': outputs_ego_trajs,
            # 'ego_fut_preds': init_ego_trajs,
            'init_ego_fut_preds': init_ego_trajs,
        }

        return outs
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             map_gt_bboxes_list,
             map_gt_labels_list,
             preds_dicts,
             ego_fut_gt,
             ego_fut_masks,
             ego_fut_cmd,
             gt_attr_labels,
             gt_bboxes_ignore=None,
             map_gt_bboxes_ignore=None,
             img_metas=None):

        ego_fut_preds = preds_dicts['ego_fut_preds']
        init_ego_fut_preds = preds_dicts['init_ego_fut_preds']
        loss_dict = dict()

        # Planning Loss
        ego_fut_gt = ego_fut_gt.squeeze(1)
        ego_fut_masks = ego_fut_masks.squeeze(1).squeeze(1)
        ego_fut_cmd = ego_fut_cmd.squeeze(1).squeeze(1)

        ego_fut_gt = ego_fut_gt.unsqueeze(1).repeat(1, self.ego_fut_mode, 1, 1)
        loss_plan_l1_weight = ego_fut_cmd[..., None, None] * ego_fut_masks[:, None, :, None]
        loss_plan_l1_weight = loss_plan_l1_weight.repeat(1, 1, 1, 2)

        loss_plan_l1 = self.loss_plan_reg(
            ego_fut_preds,
            ego_fut_gt,
            loss_plan_l1_weight
        )

        loss_plan_l1_init = self.loss_plan_reg_init(
            init_ego_fut_preds,
            ego_fut_gt,
            loss_plan_l1_weight
        )
      
        loss_dict['loss_plan_reg'] = loss_plan_l1
        loss_dict['loss_plan_reg_init'] = loss_plan_l1_init

        return loss_dict


