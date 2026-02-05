# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE

from mmdet3d.models.builder import NECKS
from mmcv.cnn import build_norm_layer
from torch.utils.checkpoint import checkpoint

# from mmcv.cnn.bricks.conv_module import ConvModule
from mmdet.models import BACKBONES
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck

@NECKS.register_module()
class CustomFPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 out_ids=[],
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(CustomFPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.out_ids = out_ids
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            # assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            if i in self.out_ids:
                fpn_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [self.fpn_convs[i](laterals[i]) for i in self.out_ids]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return outs[0]

@NECKS.register_module()
class CustomFPN_all(CustomFPN):
    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [self.fpn_convs[i](laterals[i]) for i in self.out_ids]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return outs

@NECKS.register_module()
class FPN_LSS(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 input_feature_index=(0, 2),
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2,
                 lateral=None,
                 use_input_conv=False):
        super().__init__()
        self.input_feature_index = input_feature_index
        self.extra_upsample = extra_upsample is not None
        self.up = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=True)
        # assert norm_cfg['type'] in ['BN', 'SyncBN']
        channels_factor = 2 if self.extra_upsample else 1
        self.input_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * channels_factor,
                kernel_size=1,
                padding=0,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        ) if use_input_conv else None
        if use_input_conv:
            in_channels = out_channels * channels_factor
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * channels_factor,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels * channels_factor,
                out_channels * channels_factor,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        )
        if self.extra_upsample:
            self.up2 = nn.Sequential(
                nn.Upsample(
                    scale_factor=extra_upsample,
                    mode='bilinear',
                    align_corners=True),
                nn.Conv2d(
                    out_channels * channels_factor,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False),
                build_norm_layer(norm_cfg, out_channels, postfix=0)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=1, padding=0),
            )
        self.lateral = lateral is not None
        if self.lateral:
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(
                    lateral, lateral, kernel_size=1, padding=0, bias=False),
                build_norm_layer(norm_cfg, lateral, postfix=0)[1],
                nn.ReLU(inplace=True),
            )

    def forward(self, feats):
        x2, x1 = feats[self.input_feature_index[0]], \
                 feats[self.input_feature_index[1]]
        if self.lateral:
            x2 = self.lateral_conv(x2)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        if self.input_conv is not None:
            x = self.input_conv(x)
        x = self.conv(x)
        if self.extra_upsample:
            x = self.up2(x)
        return x

@NECKS.register_module()
class FPN_LSS_single(FPN_LSS):
    def forward(self, feats):
        x = feats[self.input_feature_index]
        x = self.conv(x)
        return x

@NECKS.register_module()
class BEVDecoderPure1(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2,
                 lateral=None,
                 use_input_conv=False):
        super(BEVDecoderPure1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels[-1] + in_channels[-2], out_channels * 2, kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * 2)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=3,
                      padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * 2)[1],
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels * 2 + in_channels[-3], out_channels * 2, kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * 2)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3,
                      padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.Upsample(scale_factor=extra_upsample, mode='bilinear', align_corners=True)
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        )
    
    def forward(self, feats):
        x1, x2, x3 = feats
        x = torch.cat([self.up1(x3), x2], dim=1)
        x = self.conv1(x)
        x = torch.cat([self.up2(x), x1], dim=1)
        x = self.conv2(x)
        x = self.conv3(self.up3(x))
        return x

@BACKBONES.register_module()
class CustomResNet(nn.Module):

    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            norm_cfg=dict(type='BN'),
            with_cp=False,
            block_type='Basic',
    ):
        super(CustomResNet, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        if block_type == 'BottleNeck':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [
                    Bottleneck(
                        curr_numC,
                        num_channels[i] // 4,
                        stride=stride[i],
                        downsample=nn.Conv2d(curr_numC, num_channels[i], 3,
                                             stride[i], 1),
                        norm_cfg=norm_cfg)
                ]
                curr_numC = num_channels[i]
                layer.extend([
                    Bottleneck(curr_numC, curr_numC // 4, norm_cfg=norm_cfg)
                    for _ in range(num_layer[i] - 1)
                ])
                layers.append(nn.Sequential(*layer))
        elif block_type == 'Basic':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [
                    BasicBlock(
                        curr_numC,
                        num_channels[i],
                        stride=stride[i],
                        downsample=nn.Conv2d(curr_numC, num_channels[i], 3,
                                             stride[i], 1),
                        norm_cfg=norm_cfg)
                ]
                curr_numC = num_channels[i]
                layer.extend([
                    BasicBlock(curr_numC, curr_numC, norm_cfg=norm_cfg)
                    for _ in range(num_layer[i] - 1)
                ])
                layers.append(nn.Sequential(*layer))
        else:
            assert False
        self.layers = nn.Sequential(*layers)

        self.with_cp = with_cp

    def forward(self, x):
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class CustomTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default: `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(CustomTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                key_padding_mask=None,
                *args,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        intermediate = []
        for lid, layer in enumerate(self.layers):
            query = layer(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                key_padding_mask=key_padding_mask,
                *args,
                **kwargs)

            if self.return_intermediate:
                intermediate.append(query)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return query