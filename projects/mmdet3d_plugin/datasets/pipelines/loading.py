import os

import mmcv
import numpy as np
import torch
from PIL import Image, ImageEnhance
from pyquaternion import Quaternion

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet.datasets.builder import PIPELINES
import torch.nn.functional as F

def mmlabNormalize(img):
    from mmcv.image.photometric import imnormalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    to_rgb = True
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img


@PIPELINES.register_module()
class PrepareImageInputs(object):

    def __init__(
        self,
        data_config,
        is_train=False,
        sequential=False,
        load_point_label=False,
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential
        self.load_point_label = load_point_label

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(
                self.data_config['cams']):
            cam_names = np.random.choice(
                self.data_config['cams'],
                self.data_config['Ncams'],
                replace=False)
        else:
            cam_names = self.data_config['cams']
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize_fix = float(fW) / float(W)
            resize_random = np.random.uniform(*self.data_config['resize'])
            depth_scale = 1 + resize_random
            resize = depth_scale * resize_fix
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) *
                         newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW) / float(W)
            depth_scale=1
            if scale is not None:
                resize += scale
            else:
                resize += self.data_config.get('resize_test', 0.0)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0   
        return resize, resize_dims, crop, flip, rotate, depth_scale

    def get_sensor_transforms(self, cam_info, cam_name):
        w, x, y, z = cam_info['cams'][cam_name]['sensor2ego_rotation']
        # sweep sensor to sweep ego
        sensor2ego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sensor2ego_tran = torch.Tensor(
            cam_info['cams'][cam_name]['sensor2ego_translation'])
        sensor2ego = sensor2ego_rot.new_zeros((4, 4))
        sensor2ego[3, 3] = 1
        sensor2ego[:3, :3] = sensor2ego_rot
        sensor2ego[:3, -1] = sensor2ego_tran
        # sweep ego to global
        w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
        ego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        ego2global_tran = torch.Tensor(
            cam_info['cams'][cam_name]['ego2global_translation'])
        ego2global = ego2global_rot.new_zeros((4, 4))
        ego2global[3, 3] = 1
        ego2global[:3, :3] = ego2global_rot
        ego2global[:3, -1] = ego2global_tran
        return sensor2ego, ego2global

    def point_label_transform(self, point_label, resize, resize_dims, crop, flip, rotate):
        H, W = resize_dims
        point_label[:, :2] = point_label[:, :2] * resize
        point_label[:, 0] -= crop[0]
        point_label[:, 1] -= crop[1]
        if flip:
            point_label[:, 0] = resize_dims[1] - point_label[:, 0]

        point_label[:, 0] -= W / 2.0
        point_label[:, 1] -= H / 2.0

        h = rotate / 180 * np.pi
        rot_matrix = [
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ]
        point_label[:, :2] = np.matmul(rot_matrix, point_label[:, :2].T).T

        point_label[:, 0] += W / 2.0
        point_label[:, 1] += H / 2.0

        coords = point_label[:, :2].astype(np.int16)

        depth_map = np.zeros(resize_dims)
        valid_mask = ((coords[:, 1] < resize_dims[0])
                    & (coords[:, 0] < resize_dims[1])
                    & (coords[:, 1] >= 0)
                    & (coords[:, 0] >= 0))
        depth_map[coords[valid_mask, 1],
                coords[valid_mask, 0]] = point_label[valid_mask, 2]
        semantic_map = np.zeros(resize_dims)
        semantic_map[coords[valid_mask, 1],
                coords[valid_mask, 0]] = (point_label[valid_mask, 3] >= 0)
        return torch.Tensor(depth_map), torch.Tensor(semantic_map)

    def get_inputs(self, results, flip=None, scale=None):
        imgs = []
        gt_depth = []
        gt_semantic = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names
        canvas = []
        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            img = Image.open(filename)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            intrin = torch.Tensor(cam_data['cam_intrinsic'])

            sensor2ego, ego2global = \
                self.get_sensor_transforms(results['curr'], cam_name)
            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate, depth_scale = img_augs
            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot, post_tran, resize, resize_dims, crop, 
                                        flip, rotate)
            
            if self.load_point_label:
                point_filename = filename.replace('samples/', 'samples_point_label/'
                                                  ).replace('.jpg','.npy')
                point_label = np.load(point_filename).astype(np.float64)[:4].T
                
                point_depth_augmented, point_semantic_augmented= \
                    self.point_label_transform(
                            point_label, resize, self.data_config['input_size'],
                            crop, flip, rotate)
                point_depth_augmented = point_depth_augmented / depth_scale
                gt_depth.append(point_depth_augmented)
                gt_semantic.append(point_semantic_augmented)
            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2
            post_rot[2, 2] = 1 / depth_scale

            canvas.append(np.array(img))
            imgs.append(self.normalize_img(img))

            if self.sequential:
                assert 'adjacent' in results
                for adj_info in results['adjacent']:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    img_adjacent = Image.open(filename_adj)
                    img_adjacent = self.img_transform_core(
                        img_adjacent,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))
            intrins.append(intrin)
            sensor2egos.append(sensor2ego)
            ego2globals.append(ego2global)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        if self.sequential:
            for adj_info in results['adjacent']:
                post_trans.extend(post_trans[:len(cam_names)])
                post_rots.extend(post_rots[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])

                # align
                for cam_name in cam_names:
                    sensor2ego, ego2global = \
                        self.get_sensor_transforms(adj_info, cam_name)
                    sensor2egos.append(sensor2ego)
                    ego2globals.append(ego2global)

        imgs = torch.stack(imgs)
        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)
        intrins = torch.stack(intrins)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        results['canvas'] = canvas
        if self.load_point_label:
            gt_depth = torch.stack(gt_depth)
            gt_semantic = torch.stack(gt_semantic)
            results['gt_depth'] = gt_depth
            results['gt_semantic'] = gt_semantic

        # collect can_bus of each sample
        can_bus = [torch.tensor(results['curr']['can_bus']).float()]
        if self.sequential:
            for adj_info in results['adjacent']:
                can_bus.append(torch.tensor(adj_info['can_bus']).float())
        can_bus = torch.stack(can_bus)
        can_bus[:-1, :3] -= can_bus[1:, :3]
        can_bus[:-1, -1] -= can_bus[1:, -1]
        can_bus[-1, :3] = 0
        can_bus[-1, -1] = 0
        results['can_bus'] = can_bus
        # results['img'] = imgs
        bda = torch.eye(3)
        return (imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda)

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        return results

def _do_paste_mask(masks, boxes, img_h, img_w, skip_empty=True):
    device = masks.device
    if skip_empty:
        x0_int, y0_int = torch.clamp(
            boxes.min(dim=0).values.floor()[:2] - 1,
            min=0).to(dtype=torch.int32)
        x1_int = torch.clamp(
            boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(
            boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device).to(torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device).to(torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)
    # IsInf op is not supported with ONNX<=1.7.0
    if not torch.onnx.is_in_onnx_export():
        if torch.isinf(img_x).any():
            inds = torch.where(torch.isinf(img_x))
            img_x[inds] = 0
        if torch.isinf(img_y).any():
            inds = torch.where(torch.isinf(img_y))
            img_y[inds] = 0

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    img_masks = F.grid_sample(
        masks.to(dtype=torch.float32), grid, align_corners=False)

    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()

def get_mask(mask_filename, img_shape=(900,1600)):
    if not os.path.exists(mask_filename):
        return Image.fromarray(np.zeros(img_shape, dtype=np.float32))
    mask_preds = torch.load(mask_filename, map_location='cpu')    
    mask_pred = mask_preds['masks'].float() / 255
    bboxes = mask_preds['bboxes'][:,:4]
    labels = mask_preds['labels']
    N = len(mask_pred)
    img_h, img_w = img_shape
    im_mask = torch.zeros(img_h, img_w, dtype=torch.float32)
    for i in range(N):
        masks_chunk, spatial_inds = _do_paste_mask(mask_pred[i:i+1], bboxes[i:i+1], img_h, img_w, skip_empty=True)
        im_mask[spatial_inds] += masks_chunk.squeeze(0)
    mask = Image.fromarray(im_mask.numpy())
    return mask

@PIPELINES.register_module()
class PrepareImageInputsGeoBEV(PrepareImageInputs):

    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize_fix = float(fW) / float(W)
            resize_random = np.random.uniform(*self.data_config['resize'])
            depth_scale = 1 + resize_random
            resize = depth_scale * resize_fix
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) *
                         newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW) / float(W)
            depth_scale=1
            if scale is not None:
                resize += scale
            else:
                resize += self.data_config.get('resize_test', 0.0)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0   
        return resize, resize_dims, crop, flip, rotate, depth_scale

    def img_mask_transform(self, img, mask, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):        
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)
        mask = self.img_transform_core(mask, resize_dims, crop, flip, rotate)
        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, mask, post_rot, post_tran

    def get_inputs(self, results, flip=None, scale=None):
        imgs = []
        masks = []
        gt_depth = []
        gt_semantic = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names
        canvas = []
        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            img = Image.open(filename)
            mask_filename = filename.replace('samples/', 'samples_instance_mask/')[:-4]+'.pt'
            mask = get_mask(mask_filename)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            intrin = torch.Tensor(cam_data['cam_intrinsic'])

            sensor2ego, ego2global = \
                self.get_sensor_transforms(results['curr'], cam_name)
            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate, depth_scale = img_augs
            img, mask, post_rot2, post_tran2 = \
                self.img_mask_transform(img, mask, post_rot, post_tran, resize, resize_dims, crop, 
                                        flip, rotate)
            mask = torch.Tensor((np.array(mask)>0.25))

            if self.load_point_label:
                point_filename = filename.replace('samples/', 'samples_point_label/'
                                                  ).replace('.jpg','.npy')
                point_label = np.load(point_filename).astype(np.float64)[:4].T
                
                point_depth_augmented, point_semantic_augmented= \
                    self.point_label_transform(
                            point_label, resize, self.data_config['input_size'],
                            crop, flip, rotate)
                point_depth_augmented = point_depth_augmented / depth_scale
                gt_depth.append(point_depth_augmented)
                gt_semantic.append(point_semantic_augmented)
            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2
            post_rot[2, 2] = 1 / depth_scale

            canvas.append(np.array(img))
            imgs.append(self.normalize_img(img))
            masks.append(mask)

            if self.sequential:
                assert 'adjacent' in results
                for adj_info in results['adjacent']:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    img_adjacent = Image.open(filename_adj)
                    img_adjacent = self.img_transform_core(
                        img_adjacent,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))
            intrins.append(intrin)
            sensor2egos.append(sensor2ego)
            ego2globals.append(ego2global)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        if self.sequential:
            for adj_info in results['adjacent']:
                post_trans.extend(post_trans[:len(cam_names)])
                post_rots.extend(post_rots[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])

                # align
                for cam_name in cam_names:
                    sensor2ego, ego2global = \
                        self.get_sensor_transforms(adj_info, cam_name)
                    sensor2egos.append(sensor2ego)
                    ego2globals.append(ego2global)

        imgs = torch.stack(imgs)
        masks = torch.stack(masks)
        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)
        intrins = torch.stack(intrins)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        results['canvas'] = canvas
        results['gt_fg'] = masks
        if self.load_point_label:
            gt_depth = torch.stack(gt_depth)
            gt_semantic = torch.stack(gt_semantic)
            results['gt_depth'] = gt_depth
            results['gt_semantic'] = gt_semantic

        # collect can_bus of each sample
        can_bus = [torch.tensor(results['curr']['can_bus']).float()]
        if self.sequential:
            for adj_info in results['adjacent']:
                can_bus.append(torch.tensor(adj_info['can_bus']).float())
        can_bus = torch.stack(can_bus)
        can_bus[:-1, :3] -= can_bus[1:, :3]
        can_bus[:-1, -1] -= can_bus[1:, -1]
        can_bus[-1, :3] = 0
        can_bus[-1, -1] = 0
        results['can_bus'] = can_bus
        # results['img'] = imgs
        bda = torch.eye(3)
        return (imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda)
    

@PIPELINES.register_module()
class PrepareImageInputsGeoBEV_canbus(PrepareImageInputsGeoBEV):
    def get_inputs(self, results, flip=None, scale=None):
        imgs = []
        masks = []
        gt_depth = []
        gt_semantic = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names
        canvas = []
        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            img = Image.open(filename)
            mask_filename = filename.replace('samples/', 'samples_instance_mask/')[:-4]+'.pt'
            mask = get_mask(mask_filename)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            intrin = torch.Tensor(cam_data['cam_intrinsic'])

            sensor2ego, ego2global = \
                self.get_sensor_transforms(results['curr'], cam_name)
            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate, depth_scale = img_augs
            img, mask, post_rot2, post_tran2 = \
                self.img_mask_transform(img, mask, post_rot, post_tran, resize, resize_dims, crop, 
                                        flip, rotate)
            mask = torch.Tensor((np.array(mask)>0.25))

            if self.load_point_label:
                point_filename = filename.replace('samples/', 'samples_point_label/'
                                                  ).replace('.jpg','.npy')
                point_label = np.load(point_filename).astype(np.float64)[:4].T
                
                point_depth_augmented, point_semantic_augmented= \
                    self.point_label_transform(
                            point_label, resize, self.data_config['input_size'],
                            crop, flip, rotate)
                point_depth_augmented = point_depth_augmented / depth_scale
                gt_depth.append(point_depth_augmented)
                gt_semantic.append(point_semantic_augmented)
            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2
            post_rot[2, 2] = 1 / depth_scale

            canvas.append(np.array(img))
            imgs.append(self.normalize_img(img))
            masks.append(mask)

            if self.sequential:
                assert 'adjacent' in results
                for adj_info in results['adjacent']:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    img_adjacent = Image.open(filename_adj)
                    img_adjacent = self.img_transform_core(
                        img_adjacent,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))
            intrins.append(intrin)
            sensor2egos.append(sensor2ego)
            ego2globals.append(ego2global)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        if self.sequential:
            for adj_info in results['adjacent']:
                post_trans.extend(post_trans[:len(cam_names)])
                post_rots.extend(post_rots[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])

                # align
                for cam_name in cam_names:
                    sensor2ego, ego2global = \
                        self.get_sensor_transforms(adj_info, cam_name)
                    sensor2egos.append(sensor2ego)
                    ego2globals.append(ego2global)

        imgs = torch.stack(imgs)
        masks = torch.stack(masks)
        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)
        intrins = torch.stack(intrins)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        results['canvas'] = canvas
        results['gt_fg'] = masks
        if self.load_point_label:
            gt_depth = torch.stack(gt_depth)
            gt_semantic = torch.stack(gt_semantic)
            results['gt_depth'] = gt_depth
            results['gt_semantic'] = gt_semantic

        # collect can_bus of each sample
        can_bus = [torch.tensor(results['curr']['can_bus']).float()]
        if self.sequential:
            for adj_info in results['adjacent']:
                can_bus.append(torch.tensor(adj_info['can_bus']).float())
        can_bus = torch.stack(can_bus)
        can_bus[:-1, :3] -= can_bus[1:, :3]
        can_bus[:-1, -1] -= can_bus[1:, -1]
        can_bus[-1, :3] = 0
        can_bus[-1, -1] = 0
        results['can_bus'] = can_bus
        # results['img'] = imgs
        bda = torch.eye(3)
        return (imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda)