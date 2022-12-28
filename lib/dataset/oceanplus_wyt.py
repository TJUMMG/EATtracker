
from __future__ import division

import os
import cv2
import json
import torch
import random
import logging
import numpy as np
import torchvision.transforms as transforms
from scipy.ndimage.filters import gaussian_filter
from os.path import join
from easydict import EasyDict as edict
from torch.utils.data import Dataset

import sys

sys.path.append('../')
from utils.utils import *
from utils.cutout import Cutout
from core.config_ocean import config

sample_random = random.Random()


class OceanPlusDataset(Dataset):
    def __init__(self, cfg):
        super(OceanPlusDataset, self).__init__()
        # pair information
        self.template_size = cfg.OCEANPLUS.TRAIN.TEMPLATE_SIZE
        self.search_size = cfg.OCEANPLUS.TRAIN.SEARCH_SIZE

        self.size = 25
        self.stride = cfg.OCEANPLUS.TRAIN.STRIDE

        # aug information
        self.color = cfg.OCEANPLUS.DATASET.COLOR
        self.flip = cfg.OCEANPLUS.DATASET.FLIP
        self.rotation = cfg.OCEANPLUS.DATASET.ROTATION
        self.blur = cfg.OCEANPLUS.DATASET.BLUR
        self.shift = cfg.OCEANPLUS.DATASET.SHIFT
        self.scale = cfg.OCEANPLUS.DATASET.SCALE
        self.gray = cfg.OCEANPLUS.DATASET.GRAY
        self.label_smooth = cfg.OCEANPLUS.DATASET.LABELSMOOTH
        self.mixup = cfg.OCEANPLUS.DATASET.MIXUP
        self.cutout = cfg.OCEANPLUS.DATASET.CUTOUT

        # aug for search image
        self.shift_s = cfg.OCEANPLUS.DATASET.SHIFTs
        self.scale_s = cfg.OCEANPLUS.DATASET.SCALEs

        self.grids()

        self.transform_extra = transforms.Compose(
            [transforms.ToPILImage(), ] +
            ([transforms.ColorJitter(0.05, 0.05, 0.05, 0.05), ] if self.color > random.random() else [])
            + ([transforms.RandomHorizontalFlip(), ] if self.flip > random.random() else [])
            + ([transforms.RandomRotation(degrees=10), ] if self.rotation > random.random() else [])
            + ([transforms.Grayscale(num_output_channels=3), ] if self.gray > random.random() else [])
            + ([Cutout(n_holes=1, length=16)] if self.cutout > random.random() else [])
        )

        # train data information
        print('train datas: {}'.format(cfg.OCEANPLUS.TRAIN.WHICH_USE))
        self.train_datas = []  # all train dataset
        start = 0
        self.num = 0
        for data_name in cfg.OCEANPLUS.TRAIN.WHICH_USE:
            dataset = subData(cfg, data_name, start)
            self.train_datas.append(dataset)
            start += dataset.num  # real video number
            self.num += dataset.num_use  # the number used for subset shuffle

        self._shuffle()
        print(cfg)

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        """
        pick a vodeo/frame --> pairs --> data aug --> label
        """
        index = self.pick[index]
        dataset, index = self._choose_dataset(index)

        template, search = dataset._get_pairs(index, dataset.data_name)##每个都是元组，保存有511*511的图片路径，原始bbox，511*511的mask路径
        template, search = self.check_exists(index, dataset, template, search)

        template_image = cv2.imread(template[0])
        search_image = cv2.imread(search[0])

        template_box = self._toBBox(template_image, template[1])
        search_box = self._toBBox(search_image, search[1])##生成crop & resize后图片上的bbox,两点bbox

        ##读取mask
        template_mask = (cv2.imread(template[2], 0) > 0).astype(np.float32)##511*511,0/1
        search_mask = (cv2.imread(search[2], 0) > 0).astype(np.float32)

        ##读取template两种边缘contour, edge

        template_contour = (cv2.imread(template[3], 0) > 0).astype(np.float32)
        template_edge = (cv2.imread(template[4], 0) > 0).astype(np.float32)

        ##读取search两种边缘contour, edge

        search_contour = (cv2.imread(search[3], 0) > 0).astype(np.float32)
        search_edge = (cv2.imread(search[4], 0) > 0).astype(np.float32)


        template,template_mask, _, _, _, _ = self._augmentation(template_image, template_box,template_mask , template_contour, template_edge, self.template_size)
        search, search_mask, bbox, search_contour, search_edge, dag_param = self._augmentation(search_image, search_box, search_mask, search_contour, search_edge, self.search_size, search=True)

        ##template_mask = (np.expand_dims(template_mask, axis=0) > 0.5) * 2 - 1  ##将mask从0/1 变为1/-1
        template_mask = (template_mask > 0.5).astype(np.float32)##augumentation出来的mask经过缩放后边缘有小数出现,(127,127)
        search_mask = (search_mask> 0.5).astype(np.float32)##(255,255)
        search_contour = (search_contour> 0.5).astype(np.int32)
        search_edge = (search_edge > 0.5).astype(np.int32)

        ##获得edge计算损失时候的权重
        search_edge_weight = self.edge_weight(search_edge)
        # search_edge_weight_counter = self.edge_weight(1-search_edge)
        # search_edge_weight = search_edge_weight[np.newaxis,:]
        # search_edge_weight_counter = search_edge_weight_counter[np.newaxis,:]
        # search_edge_weight = np.concatenate((search_edge_weight_counter, search_edge_weight), axis=0)


        # # ##获得search mask resize 到 127*127大小的以后的edge
        # search_mask_resize  = cv2.resize(search_mask,(125,125))
        # search_mask_resize = (search_mask_resize>0.5).astype(np.float32)
        # ##----EGNet
        # [gx, gy] = np.gradient(search_mask_resize)
        # edge = gy * gy + gx * gx
        # search_resize_edge = (edge > 0).astype(np.float32)


        # from PIL image to numpy
        template = np.array(template)##(3,127,127)
        search = np.array(search)##(3,255,255)

        # ##保存训练数据查看
        # cv2.imwrite('/media/HardDisk/wyt/EATracker/tracking/xunlian_image/template_image.jpg',
        #             template_image)
        # cv2.imwrite('/media/HardDisk/wyt/EATracker/tracking/xunlian_image/search_image.jpg',
        #             search_image)
        # cv2.imwrite('/media/HardDisk/wyt/EATracker/tracking/xunlian_image/template.jpg',
        #             template)
        # cv2.imwrite('/media/HardDisk/wyt/EATracker/tracking/xunlian_image/search.jpg',
        #             search)
        # cv2.imwrite('/media/HardDisk/wyt/EATracker/tracking/xunlian_image/template_mask.jpg',
        #             template_mask * 255)
        # cv2.imwrite('/media/HardDisk/wyt/EATracker/tracking/xunlian_image/search_mask.jpg',
        #             search_mask * 255)
        # cv2.imwrite('/media/HardDisk/wyt/EATracker/tracking/xunlian_image/search_contour.jpg',
        #             search_contour * 255)
        # cv2.imwrite('/media/HardDisk/wyt/EATracker/tracking/xunlian_image/search_edge.jpg',
        #             search_edge * 255)
        # cv2.imwrite('/media/HardDisk/wyt/EATracker/tracking/xunlian_image/search_edge_weight_c.jpg',
        #             search_edge_weight[0] * 255)
        # cv2.imwrite('/media/HardDisk/wyt/EATracker/tracking/xunlian_image/search_edge_weight.jpg',
        #             search_edge_weight[1] * 255)

        template, search = map(lambda x: np.transpose(x, (2, 0, 1)).astype(np.float32), [template, search])


        return template, search,  np.array(template_mask, np.float32),np.array(search_mask, np.float32),\
               np.array(search_contour, np.float32), np.array(search_edge, np.float32), np.array(search_edge_weight, np.float32)# self.label 15*15/17*17
    ##计算edge损失时候使用的权重
    def edge_weight(self, target):
        shape = target.shape
        ones = np.ones(shape, dtype=np.int32)
        zeros = np.zeros(shape, dtype=np.int32)
        pos = (target == ones).astype(np.float32)  ##torch.eq返回每个位置的比较结果，相等为1,不等为0
        neg = (target == zeros).astype(np.float32)


        num_pos = np.sum(pos)
        num_neg = np.sum(neg)
        num_total = num_pos + num_neg

        alpha = num_neg / num_total  ##计算每个负样本的权重
        beta = 1.1 * num_pos / num_total

        weights = alpha * pos + beta * neg  ##正负样本的权重

        return weights
    # ------------------------------------
    # function groups for selecting pairs
    # ------------------------------------
    def grids(self):
        """
        each element of feature map on input search image
        :return: H*W*2 (position for each element)
        """
        sz = self.size

        sz_x = sz // 2
        sz_y = sz // 2

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        self.grid_to_search = {}
        self.grid_to_search_x = x * self.stride + self.search_size // 2
        self.grid_to_search_y = y * self.stride + self.search_size // 2

    def reg_label(self, bbox):
        """
        generate regression label
        :param bbox: [x1, y1, x2, y2]
        :return: [l, t, r, b]
        """
        x1, y1, x2, y2 = bbox
        l = self.grid_to_search_x - x1  # [17, 17]
        t = self.grid_to_search_y - y1
        r = x2 - self.grid_to_search_x
        b = y2 - self.grid_to_search_y

        l, t, r, b = map(lambda x: np.expand_dims(x, axis=-1), [l, t, r, b])
        reg_label = np.concatenate((l, t, r, b), axis=-1)  # [17, 17, 4]
        reg_label_min = np.min(reg_label, axis=-1)
        inds_nonzero = (reg_label_min > 0).astype(float)

        return reg_label, inds_nonzero

    def check_exists(self, index, dataset, template, search):
        name = dataset.data_name
        while True:
            if 'RGBT' in name or 'GTOT' in name and 'RGBTRGB' not in name and 'RGBTT' not in name:
                if not (os.path.exists(template[0][0]) and os.path.exists(search[0][0])):
                    index = random.randint(0, 100)
                    template, search = dataset._get_pairs(index, name)
                    continue
                else:
                    return template, search
            else:
                if not (os.path.exists(template[0]) and os.path.exists(search[0])):
                    index = random.randint(0, 100)
                    template, search = dataset._get_pairs(index, name)
                    continue
                else:
                    return template, search

    def _shuffle(self):
        """
        random shuffel
        """
        pick = []
        m = 0
        while m < self.num:
            p = []
            for subset in self.train_datas:
                sub_p = subset.pick
                p += sub_p
            sample_random.shuffle(p)

            pick += p
            m = len(pick)
        self.pick = pick
        print("dataset length {}".format(self.num))

    def _choose_dataset(self, index):
        for dataset in self.train_datas:
            if dataset.start + dataset.num > index:
                return dataset, index - dataset.start

    def _get_image_anno(self, video, track, frame, RGBT_FLAG=False):
        """
        get image and annotation
        """

        frame = "{:06d}".format(frame)
        if not RGBT_FLAG:
            image_path = join(self.root, video, "{}.{}.x.jpg".format(frame, track))
            image_anno = self.labels[video][track][frame]
            return image_path, image_anno
        else:  # rgb
            in_image_path = join(self.root, video, "{}.{}.in.x.jpg".format(frame, track))
            rgb_image_path = join(self.root, video, "{}.{}.rgb.x.jpg".format(frame, track))
            image_anno = self.labels[video][track][frame]
            in_anno = np.array(image_anno[-1][0])
            rgb_anno = np.array(image_anno[-1][1])

            return [in_image_path, rgb_image_path], (in_anno + rgb_anno) / 2

    def _get_pairs(self, index):
        """
        get training pairs
        """
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = random.choice(list(video.keys()))
        track_info = video[track]
        try:
            frames = track_info['frames']
        except:
            frames = list(track_info.keys())

        template_frame = random.randint(0, len(frames) - 1)

        left = max(template_frame - self.frame_range, 0)
        right = min(template_frame + self.frame_range, len(frames) - 1) + 1
        search_range = frames[left:right]
        template_frame = int(frames[template_frame])
        search_frame = int(random.choice(search_range))

        return self._get_image_anno(video_name, track, template_frame), \
               self._get_image_anno(video_name, track, search_frame)

    def _posNegRandom(self):
        """
        random number from [-1, 1]
        """
        return random.random() * 2 - 1.0

    def _toBBox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2] - shape[0], shape[3] - shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = self.template_size

        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)

        # if(w >= h):
        #     s_z = w
        # else:
        #     s_z = h

        scale_z = exemplar_size / s_z
        w = w * scale_z
        h = h * scale_z
        cx, cy = imw // 2, imh // 2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def _crop_hwc(self, image, bbox, out_sz, padding=(0, 0, 0)):
        """
        crop image
        """
        bbox = [float(x) for x in bbox]
        a = (out_sz - 1) / (bbox[2] - bbox[0])
        b = (out_sz - 1) / (bbox[3] - bbox[1])
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
        return crop

    def _draw(self, image, box, name):
        """
        draw image for debugging
        """
        draw_image = np.array(image.copy())
        x1, y1, x2, y2 = map(lambda x: int(round(x)), box)
        cv2.rectangle(draw_image, (x1, y1), (x2, y2), (0, 255, 0))
        cv2.circle(draw_image, (int(round(x1 + x2) / 2), int(round(y1 + y2) / 2)), 3, (0, 0, 255))
        cv2.putText(draw_image, '[x: {}, y: {}]'.format(int(round(x1 + x2) / 2), int(round(y1 + y2) / 2)),
                    (int(round(x1 + x2) / 2) - 3, int(round(y1 + y2) / 2) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (255, 255, 255), 1)
        cv2.imwrite(name, draw_image)

    def _draw_reg(self, image, grid_x, grid_y, reg_label, reg_weight, save_path, index):
        """
        visiualization
        reg_label: [l, t, r, b]
        """
        draw_image = image.copy()
        # count = 0
        save_name = join(save_path, '{:06d}.jpg'.format(index))
        h, w = reg_weight.shape
        for i in range(h):
            for j in range(w):
                if not reg_weight[i, j] > 0:
                    continue
                else:
                    x1 = int(grid_x[i, j] - reg_label[i, j, 0])
                    y1 = int(grid_y[i, j] - reg_label[i, j, 1])
                    x2 = int(grid_x[i, j] + reg_label[i, j, 2])
                    y2 = int(grid_y[i, j] + reg_label[i, j, 3])

                    draw_image = cv2.rectangle(draw_image, (x1, y1), (x2, y2), (0, 255, 0))

        cv2.imwrite(save_name, draw_image)

    def _mixupRandom(self):
        """
        gaussian random -- 0.3~0.7
        """
        return random.random() * 0.4 + 0.3

    # ------------------------------------
    # function for data augmentation
    # ------------------------------------
    def _augmentation(self, image, bbox, mask, contour, edge,  size, search=False):
        """
        data augmentation for input pairs (modified from SiamRPN.)
        """
        shape = image.shape
        crop_bbox = center2corner((shape[0] // 2, shape[1] // 2, size, size))
        param = edict()

        if search:
            param.shift = (self._posNegRandom() * self.shift_s, self._posNegRandom() * self.shift_s)  # shift
            param.scale = (
            (1.0 + self._posNegRandom() * self.scale_s), (1.0 + self._posNegRandom() * self.scale_s))  # scale change
        else:
            param.shift = (self._posNegRandom() * self.shift, self._posNegRandom() * self.shift)  # shift
            param.scale = (
            (1.0 + self._posNegRandom() * self.scale), (1.0 + self._posNegRandom() * self.scale))  # scale change

        crop_bbox, _ = aug_apply(Corner(*crop_bbox), param, shape)

        x1, y1 = crop_bbox.x1, crop_bbox.y1
        bbox = BBox(bbox.x1 - x1, bbox.y1 - y1, bbox.x2 - x1, bbox.y2 - y1)

        scale_x, scale_y = param.scale
        bbox = Corner(bbox.x1 / scale_x, bbox.y1 / scale_y, bbox.x2 / scale_x, bbox.y2 / scale_y)

        image = self._crop_hwc(image, crop_bbox, size)  # shift and scale
        mask = self._crop_hwc(mask, crop_bbox, size)
        contour = self._crop_hwc(contour, crop_bbox, size)
        edge = self._crop_hwc(edge, crop_bbox, size)

        if self.blur > random.random():
            image = gaussian_filter(image, sigma=(1, 1, 0))

        image = self.transform_extra(image)  # other data augmentation
        return image, mask, bbox, contour, edge, param

    def _mixupShift(self, image, size):
        """
        random shift mixed-up image
        """
        shape = image.shape
        crop_bbox = center2corner((shape[0] // 2, shape[1] // 2, size, size))
        param = edict()

        param.shift = (self._posNegRandom() * 64, self._posNegRandom() * 64)  # shift
        crop_bbox, _ = aug_apply(Corner(*crop_bbox), param, shape)

        image = self._crop_hwc(image, crop_bbox, size)  # shift and scale

        return image

    # ------------------------------------
    # function for creating training label
    # ------------------------------------
    def _dynamic_label(self, fixedLabelSize, c_shift, rPos=2, rNeg=0):
        if isinstance(fixedLabelSize, int):
            fixedLabelSize = [fixedLabelSize, fixedLabelSize]

        assert (fixedLabelSize[0] % 2 == 1)

        d_label = self._create_dynamic_logisticloss_label(fixedLabelSize, c_shift, rPos, rNeg)

        return d_label

    def _create_dynamic_logisticloss_label(self, label_size, c_shift, rPos=2, rNeg=0):
        if isinstance(label_size, int):
            sz = label_size
        else:
            sz = label_size[0]

        sz_x = sz // 2 + int(-c_shift[0] / 8)  # 8 is strides
        sz_y = sz // 2 + int(-c_shift[1] / 8)

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        dist_to_center = np.abs(x) + np.abs(y)  # Block metric
        label = np.where(dist_to_center <= rPos,
                         np.ones_like(y),
                         np.where(dist_to_center < rNeg,
                                  0.5 * np.ones_like(y),
                                  np.zeros_like(y)))
        return label


# ---------------------
# for a single dataset
# ---------------------

class subData(object):
    """
    for training with multi dataset, modified from SiamRPN
    """

    def __init__(self, cfg, data_name, start):
        self.data_name = data_name
        self.start = start

        info = cfg.OCEANPLUS.DATASET[data_name]
        self.frame_range = info.RANGE
        self.num_use = info.USE
        self.root = info.PATH

        with open(info.ANNOTATION) as fin:
            self.labels = json.load(fin)
            self._clean()
            self.num = len(self.labels)  # video numer

        self._shuffle()

    def _clean(self):
        """
        remove empty videos/frames/annos in dataset
        """
        # no frames
        to_del = []
        for video in self.labels:
            for track in self.labels[video]:
                frames = self.labels[video][track]
                frames = list(map(int, frames.keys()))
                frames.sort()
                self.labels[video][track]['frames'] = frames
                if len(frames) <= 0:
                    print("warning {}/{} has no frames.".format(video, track))
                    to_del.append((video, track))

        for video, track in to_del:
            try:
                del self.labels[video][track]
            except:
                pass

        # no track/annos
        to_del = []

        if self.data_name == 'YTB':
            to_del.append('train/1/YyE0clBPamU')  # This video has no bounding box.
        print(self.data_name)

        for video in self.labels:
            if len(self.labels[video]) <= 0:
                print("warning {} has no tracks".format(video))
                to_del.append(video)

        for video in to_del:
            try:
                del self.labels[video]
            except:
                pass

        self.videos = list(self.labels.keys())
        print('{} loaded.'.format(self.data_name))

    def _shuffle(self):
        """
        shuffel to get random pairs index (video)
        """
        lists = list(range(self.start, self.start + self.num))
        m = 0
        pick = []
        while m < self.num_use:
            sample_random.shuffle(lists)
            pick += lists
            m += self.num

        self.pick = pick[:self.num_use]
        return self.pick

    def _get_image_anno(self, video, track, frame):
        """
        get image and annotation
        """

        frame = "{:06d}".format(frame)

        image_path = join(self.root, video, "{}.{}.x.jpg".format(frame, track))
        image_anno = self.labels[video][track][frame]
        mask_path = join(self.root, video, "{}.{}.m.png".format(frame, track))##增加返回mask图片的路径
        contour_path = join(self.root, video, "{}.{}.c.png".format(frame, track))
        edge_path = join(self.root, video, "{}.{}.e.png".format(frame, track))
        return image_path, image_anno, mask_path, contour_path, edge_path

    def _get_pairs(self, index, data_name):
        """
        get training pairs
        """
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = random.choice(list(video.keys()))
        track_info = video[track]
        try:
            frames = track_info['frames']
        except:
            frames = list(track_info.keys())

        template_frame = random.randint(0, len(frames) - 1)

        left = max(template_frame - self.frame_range, 0)
        right = min(template_frame + self.frame_range, len(frames) - 1) + 1
        search_range = frames[left:right]

        template_frame = int(frames[template_frame])
        search_frame = int(random.choice(search_range))

        return self._get_image_anno(video_name, track, template_frame), \
               self._get_image_anno(video_name, track, search_frame)

    def _get_negative_target(self, index=-1):
        """
        dasiam neg
        """
        if index == -1:
            index = random.randint(0, self.num - 1)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = random.choice(list(video.keys()))
        track_info = video[track]

        frames = track_info['frames']
        frame = random.choice(frames)

        return self._get_image_anno(video_name, track, frame)


if __name__ == '__main__':
    # label = dynamic_label([17,17], [50, 30], 'balanced')shiw
    # print(label)
    import os
    from torch.utils.data import DataLoader
    from core.config import config

    train_set = OceanPlusDataset(config)
    train_loader = DataLoader(train_set, batch_size=16, num_workers=1, pin_memory=False)

    for iter, input in enumerate(train_loader):
        # label_cls = input[2].numpy()  # BCE need float
        template = input[0]
        search = input[1]
        print(template.size())
        print(search.size())

        print('dataset test')

    print()

