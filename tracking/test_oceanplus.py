import _init_paths
import os
import cv2
import json
import torch
import random
import argparse
import numpy as np
import models.models as models
from tqdm import tqdm
from numpy import *
import yaml
try:
    from torch2trt import TRTModule
except:
    print('Warning: TensorRT is not successfully imported')
from PIL import Image
from os.path import exists, join, dirname, realpath
from tracker.oceanplus import OceanPlus
from tracker.online import ONLINE
from easydict import EasyDict as edict
from utils.utils import load_pretrain, cxy_wh_2_rect, get_axis_aligned_bbox, load_dataset, poly_iou
import pdb

def parse_args():
    """
    args for fc testing.
    """
    parser = argparse.ArgumentParser(description='PyTorch SiamFC Tracking Test')
    parser.add_argument('--arch', dest='arch', default='OceanPlus', choices=['OceanPlus', 'OceanPlusTRT'], help='backbone architecture')
    parser.add_argument('--mms', default='True', type=str, choices=['True', 'False'], help='wether to use MMS')
    parser.add_argument('--resume', default="/media/HardDisk/wyt/EATracker/tracking/snapshot/", type=str, help='pretrained model')
    parser.add_argument('--dataset', default='DAVIS2016', help='dataset test')
    parser.add_argument('--online', action="store_true", help='whether to use online')
    parser.add_argument('--vis', default=False, action="store_true", help='visualize tracking results')
    parser.add_argument('--hp', default=None, type=str, help='hyper-parameters')
    parser.add_argument('--debug', default=False, type=str, help='debug or not')
    args = parser.parse_args()

    return args


def rle_to_mask(rle, width, height):
    """
    rle: input rle mask encoding
    each evenly-indexed element represents number of consecutive 0s
    each oddly indexed element represents number of consecutive 1s
    width and height are dimensions of the mask
    output: 2-D binary mask
    """
    # allocate list of zeros
    v = [0] * (width * height)

    # set id of the last different element to the beginning of the vector
    idx_ = 0

    for i in range(len(rle)):
        if i % 2 != 0:
            # write as many 1s as RLE says (zeros are already in the vector)
            for j in range(rle[i]):
                v[idx_+j] = 1
        idx_ += rle[i]

    # reshape vector into 2-D mask
    # return np.reshape(np.array(v, dtype=np.uint8), (height, width)) # numba bug / not supporting np.reshape
    return np.array(v, dtype=np.uint8).reshape((height, width))


def create_mask_from_string(mask_encoding):
    """
    mask_encoding: a string in the following format: x0, y0, w, h, RLE
    output: mask, offset
    mask: 2-D binary mask, size defined in the mask encoding
    offset: (x, y) offset of the mask in the image coordinates
    """
    elements = [int(el) for el in mask_encoding]
    tl_x, tl_y, region_w, region_h = elements[:4]
    rle = np.array([el for el in elements[4:]], dtype=np.int32)
    mask = rle_to_mask(rle, region_w, region_h)

    return mask

def save_prediction(prediction, palette, save_path, save_name):
    if prediction.ndim > 2:
        img = Image.fromarray(np.uint8(prediction[0, ...]))
    else:
        img = Image.fromarray(np.uint8(prediction))
    img = img.convert('L')
    img.putpalette(palette)
    img = img.convert('P')
    img.save('{}/{}.png'.format(save_path, save_name))


def track(siam_tracker, online_tracker, siam_net, video, args):
    """
    track a single video in VOT2020
    attention: not for benchmark evaluation, just a demo
    TODO: add cyclic initiation
    """

    start_frame, toc = 0, 0
    image_files, gt = video['image_files'], video['gt']

    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)
        if args.online:
            rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if len(im.shape) == 2: im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)   # align with training

        tic = cv2.getTickCount()
        if f == start_frame:  # init
            lx, ly, w, h = eval(gt[f][1:])[:4]
            cx = lx + w / 2
            cy = ly + h / 2

            target_pos = np.array([cx, cy])
            target_sz = np.array([w, h])

            mask_roi = create_mask_from_string(eval(gt[f][1:]))
            hi, wi, _ = im.shape
            mask_gt = np.zeros((hi, wi))
            mask_gt[ly:ly + h, lx:lx + w] = mask_roi

            state = siam_tracker.init(im, target_pos, target_sz, siam_net, online=args.online, mask=mask_gt, debug=args.debug)  # init siamese tracker

            if args.online:
                online_tracker.init(im, rgb_im, siam_net, target_pos, target_sz, True, dataname=args.dataset, resume=args.resume)

        elif f > start_frame:  # tracking
            if args.online:
                state = online_tracker.track(im, rgb_im, siam_tracker, state)
            else:
                state = siam_tracker.track(state, im, name=image_file)
            mask = state['mask']

            if args.vis:
                COLORS = np.random.randint(128, 255, size=(1, 3), dtype="uint8")
                COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")
                mask = COLORS[mask]
                output = ((0.4 * im) + (0.6 * mask)).astype("uint8")
                cv2.imshow("mask", output)
                cv2.waitKey(1)

        toc += cv2.getTickCount() - tic

    toc /= cv2.getTickFrequency()
    print('Video: {:12s} Time: {:2.1f}s Speed: {:3.1f}fps'.format(video['name'], toc, f / toc))


def track_vos(siam_tracker, online_tracker, siam_net, video, args, epoch, hp=None):
    # re = args.resume.split('/')[-1].split('.')[0]

    re = epoch

    if hp is None:
        save_path = join('result', args.dataset, re, video['name'])
        save_mask_path = join('result_mask', args.dataset, re, video['name'])
        if not exists(save_mask_path):
            os.makedirs(save_mask_path)
    else:
        # re = re+'_thr_{:.2f}_lambdaU_{:.2f}_lambdaS_{:.2f}_iter1_{:.2f}_iter2_{:.2f}'.format(hp['seg_thr'], hp['lambda_u'], hp['lambda_s'], hp['iter1'], hp['iter2'])
        re = re+'_pk_{:.3f}_wi_{:.2f}_lr_{:.2f}'.format(hp['penalty_k'], hp['window_influence'], hp['lr'])
        save_path = join('result', args.dataset, re, video['name'])

    if exists(save_path):
        return

    image_files = video['image_files']##list-dir
    annos = [Image.open(x) for x in video['anno_files'] if exists(x)]##list-PngImageFile
    palette = annos[0].getpalette()##list-768,以列表形式返回图像调色板
    annos = [np.array(an) for an in annos]##list-ndarray，背景为0,物体为1,2，....(dtype=uint8)

    if 'anno_init_files' in video:
        annos_init = [np.array(Image.open(x)) for x in video['anno_init_files']]
    else:
        annos_init = [annos[0]]

    mot_enable = args.dataset in ['DAVIS2017', 'YTBVOS']

    if not mot_enable:
        annos = [(anno > 0).astype(np.uint8) for anno in annos]
        annos_init = [(anno_init > 0).astype(np.uint8) for anno_init in annos_init]

    if 'start_frame' in video:
        object_ids = [int(id) for id in video['start_frame']]
    else:
        object_ids = [o_id for o_id in np.unique(annos[0]) if o_id != 0]##找到初始帧mask中所有的物体以及其mask中所代表的数值
        if len(object_ids) != len(annos_init):
            annos_init = annos_init*len(object_ids)##按照物体的个数复制初始帧mask
    object_num = len(object_ids)##物体数量
    toc = 0
    pred_masks = np.zeros((object_num, len(image_files), annos[0].shape[0], annos[0].shape[1]))
    for obj_id, o_id in enumerate(object_ids):
        if 'start_frame' in video:
            start_frame = video['start_frame'][str(o_id)]
            end_frame = video['end_frame'][str(o_id)]
        else:
            start_frame, end_frame = 0, len(image_files)

        for f, image_file in enumerate(image_files):
            im = cv2.imread(image_file)
            if args.online:
                rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            tic = cv2.getTickCount()
            if f == start_frame:  # init
                mask = annos_init[obj_id] == o_id##物体标注数字与mask中所有数字相比较，相同为True，不同为False
                mask = mask.astype(np.uint8)##True/False mask 变成 0/1 mask
                x, y, w, h = cv2.boundingRect(mask)##生成mask的边界框，无bbox初始化需要自己生成，用来生成搜索
                cx, cy = x + w/2, y + h/2
                target_pos = np.array([cx, cy])
                target_sz = np.array([w, h])
                state = siam_tracker.init(im, target_pos, target_sz, siam_net, online=args.online, mask=mask, hp=hp, debug=args.debug)  # init tracker

                if args.online:
                    online_tracker.init(im, rgb_im, siam_net, target_pos, target_sz, True, dataname=args.dataset,
                                        resume=args.resume)
                pred_masks[obj_id, f, :, :] = mask
            elif end_frame >= f > start_frame:  # tracking
                if args.online:
                    state = online_tracker.track(im, rgb_im, siam_tracker, state, name=image_file)
                else:
                    state = siam_tracker.track(state, im, name=image_file)
                mask = state['mask']   # binary
                mask_ori = state['mask_ori']   # probabilistic
                # ##---
                # cv2.imshow("mask", mask_ori*255)
                # cv2.waitKey(1)
                # ##---
            toc += cv2.getTickCount() - tic
            if end_frame >= f >= start_frame:
                if f == start_frame:
                    pred_masks[obj_id, f, :, :] = mask
                else:
                    if args.dataset in ['DAVIS2017', 'YTBVOS']:   # multi-object
                        pred_masks[obj_id, f, :, :] = mask_ori
                    else:
                        pred_masks[obj_id, f, :, :] = mask

            # if args.vis:
            #     COLORS = np.random.randint(128, 255, size=(1, 3), dtype="uint8")
            #     COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")
            #     mask = COLORS[mask]
            #     output = ((0.4 * im) + (0.6 * mask)).astype("uint8")
            #     cv2.imshow("mask", output)
            #     cv2.waitKey(1)

            if args.vis:
                COLORSmask = np.zeros(shape=(1, 3), dtype="uint8")
                COLORSmask[0][1] = COLORSmask[0][1]+255
                COLORSmask[0][2] = COLORSmask[0][2] +255
                COLORSmask = np.vstack([[0, 0, 0], COLORSmask]).astype("uint8")

                COLORSedge = np.zeros(shape=(1, 3), dtype="uint8")
                COLORSedge[0][1] = COLORSedge[0][1]+255
                COLORSedge[0][2] = COLORSedge[0][2] +255
                COLORSedge = np.vstack([[0, 0, 0], COLORSedge]).astype("uint8")
                ## get edge
                [gx, gy] = np.gradient(mask)
                edge = gy * gy + gx * gx
                edge = (edge > 0).astype("uint8")

                antiedge = (edge == 0).astype("uint8")
                antiedge = np.expand_dims(antiedge, 2)
                antimask = (mask == 0).astype("uint8")
                antimask = np.expand_dims(antimask,2)
                antiim = im * antimask



                mask = COLORSmask[mask]
                edge = COLORSedge[edge]


                output = ( ((0.7*im)+ (0.3*mask)+(0.3*antiim))*antiedge + edge).astype("uint8")

                cv2.imshow("mask", output)
                cv2.waitKey(1)

                ##save predictmask
                save_mask = join(save_mask_path,str(f).zfill(5)+'.png')
                cv2.imwrite(save_mask,output)

    toc /= cv2.getTickFrequency()

    # ##save for evaluation------
    #
    # re = epoch
    #
    # if hp is None:
    #     re = re+'_sg_{:.4f}'.format( state['p'].seg_thr)
    #     save_path = join('result', args.dataset, re, video['name'])
    # else:
    #     # re = re+'_thr_{:.2f}_lambdaU_{:.2f}_lambdaS_{:.2f}_iter1_{:.2f}_iter2_{:.2f}'.format(hp['seg_thr'], hp['lambda_u'], hp['lambda_s'], hp['iter1'], hp['iter2'])
    #     re = re+'_pk_{:.3f}_wi_{:.2f}_lr_{:.2f}'.format(hp['penalty_k'], hp['window_influence'], hp['lr'])
    #     save_path = join('result', args.dataset, re, video['name'])
    #
    # if exists(save_path):
    #     return
    # ##----------

    if not exists(save_path):
        os.makedirs(save_path)

    if args.dataset == 'DAVIS2016':
        for idx in range(f+1):
            save_name = str(idx).zfill(5)
            save_prediction(pred_masks[:, idx, ...], palette, save_path, save_name)
    elif args.dataset in ['DAVIS2017', 'YTBVOS']:
        pred_mask_final = np.array(pred_masks)
        pred_mask_final = (np.argmax(pred_mask_final,  axis=0).astype('uint8') + 1) * (
                np.max(pred_mask_final, axis=0) > state['p'].seg_thr).astype('uint8')
        for idx in range(f+1):
            if not args.dataset == 'YTBVOS':
                save_name = str(idx).zfill(5)
            else:
                save_name = image_files[idx].split('/')[-1].split('.')[0]

            save_prediction(pred_mask_final[idx, ...], palette, save_path, save_name)
    else:
        raise ValueError('not supported dataset')

    print('Video: {:12s} Time: {:2.1f}s Speed: {:3.1f}fps'.format(video['name'], toc, (f+1) / toc))
    return (f+1) / toc


def main(epochlist):
    print('Warning: this is a demo to test OceanPlus')
    print('Warning: if you want to test it on VOT2020, please use our integration scripts')
    args = parse_args()

    info = edict()
    info.arch = args.arch
    info.dataset = args.dataset
    info.online = args.online
    info.TRT = 'TRT' in args.arch

    siam_info = edict()
    siam_info.arch = args.arch
    siam_info.dataset = args.dataset
    siam_info.vis = args.vis
    siam_tracker = OceanPlus(siam_info)

    if args.mms == 'True':
        MMS = True
    else:
        MMS = False
    for epoch in epochlist:
        siam_net = models.__dict__[args.arch](online=args.online, mms=MMS)
        print('===> init Siamese <====')
        siam_net = load_pretrain(siam_net, os.path.join(args.resume, epoch))
        siam_net.eval()
        siam_net = siam_net.cuda()

        # if info.TRT:
        #     print('===> load model from TRT <===')
        #     print('===> please ignore the warning information of TRT <===')
        #     trtNet = reloadTRT()
        #     siam_net.tensorrt_init(trtNet)

        if args.online:
            online_tracker = ONLINE(info)
        else:
            online_tracker = None

        print('====> warm up <====')
        for i in tqdm(range(20)):
            siam_net.template(torch.rand(1, 3, 127, 127).cuda(), torch.rand(1, 127, 127).cuda())
            siam_net.track(torch.rand(1, 3, 255, 255).cuda())

        # prepare video
        print('====> load dataset <====')
        dataset = load_dataset(args.dataset)
        video_keys = list(dataset.keys()).copy()

        # hyper-parameters in or not
        if args.hp is None:
            hp = None
        elif isinstance(args.hp, str):
            f = open(join('tune', args.hp), 'r')
            hp = json.load(f)
            f.close()
            print('====> tuning hp: {} <===='.format(hp))
        else:
            raise ValueError('not supported hyper-parameters')

        # tracking all videos in benchmark
        for video in video_keys:
            video_speeds = []
            speed_sum = 0
            if args.dataset in ['DAVIS2016', 'DAVIS2017', 'YTBVOS']:  # VOS
                video_speed = track_vos(siam_tracker, online_tracker, siam_net, dataset[video], args, epoch, hp)
                video_speeds.append(video_speed)
                # speed_sum += video_speed

            else:  # VOTS (i.e. VOT2020)
                if video == 'butterfly':
                    track(siam_tracker, online_tracker, siam_net, dataset[video], args)

        # print('speed avg {}'.format(speed_sum/len(video_speeds)))

def set_yaml_para(yamlPath, para, num):

    with open(yamlPath) as f:
        doc = yaml.safe_load(f)
    doc['TEST']['DAVIS2016'][para]  = num
    with open(yamlPath, 'w') as f:
        yaml.safe_dump(doc, f, default_flow_style=False)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # epochlist = ['checkpoint_e1.pth','checkpoint_e2.pth','checkpoint_e3.pth','checkpoint_e4.pth','checkpoint_e5.pth','checkpoint_e6.pth','checkpoint_e7.pth','checkpoint_e8.pth','checkpoint_e9.pth','checkpoint_e10.pth','checkpoint_e11.pth','checkpoint_e12.pth', 'checkpoint_e13.pth', 'checkpoint_e14.pth', 'checkpoint_e15.pth',
    #              'checkpoint_e16.pth', 'checkpoint_e17.pth', 'checkpoint_e18.pth', 'checkpoint_e19.pth',
    #              'checkpoint_e20.pth']
    # epochlist = ['checkpoint_e1.pth','checkpoint_e2.pth','checkpoint_e3.pth','checkpoint_e4.pth','checkpoint_e5.pth','checkpoint_e6.pth','checkpoint_e7.pth','checkpoint_e8.pth','checkpoint_e9.pth','checkpoint_e10.pth','checkpoint_e11.pth','checkpoint_e12.pth', 'checkpoint_e13.pth', 'checkpoint_e14.pth', 'checkpoint_e15.pth',
    #              'checkpoint_e16.pth', 'checkpoint_e17.pth', 'checkpoint_e18.pth', 'checkpoint_e19.pth',
    #              'checkpoint_e20.pth','checkpoint_e21.pth', 'checkpoint_e22.pth', 'checkpoint_e23.pth',
    #              'checkpoint_e24.pth', 'checkpoint_e25.pth', 'checkpoint_e26.pth', 'checkpoint_e27.pth',
    #              'checkpoint_e28.pth', 'checkpoint_e29.pth', 'checkpoint_e30.pth','checkpoint_e31.pth','checkpoint_e32.pth','checkpoint_e33.pth',
    #              'checkpoint_e34.pth', 'checkpoint_e35.pth',
    #              'checkpoint_e36.pth', 'checkpoint_e37.pth', 'checkpoint_e38.pth', 'checkpoint_e39.pth',
    #              'checkpoint_e40.pth','checkpoint_e41.pth','checkpoint_e42.pth','checkpoint_e43.pth','checkpoint_e44.pth','checkpoint_e45.pth',
    #              'checkpoint_e46.pth','checkpoint_e47.pth','checkpoint_e48.pth','checkpoint_e49.pth','checkpoint_e50.pth',
    #              ]
    # # epochlist = ['checkpoint_e1.pth','checkpoint_e2.pth','checkpoint_e3.pth','checkpoint_e4.pth','checkpoint_e5.pth','checkpoint_e6.pth','checkpoint_e7.pth','checkpoint_e8.pth','checkpoint_e9.pth','checkpoint_e10.pth']
    # # epochlist = [  'checkpoint_e9.pth','checkpoint_e10.pth']
    # epochlist = ['OceanPlusMMS.pth']
    epochlist = ['checkpoint_e14.pth']
    # epochlist = [ 'checkpoint_e2.pth', 'checkpoint_e3.pth', 'checkpoint_e4.pth',
    #              'checkpoint_e5.pth', 'checkpoint_e6.pth', 'checkpoint_e7.pth', 'checkpoint_e8.pth',
    #              'checkpoint_e9.pth', 'checkpoint_e10.pth', 'checkpoint_e11.pth', 'checkpoint_e12.pth',
    #              'checkpoint_e13.pth', 'checkpoint_e14.pth', 'checkpoint_e15.pth',
    #              'checkpoint_e16.pth', 'checkpoint_e17.pth', 'checkpoint_e18.pth', 'checkpoint_e19.pth',
    #              'checkpoint_e20.pth', 'checkpoint_e21.pth', 'checkpoint_e22.pth', 'checkpoint_e23.pth',
    #              'checkpoint_e24.pth', 'checkpoint_e25.pth', 'checkpoint_e26.pth', 'checkpoint_e27.pth',
    #              'checkpoint_e28.pth', 'checkpoint_e29.pth', 'checkpoint_e30.pth']

    # yamlPath = '/media/HardDisk/wyt/EATracker/experiments/test/DAVIS/OceanPlus.yaml'
    # num = 0.82
    # for i in range(50):
    #     num += 0.001
    #     num = round(num, 5)
    #     set_yaml_para(yamlPath,'seg_thr',num)
    #     print(i,'-',num)
    #     main(epochlist)

    main(epochlist)

