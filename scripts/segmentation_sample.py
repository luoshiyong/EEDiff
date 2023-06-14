

import argparse
import os
import nibabel as nib
from visdom import Visdom
viz = Visdom()
import sys
from glob import glob
from guided_diffusion.lits_edge_dataset import Dataset
import random
sys.path.append(".")
import numpy as np
import time
from tqdm import tqdm
import torch as th
from time import time
from PIL import Image
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.isicloader import ISICDataset
import torchvision.utils as vutils
from guided_diffusion.utils import staple
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torchvision.transforms as transforms
from torchsummary import summary
seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)
    # dataset
    val_img_paths = glob('U:/paper4/data256/val_image/*')
    val_mask_paths = glob('U:/paper4/data256/val_mask/*')
    # val_img_paths = glob('U:/paper4/data256/test_image/*')
    # val_mask_paths = glob('U:/paper4/data256/test_mask/*')
    ds = Dataset("", val_img_paths, val_mask_paths,val=True)
    args.in_ch = 4

    datal = th.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False)
    data = iter(datal)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    all_images = []


    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    # 0 --- 1 --- 1
    # while len(all_images) * args.batch_size < args.num_samples:
    for ixx in range(1):
        b, m,dismap, path = next(data)  #should return an image from the dataloader "data"
        # print("img = {} | m ={} | path = {}".format(b.shape,m.shape,path))
        # [1,3,256,256] --- [1,1,256,256] --- 32_232
        c = th.randn_like(b[:, :1, ...])          # [1,1,256,256]
        img = th.cat((b, c), dim=1)     #add a noise channel [1,4,256,256]
        slice_ID=path[0]
        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        enslist = []
        liver_dice = []
        tumor_dice = []
        for ipx, (b, m, dismap, path) in tqdm(enumerate(datal), total=len(datal)):
            # b, m,dismap, path = next(data)  #should return an image from the dataloader "data"
            # print("img = {} | m ={} | path = {}".format(b.shape,m.shape,path))
            # [1,3,256,256] --- [1,1,256,256] --- 32_232
            c = th.randn_like(b[:, :1, ...])  # [1,1,256,256]
            img = th.cat((b, c), dim=1)  # add a noise channel [1,4,256,256]
            slice_ID = path[0]
            logger.log("sampling...")

            start = th.cuda.Event(enable_timing=True)
            end = th.cuda.Event(enable_timing=True)
            enslist = []
            res_metric_old = {'liver_dice': 0.0, 'tumor_dice': 0.0, 'liver_save_numpy': None, 'tumor_save_numpy': None}
            stime = time()
            for i in range(1):  # args.num_ensemble= 5 ///this is for the generation of an ensemble of 5 masks.
                model_kwargs = {}
                start.record()
                sample_fn = (
                    diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
                )
                # [] --- [1,4,256,256] --- [1,4,256,256] --- [1,1,256,256] --- [1,1,256,256]
                # seg_dpm,_,_,seg head,fusion seg
                sample, x_noisy, org, cal, cal_out, res_metric = sample_fn(
                    model,
                    (args.batch_size, 3, args.image_size, args.image_size), img, slice_ID, m, dismap,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )
                etime = time()
                print('this case use {:.3f} s'.format(etime-stime))
                # update metirc
                if res_metric['liver_dice'] > res_metric_old['liver_dice']:
                    res_metric_old['liver_dice'] = res_metric['liver_dice']
                    res_metric_old['liver_save_numpy'] = res_metric['liver_save_numpy']
                if res_metric['tumor_dice'] > res_metric_old['tumor_dice']:
                    res_metric_old['tumor_dice'] = res_metric['tumor_dice']
                    res_metric_old['tumor_save_numpy'] = res_metric['tumor_save_numpy']
                end.record()
                th.cuda.synchronize()
                print('time for 1 sample', start.elapsed_time(end))  # time measurement for the generation of 1 sample

                co = th.tensor(cal_out)
                co = th.cat((co, co, co), 1)  # [1,3,256,256]
                enslist.append(co)
            print("best liver dice = {} | best tumor dice = {}".format(res_metric_old['liver_dice'],res_metric_old['tumor_dice']))

            """
            # save gt
            liver_gt = m.detach().cpu().numpy()[0][0]
            # tumor_gt = m.detach().cpu().numpy()[0][1]
            liver_gt[liver_gt > 0] = 255
            # liver_gt[tumor_gt > 0] = 255
            save_gt = Image.fromarray(np.uint8(liver_gt))
            save_gt.convert('L').save("U:/paper4/EHDiffsegLIver/img_out/gt/" + slice_ID + '.jpg')
            # save dismap
            dismap_save = dismap.detach().cpu().numpy()[0][0] * 255
            dismap_save = Image.fromarray(np.uint8(dismap_save))
            dismap_save.convert('L').save("U:/paper4/EHDiffsegLIver/img_out/dismap_gt/" + slice_ID + '.jpg')
            """

            # save liver and tumor img
            # print("liver max = {} | liver min = {}".format(res_metric_old['liver_save_numpy'].max(),res_metric_old['liver_save_numpy'].min()))
            # print("tumor max = {} | tumor min = {}".format(res_metric_old['tumor_save_numpy'].max(),res_metric_old['tumor_save_numpy'].min()))
            res_metric_old['liver_save_numpy'][res_metric_old['liver_save_numpy'] >= 0.5] = 255
            res_metric_old['liver_save_numpy'][res_metric_old['liver_save_numpy'] < 0.5] = 0
            # res_metric_old['liver_save_numpy'][res_metric_old['tumor_save_numpy'] >= 0.5] = 255
            save_img = Image.fromarray(np.uint8(res_metric_old['liver_save_numpy']))
            save_img.convert('L').save("U:/paper4/EHDiffsegLIver/img_out/predict_liver/" + slice_ID + '.jpg')
            # record liver and tumor dice for compute
            liver_dice.append(res_metric_old['liver_dice'])
            tumor_dice.append(res_metric_old['tumor_dice'])

            ensres = staple(th.stack(enslist, dim=0)).squeeze(0)  # [1,3,256,256]
            # print("ensres shape = ", ensres.shape)
            ensres[ensres >= 0.5] = 255
            ensres[ensres < 0.5] = 0
            vutils.save_image(ensres, fp="U:/paper4/EHDiffsegLIver/img_out/ensemble_seg/" + str(slice_ID) + ".jpg", nrow=1, padding=10)
            # vutils.save_image(ensres, fp=args.out_dir + str(slice_ID) + ".jpg", nrow=1, padding=10)
            # max = 0.8564726710319519 | min = 0.05633966997265816
        print("liver dice = ", np.mean(np.array(liver_dice)))
        print("tumor dice = ", np.mean(np.array(tumor_dice)))
def create_argparser():
    defaults = dict(
        data_name = 'ISIC',
        data_dir="/home/luosy/EHDiffsegLIver/data_dir",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="U:/paper4/savedmodel045000_nomst.pt",
        num_ensemble=5,      #number of samples in the ensemble
        gpu_dev = "0",
        out_dir='./results/'
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
