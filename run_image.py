from __future__ import absolute_import, division, print_function
import os
import argparse
from tqdm import tqdm
import yaml
import numpy as np
import cv2
import matplotlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from saver import save_point_cloud
from networks.models import *
from depth_anything_utils import Resize, NormalizeImage, PrepareForNet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--height', type=int, default=504)
    parser.add_argument('--resize', dest='resize', action='store_true', help='resize the output depth to the original size')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--save-cloud', dest='save_cloud', action='store_true', help='save point cloud result')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    # Load pre-trained weights
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DEVICE = 'cpu'
    model_dict = torch.load(config["model_path"], map_location=DEVICE)

    # network
    model = make(config['model'])
    if any(key.startswith('module') for key in model_dict.keys()):
        model = nn.DataParallel(model)
    if DEVICE == 'cuda':
        model.cuda()
    model_state_dict = model.state_dict()
    model.load_state_dict({k: v for k, v in model_dict.items() if k in model_state_dict})
    model.eval()

    # transform
    erp_height = args.height
    erp_width = 2 * erp_height
    transform = Compose([
        Resize(
            width=erp_width,
            height=erp_height,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = os.listdir(args.img_path)
        filenames = [os.path.join(args.img_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()
    
    os.makedirs(args.outdir, exist_ok=True)

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    for filename in tqdm(filenames):
        raw_image = cv2.imread(filename)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        h, w = image.shape[:2]
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            depth = model(image)["pred_depth"]

        cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_depth.png'), (depth[0,0].cpu().numpy()*1000).astype(np.uint16))

        if args.save_cloud:
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
            raw_image = cv2.resize(raw_image, (erp_width, erp_height), interpolation=cv2.INTER_CUBIC)
            save_point_cloud(raw_image, depth[0, 0].cpu().numpy(), os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.ply'))

        if args.resize:
            depth = F.interpolate(depth, (h, w), mode='bilinear', align_corners=False)[0, 0]
        else:
            depth = depth[0, 0]

        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.cpu().numpy().astype(np.uint8)
        
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png')
        if args.pred_only:
            cv2.imwrite(output_path, depth)
        else:
            if args.resize:
                split_region = np.ones((h, 50, 3), dtype=np.uint8) * 255
            else:
                raw_image = cv2.resize(raw_image, (erp_width, erp_height), interpolation=cv2.INTER_CUBIC)
                split_region = np.ones((erp_height, 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])
            
            cv2.imwrite(output_path, combined_result)