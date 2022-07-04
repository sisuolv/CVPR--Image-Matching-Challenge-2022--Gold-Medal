import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from dkm import dkm_base


import kornia
from kornia_moons.feature import *
import kornia as K
import kornia.feature as KF


def draw_img_match(img1, img2, mkpts0, mkpts1, inliers, ax):
    display_vertical = False  # img1.shape[1]/img1.shape[0]>0.8 or img2.shape[1]/img2.shape[0]>0.8

    draw_LAF_matches(
        KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0).view(1, -1, 2),
                                     torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
                                     torch.ones(mkpts0.shape[0]).view(1, -1, 1)),

        KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1).view(1, -1, 2),
                                     torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
                                     torch.ones(mkpts1.shape[0]).view(1, -1, 1)),
        torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
        K.tensor_to_image(img1),
        K.tensor_to_image(img2),
        inliers,
        draw_dict={'inlier_color': (0.2, 1, 0.2),
                   'tentative_color': None,
                   'feature_color': (0.2, 0.5, 1), 'vertical': display_vertical}, ax=ax)


# import torch
#
# torch.hub.set_dir('/kaggle/working/pretrained/')


use_dkm_filter_strategy = True
dkm_ft_num = 2_000  # 1000
dkm_max_confid_matches = 200
dkm_thrs_conf_match = 0.2


def filter_conf_matches(mkpts0, mkpts1, mconf, thrs_conf_match=0.2, max_matches=-1, at_least_matches=-1, tag=''):
    # if dry_run:
    #     print(tag, 'mconf len', len(mconf))
    #     print(tag, 'mkpts0 len', len(mkpts0))
    #     print(tag, 'mkpts0', mkpts0[:5])

    # sort points by confidence descending
    mkps1_sorted = [x for (y, x) in sorted(zip(mconf, mkpts0), key=lambda pair: pair[0], reverse=True)]
    mkps2_sorted = [x for (y, x) in sorted(zip(mconf, mkpts1), key=lambda pair: pair[0], reverse=True)]

    # overwrite
    mkps1 = np.array(mkps1_sorted)
    mkps2 = np.array(mkps2_sorted)

    # limit
    # at least conf > 0.3 so that point is confident
    # thrsh_certain = sum(conf > dkm_thrs_conf_match for conf in sparse_certainty)
    num_thrsh_greater = (mconf >= thrs_conf_match).sum()

    take_first_el = num_thrsh_greater if max_matches == -1 else min(max_matches, num_thrsh_greater)
    if at_least_matches != -1 and take_first_el < at_least_matches:
        take_first_el = min(len(mkps1), at_least_matches)  # take list at most or at most least matches
    if take_first_el > 0 and len(mkps1) > take_first_el:
        mkps1 = mkps1[:take_first_el]
        mkps2 = mkps2[:take_first_el]

    # if dry_run:
    #     print(tag, 'new mkpts0 len', len(mkps1))

    return mkps1, mkps2

def dkm_inf(image_fpath_1, image_fpath_2, use_cuda):
    # https://github.com/Parskatt/DKM
    # internally it resizes to h=384 w=512
    img1 = cv2.imread(image_fpath_1)
    img2 = cv2.imread(image_fpath_2)

    img1PIL = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    img2PIL = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    dense_matches, dense_certainty = dkm_model.match(img1PIL, img2PIL, use_cuda=use_cuda)

    # # You may want to process these, e.g. we found dense_certainty = dense_certainty.sqrt() to work quite well in some cases.
    dense_certainty = dense_certainty.sqrt()
    # Sample 10000 sparse matches
    sparse_matches, sparse_certainty = dkm_model.sample(dense_matches, dense_certainty, dkm_ft_num)

    mkps1 = sparse_matches[:, :2]
    mkps2 = sparse_matches[:, 2:]

    h, w, c = img1.shape
    mkps1[:, 0] = ((mkps1[:, 0] + 1) / 2) * w
    mkps1[:, 1] = ((mkps1[:, 1] + 1) / 2) * h

    h, w, c = img2.shape
    mkps2[:, 0] = ((mkps2[:, 0] + 1) / 2) * w
    mkps2[:, 1] = ((mkps2[:, 1] + 1) / 2) * h


    if use_dkm_filter_strategy:
        mkps1, mkps2 = filter_conf_matches(mkps1, mkps2, sparse_certainty, \
            thrs_conf_match=dkm_thrs_conf_match, max_matches=dkm_max_confid_matches, tag='dkm')

    return mkps1, mkps2, sparse_certainty


def load_resized_image(fname, max_image_size):
    img = cv2.imread(fname)
    if max_image_size == -1:
        # no resize
        return img, 1.0
    scale = max_image_size / max(img.shape[0], img.shape[1])
    w = int(img.shape[1] * scale)
    h = int(img.shape[0] * scale)
    img = cv2.resize(img, (w, h))
    return img, scale


def load_torch_kornia_image(fname, device, max_image_size):
    img,_ = load_resized_image(fname, max_image_size)
    img = K.image_to_tensor(img, False).float() /255.
    img = K.color.bgr_to_rgb(img)
    orig_img_device = img.to(device)
    return orig_img_device


root_path = '/Users/igla/Downloads/image-matching-challenge-2022'
image_fpath_1 = f'{root_path}/train/notre_dame_front_facade/images/43500770_5422051381.jpg'
image_fpath_2 = f'{root_path}/train/notre_dame_front_facade/images/00092603_8230821547.jpg'

# image_fpath_1 = f'{root_path}/test_images/d91db836/81dd07fb7b9a4e01996cee637f91ca1a.png'
# image_fpath_2 = f'{root_path}/test_images/d91db836/0006b1337a0347f49b4e651c035dfa0e.png'


dkm_device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

request_im_size = (576, 768)
dkm_model = dkm_base(pretrained=True, version="v11", use_cuda=False, request_im_size=request_im_size)
dkm_model = dkm_model.to(dkm_device).eval()

dkm_mkpts1, dkm_mkpts2, sparse_certainty = dkm_inf(image_fpath_1, image_fpath_2, use_cuda=False)
print('dkm_mkpts1', len(dkm_mkpts1))
print('dkm conf MAX', np.amax(sparse_certainty), 'MIN:', np.amin(sparse_certainty))
print('dkm Num features > 0.1', (sparse_certainty >= 0.1).sum())
print('dkm Num features > 0.2', (sparse_certainty >= 0.2).sum())
print('dkm Num features > 0.3', (sparse_certainty >= 0.3).sum())

rpt_F = 0.20
conf_F = 0.999999
m_iters_F = 200_000
F, inliers_F = cv2.findFundamentalMat(dkm_mkpts1, dkm_mkpts2, cv2.USAC_MAGSAC, rpt_F, conf_F, m_iters_F)

orig_image_1 = load_torch_kornia_image(image_fpath_1, dkm_device, -1)
orig_image_2 = load_torch_kornia_image(image_fpath_2, dkm_device, -1)

fig, ax = plt.subplots(figsize=(20, 10))
draw_img_match(orig_image_1, orig_image_2, dkm_mkpts1, dkm_mkpts2, inliers_F, ax=ax)
plt.show()

print('Ready')
