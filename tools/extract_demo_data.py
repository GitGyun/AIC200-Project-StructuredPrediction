"""Copy a small demo subset of each downstream dataset into the flat demo layout.

    python tools/extract_demo_data.py \
        --data_paths /path/to/full/data_paths.yaml \
        --dest       /path/to/demo_data

Every dataset gets the same layout, with no train/test folders (read by chameleon/downstream/demo.py):

    <dataset>/
    ├── images/   0.jpg 1.jpg ... N.jpg      (Cellpose: .png -- lossless microscopy)
    └── labels/   0.png ... N.png  or  0.npy ... N.npy

Files 0 .. n_support-1 are the support set and the rest are queries; the loaders take the first
`shot` files as support. The support images are chosen with each official loader's own selection
logic, so they are (a prefix of) the support set each released checkpoint was fine-tuned with.
Source datasets are only read; nothing is moved.

    ap10k      labels/N.npy  int (17, 3) keypoints (x, y, visibility) in pixels of images/N.jpg
    davis2017  labels/N.png  instance ids; one sequence, frame order preserved
    isic2018   labels/N.png  binary lesion mask (original resolution)
    fsc147     labels/N.npy  density map; exemplar boxes in exemplars.json {"N": [box x3]}
    cellpose   labels/N.npy  float32 (3, H, W): two flow channels + the instance mask
"""

import argparse
import json
import os
import shutil
import sys
from itertools import zip_longest

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.transforms import ToTensor

CHAMELEON_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'chameleon')
META_INFO = os.path.join(CHAMELEON_ROOT, 'dataset', 'meta_info')


class Writer:
    """Writes numbered image/label pairs: images/N.<ext>, labels/N.<ext>."""

    def __init__(self, dst):
        self.dst = dst
        self.n = 0
        for sub in ('images', 'labels'):
            os.makedirs(os.path.join(dst, sub), exist_ok=True)

    def add(self, image_src, label):
        """`label` is a source file path to copy, or an array to save as .npy."""
        n = str(self.n)
        shutil.copy2(image_src, os.path.join(self.dst, 'images', n + os.path.splitext(image_src)[1]))
        if isinstance(label, str):
            shutil.copy2(label, os.path.join(self.dst, 'labels', n + os.path.splitext(label)[1]))
        else:
            np.save(os.path.join(self.dst, 'labels', n + '.npy'), label)
        self.n += 1
        return n


# ---------------------------------------------------------------------------------------------
# AP-10K
# ---------------------------------------------------------------------------------------------

def extract_ap10k(src, dst, species, shot, n_query, base_size):
    """The official support set (AP10KDataset picks it by keypoint coverage over the species'
    training annotations) and the first test queries, with keypoints in image pixels."""
    sys.path.insert(0, CHAMELEON_ROOT)
    from easydict import EasyDict
    from downstream.ap10k.dataset import AP10KDataset

    cfg = EasyDict(class_name=species, shot=shot, support_idx=0, precision='fp32',
                   dataset='ap10k', path_dict={'ap10k': src},
                   randomflip=False, randomjitter=False, randomrotate=False)
    bs = [base_size, base_size]
    support = AP10KDataset(cfg, 'train', bs, bs, eval_mode=True, resize=True, dset_size=shot)
    query = AP10KDataset(cfg, 'test', bs, bs, eval_mode=True, resize=False, dset_size=n_query)

    out = Writer(dst)
    for ds, idxs in ((support, support.data_idxs[:shot]), (query, query.data_idxs[:n_query])):
        for img_id, ann_id in idxs:
            # _load_keypoint maps the annotation into the pixel frame of the cropped image
            keypoints = ds._load_keypoint(img_id, ann_id).numpy().astype(np.int64)
            out.add(os.path.join(ds.image_dir, f'{img_id}_{ann_id}.jpg'), keypoints)
    return f'{species}: {shot} support + {n_query} query'


# ---------------------------------------------------------------------------------------------
# DAVIS-2017
# ---------------------------------------------------------------------------------------------

def extract_davis2017(src, dst, sequence, n_query, base_size):
    """Frame 0 (the 1-shot support) followed by evenly spaced later frames, in order."""
    folder = os.path.join(src, f'resized_{base_size}', sequence)
    frames = sorted(os.listdir(os.path.join(folder, 'images')))
    keep = np.linspace(0, len(frames) - 1, n_query + 1).round().astype(int)
    out = Writer(dst)
    for i in sorted(set(keep.tolist())):
        stem = os.path.splitext(frames[i])[0]
        out.add(os.path.join(folder, 'images', f'{stem}.jpg'), os.path.join(folder, 'labels', f'{stem}.png'))
    return f'{sequence}: frame 0 support + {out.n - 1} query frames (of {len(frames)})'


# ---------------------------------------------------------------------------------------------
# ISIC-2018
# ---------------------------------------------------------------------------------------------

def _isic_support(file_dict, shot, support_idx=0):
    """ISIC2018Dataset's training-file selection."""
    class_names = list(reversed(sorted(file_dict, key=lambda c: len(file_dict[c]))))  # ties as upstream
    shot_per_class = [shot // len(class_names)] * len(class_names)
    for i in range(shot % len(class_names)):
        shot_per_class[i] += 1
    return sum((file_dict[c][support_idx * n:(support_idx + 1) * n]
                for c, n in zip(class_names, shot_per_class)), [])


def extract_isic2018(src, dst, shot, n_query):
    """`shot` support files split across diagnoses as the loader does, then test files drawn
    round-robin across diagnoses."""
    train = torch.load(os.path.join(src, 'meta', 'train_files.pth'), weights_only=False)
    test = torch.load(os.path.join(src, 'meta', 'test_files.pth'), weights_only=False)
    queries = [s for bundle in zip_longest(*test.values()) for s in bundle if s is not None][:n_query]
    out = Writer(dst)
    for stem in _isic_support(train, shot) + queries:
        out.add(os.path.join(src, 'original', 'images', f'{stem}.jpg'),
                os.path.join(src, 'original', 'labels', f'{stem}.png'))
    return f'{shot} support + {len(queries)} query'


# ---------------------------------------------------------------------------------------------
# FSC-147
# ---------------------------------------------------------------------------------------------

def _fsc_round_robin(files, image_classes):
    """FSC147Dataset's ordering: one image per class in turn, classes in file order."""
    by_class = {}
    for name, cls in image_classes:
        if name in files:
            by_class.setdefault(cls, []).append(name)
    return [f for bundle in zip_longest(*by_class.values()) for f in bundle if f is not None]


def extract_fsc147(src, dst, shot, n_query):
    with open(os.path.join(src, 'ImageClasses_FSC147.txt')) as f:
        image_classes = [line.strip().split('\t') for line in f if line.strip()]
    with open(os.path.join(src, 'Train_Test_Val_FSC_147.json')) as f:
        split = json.load(f)
    with open(os.path.join(src, 'annotation_FSC147_384.json')) as f:
        meta = json.load(f)

    train = _fsc_round_robin(set(split['train']), image_classes)[:shot]
    test = _fsc_round_robin(set(split['test']), image_classes)[:n_query]
    out = Writer(dst)
    exemplars = {}
    for name in train + test:
        stem = os.path.splitext(name)[0]
        # the 384 density map is the one the loader reads in eval mode; training renormalises
        # whatever it is given, so one map per image suffices
        n = out.add(os.path.join(src, 'images_384_VarV2', name),
                    os.path.join(src, 'gt_density_map_adaptive_384_VarV2', f'{stem}.npy'))
        exemplars[n] = meta[name]['box_examples_coordinates']
    with open(os.path.join(dst, 'exemplars.json'), 'w') as f:
        json.dump(exemplars, f)
    return f'{len(train)} support + {len(test)} query'


# ---------------------------------------------------------------------------------------------
# Cellpose
# ---------------------------------------------------------------------------------------------

def extract_cellpose(src, dst, shot, n_query):
    """Support: the first `shot` entries of the shipped permutation. Queries: test 000..n-1."""
    perm = torch.load(os.path.join(META_INFO, 'cellpose', 'train_idxs_perm.pth'), weights_only=False)
    toten = ToTensor()
    out = Writer(dst)
    for split, idxs in (('train', list(perm[:shot])), ('test', range(n_query))):
        for i in idxs:
            base = os.path.join(src, split, f'{i:03d}')
            flow = np.load(f'{base}_flows.npy').astype(np.float32)             # (2, H, W)
            mask = toten(Image.open(f'{base}_masks.png')).numpy().astype(np.float32)  # (1, H, W)
            out.add(f'{base}_img.png', np.concatenate([flow, mask]))
    return f'{shot} support + {n_query} query'


# ---------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_paths', required=True, help='data_paths.yaml of the full datasets')
    parser.add_argument('--dest', required=True)
    parser.add_argument('--n_query', type=int, default=10)
    parser.add_argument('--n_support', type=int, default=10,
                        help='support examples for isic2018 / fsc147 / cellpose')
    parser.add_argument('--ap10k_species', default='Giraffe')
    parser.add_argument('--ap10k_shot', type=int, default=20,
                        help='the Step 2-4 tutorial fine-tunes with 20 shots, as the official checkpoint did')
    parser.add_argument('--davis_sequence', default='judo')
    args = parser.parse_args()

    with open(args.data_paths) as f:
        paths = yaml.safe_load(f)
    if os.path.exists(args.dest) and os.listdir(args.dest):
        sys.exit(f'{args.dest} is not empty; refusing to mix demo sets')

    jobs = [
        ('ap10k', lambda s, d: extract_ap10k(s, d, args.ap10k_species, args.ap10k_shot, args.n_query, 256)),
        ('davis2017', lambda s, d: extract_davis2017(s, d, args.davis_sequence, args.n_query, 448)),
        ('isic2018', lambda s, d: extract_isic2018(s, d, args.n_support, args.n_query)),
        ('fsc147', lambda s, d: extract_fsc147(s, d, args.n_support, args.n_query)),
        ('cellpose', lambda s, d: extract_cellpose(s, d, args.n_support, args.n_query)),
    ]
    for name, job in jobs:
        print(f'{name:<10} {job(paths[name], os.path.join(args.dest, name))}')


if __name__ == '__main__':
    main()
