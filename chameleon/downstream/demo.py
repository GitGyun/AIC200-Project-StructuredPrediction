"""[AIC200 PATCH] Loaders for the flat demo-data layout (see chameleon/PATCHES.md §10).

The full datasets each have their own folder structure and split files. The demo subsets shipped
with the starter notebook use one layout for every task instead::

    <DATA_ROOT>/<dataset>/
    ├── images/   0.jpg 1.jpg ... N.jpg
    └── labels/   0.png 1.png ... N.png     (or .npy)

There are no train/test folders. The **support set** is ``shot`` consecutive files starting at
``support_idx * shot`` (so the first ``shot`` files by default) -- that is the ``train`` split,
used both as the support set and as the fine-tuning pool. Every other file is a **query** (the
``test`` / ``valid`` split).

Label formats:

=============  =========================  ======================================================
dataset        ``labels/N.*``             notes
=============  =========================  ======================================================
ap10k          ``.npy`` int ``(17, 3)``   ``(x, y, visibility)`` per joint, in pixels of image N;
                                          visibility 2 = visible
davis2017      ``.png`` palette           pixel value = instance id (0 = background)
isic2018       ``.png``                   binary lesion mask
fsc147         ``.npy`` ``(H, W)``        density map; sums to the object count. The exemplar
                                          boxes (an *input*, not a label) are in
                                          ``exemplars.json``: ``{"N": [box, box, box]}``
cellpose       ``.npy`` ``(3, H, W)``     two flow channels + the instance mask
=============  =========================  ======================================================

Each class subclasses the official Dataset and reuses its preprocessing (augmentation, cropping,
label encoding) unchanged -- only file discovery and loading differ. ``learner_factory`` switches
to these automatically when a dataset folder has this layout, so the full datasets keep working.
"""

import json
import os

import numpy as np
import torch
from einops import repeat
from PIL import Image
from torchvision.transforms import ColorJitter, RandomRotation, Resize, ToTensor

from .ap10k.dataset import AP10KDataset
from .ap10k.learner import AP10KLearner
from .cellpose.dataset import CELLPOSEDataset
from .cellpose.learner import CELLPOSELearner
from .davis2017.dataset import DAVIS2017Dataset
from .davis2017.learner import DAVIS2017Learner
from .fsc147.dataset import FSC147Dataset
from .fsc147.learner import FSC147Learner
from .isic2018.dataset import ISIC2018Dataset
from .isic2018.learner import ISIC2018Learner


def is_flat_layout(data_root):
    return (os.path.isdir(os.path.join(data_root, 'images'))
            and os.path.isdir(os.path.join(data_root, 'labels')))


class FlatDemoMixin:
    """File discovery and the support/query split shared by every demo dataset."""

    def _setup_flat(self, config, split, base_size, crop_size, eval_mode, resize, dset_size):
        root = config.path_dict[config.dataset]
        self.image_dir = os.path.join(root, 'images')
        self.label_dir = os.path.join(root, 'labels')
        self.data_root = root
        self.base_size = base_size
        self.img_size = crop_size
        self.eval_mode = eval_mode
        self.resize = resize
        self.shot = config.shot
        self.support_idx = config.support_idx
        self.precision = config.precision
        self.toten = ToTensor()
        self.resizer = Resize(self.img_size)

        stems = sorted((os.path.splitext(f)[0] for f in os.listdir(self.image_dir)), key=int)
        start = self.support_idx * self.shot
        support = stems[start:start + self.shot]
        if split == 'train':
            if len(support) < self.shot:
                print(f'[demo data] only {len(support)} support files for shot={self.shot} '
                      f'(support_idx={self.support_idx}); using those')
            self.data_idxs = support
        else:
            self.data_idxs = [s for s in stems if s not in set(support)]
        if len(self.data_idxs) == 0:
            raise ValueError(f'no {split} files in {root} for shot={self.shot}, '
                             f'support_idx={self.support_idx} ({len(stems)} files in total)')

        if dset_size < 0:
            self.dset_size = len(self.data_idxs)
        elif not eval_mode:
            self.dset_size = dset_size
        else:
            self.dset_size = min(dset_size, len(self.data_idxs))

    def _path(self, directory, stem):
        for name in os.listdir(directory):
            if os.path.splitext(name)[0] == stem:
                return os.path.join(directory, name)
        raise FileNotFoundError(f'no file "{stem}.*" in {directory}')

    def _stem(self, idx):
        return self.data_idxs[idx % len(self.data_idxs)]

    def __len__(self):
        return self.dset_size


class AP10KDemoDataset(FlatDemoMixin, AP10KDataset):
    def __init__(self, config, split, base_size, crop_size, eval_mode=False, resize=False, dset_size=-1):
        self._setup_flat(config, split, base_size, crop_size, eval_mode, resize, dset_size)
        self.gaussian = self._make_gaussian_kernel(3, 3, 1.)
        self.kp_radius = 9
        self.randomflip = config.randomflip
        self.randomjitter = config.randomjitter
        self.randomrotate = config.randomrotate
        self.jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        self.rotate = RandomRotation(30)
        self.base_resizer = Resize(self.base_size)

    def _load(self, stem):
        img = Image.open(self._path(self.image_dir, stem))
        W, H = img.size
        X = self.base_resizer(self.toten(img))
        if len(X) == 1:
            X = X.repeat(3, 1, 1)
        keypoints = np.load(self._path(self.label_dir, stem)).astype(float)
        keypoints[:, 0] *= self.base_size[1] / W
        keypoints[:, 1] *= self.base_size[0] / H
        keypoints = torch.from_numpy(keypoints.astype(int))
        Y = self._sparse_to_dense(keypoints.numpy(), self.gaussian, self.kp_radius)
        M = repeat((keypoints[..., 2] == 2).float(), 'c -> c h w', h=self.base_size[0], w=self.base_size[1])
        return X, Y, M, keypoints

    def __getitem__(self, idx):
        stem = self._stem(idx)
        X, Y, M, keypoints = self._postprocess_data(*self._load(stem))
        if self.eval_mode:
            # (keypoints, (image id, annotation id)) as in AP10KDataset; the stem stands in for both
            return X, Y, M, (keypoints, (int(stem), int(stem)))
        return X, Y, M


class DAVIS2017DemoDataset(FlatDemoMixin, DAVIS2017Dataset):
    """One video sequence: frames in order, ``config.class_name`` names the sequence."""

    def __init__(self, config, split, base_size, crop_size, eval_mode=False, resize=False, dset_size=-1):
        assert config.class_name in self.CLASS_NAMES
        self._setup_flat(config, split, base_size, crop_size, eval_mode, resize, dset_size)
        self.n_channels = self.NUM_INSTANCES[self.CLASS_NAMES.index(config.class_name)] + 1
        self.randomscale = config.randomscale

    def __getitem__(self, idx):
        stem = self._stem(idx)
        size = (self.base_size[1], self.base_size[0])
        image = Image.open(self._path(self.image_dir, stem)).convert('RGB').resize(size, Image.BILINEAR)
        label = Image.open(self._path(self.label_dir, stem)).resize(size, Image.NEAREST)
        return self.postprocess_data(image, label)


class ISIC2018DemoDataset(FlatDemoMixin, ISIC2018Dataset):
    def __init__(self, config, split, base_size, crop_size, eval_mode=False, resize=False, dset_size=-1):
        self._setup_flat(config, split, base_size, crop_size, eval_mode, resize, dset_size)

    def __getitem__(self, idx):
        stem = self._stem(idx)
        image = Image.open(self._path(self.image_dir, stem)).convert('RGB')
        label = Image.open(self._path(self.label_dir, stem))
        if not self.eval_mode:
            # the full dataset reads pre-resized `resized_<base>` copies in training mode
            size = (self.base_size[1], self.base_size[0])
            image, label = image.resize(size, Image.BILINEAR), label.resize(size, Image.NEAREST)
        return self.postprocess_data(image, label)


class FSC147DemoDataset(FlatDemoMixin, FSC147Dataset):
    def __init__(self, config, split, base_size, crop_size, eval_mode=False, resize=False, dset_size=-1):
        self._setup_flat(config, split, base_size, crop_size, eval_mode, resize, dset_size)
        with open(os.path.join(self.data_root, 'exemplars.json')) as f:
            self.exemplars = json.load(f)
        self.cnt = 200
        self.base_resizer = Resize(base_size)
        self.randomflip = config.randomflip
        self.randomjitter = config.randomjitter
        self.jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)

    def __getitem__(self, idx):
        stem = self._stem(idx)
        image, label, image_size = self._load_data(self._path(self.image_dir, stem),
                                                   self._path(self.label_dir, stem))
        meta = {'box_examples_coordinates': self.exemplars[stem]}
        return self.postprocess_data(image, label, image_size, meta)


class CELLPOSEDemoDataset(FlatDemoMixin, CELLPOSEDataset):
    def __init__(self, config, split, base_size, crop_size, eval_mode=False, resize=False, dset_size=-1):
        self._setup_flat(config, split, base_size, crop_size, eval_mode, resize, dset_size)
        self.base_resizer = Resize(base_size)
        self.randomflip = config.randomflip
        self.randomjitter = config.randomjitter
        self.randomrotate = config.randomrotate
        self.jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        self.rotate = RandomRotation(30)
        self.max_h = 576
        self.max_w = 720

    def __getitem__(self, idx):
        stem = self._stem(idx)
        image = self.toten(Image.open(self._path(self.image_dir, stem)).convert('RGB'))
        image = torch.stack([image[1], image[0]])  # (cytoplasm, nuclei), as CELLPOSEDataset does
        label = torch.from_numpy(np.load(self._path(self.label_dir, stem))).float()
        flow, mask = label[:2], label[2:3]
        return self.postprocess_data(image, flow, mask)


class AP10KDemoLearner(AP10KLearner):
    BaseDataset = AP10KDemoDataset

    def register_evaluator(self):
        # the COCO keypoint evaluator needs the full annotation JSON, which the demo data omits
        self.kp_classes = AP10KDataset.CLASS_NAMES
        self.evaluator = {}

    def reset_evaluator(self):
        pass


class DAVIS2017DemoLearner(DAVIS2017Learner):
    BaseDataset = DAVIS2017DemoDataset


class ISIC2018DemoLearner(ISIC2018Learner):
    BaseDataset = ISIC2018DemoDataset


class FSC147DemoLearner(FSC147Learner):
    BaseDataset = FSC147DemoDataset


class CELLPOSEDemoLearner(CELLPOSELearner):
    BaseDataset = CELLPOSEDemoDataset


DEMO_LEARNERS = {
    'ap10k': AP10KDemoLearner,
    'davis2017': DAVIS2017DemoLearner,
    'isic2018': ISIC2018DemoLearner,
    'fsc147': FSC147DemoLearner,
    'cellpose': CELLPOSEDemoLearner,
}

DEMO_DATASETS = {name: learner.BaseDataset for name, learner in DEMO_LEARNERS.items()}


def get_dataset_cls(dataset, data_root):
    """The demo Dataset for a flat-layout folder, else the official one it subclasses."""
    learner = DEMO_LEARNERS[dataset]
    return learner.BaseDataset if is_flat_layout(data_root) else learner.__base__.BaseDataset
