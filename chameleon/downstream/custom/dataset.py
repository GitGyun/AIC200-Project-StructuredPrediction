"""A folder-based dataset template for your own structured-prediction task.

Chameleon's whole data interface is three tensors per example:

==========  =================  ==================================================================
tensor      shape              meaning
==========  =================  ==================================================================
``X``       ``(3M, H, W)``     ``M`` input images (modalities) stacked along channels, in ``[0, 1]``
``Y``       ``(C, H, W)``      the dense label, float in ``[0, 1]``, one channel per output map
``M``       ``(C, H, W)``      1 where ``Y`` is a valid supervision target, 0 where it is unknown
==========  =================  ==================================================================

``C`` is the ``n_tasks`` you pass to ``register_custom``. ``M`` is 1 for a plain RGB image; the
released weights accept up to ``M = 2`` (e.g. RGB + a second modality), each modality repeated to 3
channels if it is single-channel, because every image goes through the same RGB-pretrained backbone. Everything Chameleon can do -- masks,
keypoints, depth, density, flow -- is just a different way of packing information into ``Y``.

Expected directory layout (``path_dict['custom']`` in ``data_paths.yaml``)::

    CUSTOM_DATA_ROOT/
    ├── train/
    │   ├── images/  000.jpg 001.jpg ...
    │   └── labels/  000.png 001.png ...   # same stem as the image
    └── test/
        ├── images/
        └── labels/

The ``train`` split doubles as the **support set**: the first ``shot`` files (ordered by name,
offset by ``support_idx * shot``) are the examples Chameleon conditions on.

To adapt this to your own labels, subclass it and override :meth:`load_label`. That is usually the
only method you need to touch.
"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, Resize, ToTensor

from dataset.utils import crop_arrays

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')


class CustomDenseDataset(Dataset):
    """Reads ``images/`` + ``labels/`` folders and yields ``(X, Y, M)`` episodes."""

    def __init__(self, config, split, base_size, crop_size, eval_mode=False, resize=False, dset_size=-1):
        super().__init__()
        self.config = config
        self.base_size = tuple(base_size)
        self.img_size = tuple(crop_size)
        self.eval_mode = eval_mode
        self.resize = resize
        self.shot = config.shot
        self.support_idx = config.support_idx
        self.precision = config.precision
        self.n_channels = config.n_tasks

        data_root = config.path_dict[config.dataset]
        # 'valid' reuses the training folder: it is the held-out part of the labelled pool.
        split_dir = 'train' if split in ('train', 'valid') else 'test'
        self.image_dir = os.path.join(data_root, split_dir, 'images')
        self.label_dir = os.path.join(data_root, split_dir, 'labels')
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(
                f'No image folder at {self.image_dir}. Check path_dict["custom"] in data_paths.yaml '
                f'and the layout documented in downstream/custom/dataset.py.'
            )

        stems = sorted(
            os.path.splitext(f)[0] for f in os.listdir(self.image_dir)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        )
        if len(stems) == 0:
            raise FileNotFoundError(f'No images found in {self.image_dir}.')

        if split == 'train':
            # the support set: `shot` consecutive files starting at `support_idx * shot`
            start = self.support_idx * self.shot
            selected = stems[start:start + self.shot]
            if len(selected) < self.shot:
                selected = selected + stems[:self.shot - len(selected)]  # wrap around
            self.stems = selected
        elif split == 'valid':
            # everything the support set did not consume
            support = set(stems[self.support_idx * self.shot:(self.support_idx + 1) * self.shot])
            self.stems = [s for s in stems if s not in support] or stems
        else:
            self.stems = stems

        self.toten = ToTensor()
        self.base_resizer = Resize(self.base_size)
        self.resizer = Resize(self.img_size)
        self.jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        self.randomflip = getattr(config, 'randomflip', False)
        self.randomjitter = getattr(config, 'randomjitter', False)

        if dset_size < 0:
            self.dset_size = len(self.stems)
        elif not eval_mode:
            self.dset_size = dset_size           # training loops sample with replacement
        else:
            self.dset_size = min(dset_size, len(self.stems))

    def __len__(self):
        return self.dset_size

    # ------------------------------------------------------------------ loading

    def _find(self, directory, stem):
        for ext in IMAGE_EXTENSIONS + ('.npy',):
            path = os.path.join(directory, stem + ext)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f'No file for stem "{stem}" in {directory}')

    def load_image(self, stem):
        """Return the input as a float tensor ``(3M, H, W)`` in ``[0, 1]``.

        The default reads one RGB image (``M = 1``). For two modalities, override this to return
        their channel-wise concatenation, ``(6, H, W)``.
        """
        image = Image.open(self._find(self.image_dir, stem)).convert('RGB')
        return self.toten(image)

    def load_label(self, stem):
        """Return ``(Y, M)``, both ``(C, H, W)`` float tensors in ``[0, 1]``.

        **Override this method for your own task.** The default handles the two most common cases:

        * a single-channel binary or grayscale mask image -> ``C = 1``;
        * a ``.npy`` array of shape ``(C, H, W)`` or ``(H, W)``.

        Some recipes for other label types:

        * *multi-class segmentation* (``C`` classes): one-hot the index map, drop the background
          channel, so ``Y[c]`` is the binary mask of class ``c``.
        * *keypoints* (``C`` joints): draw a small Gaussian blob at each joint into ``Y[c]``; set
          ``M[c] = 0`` for joints that are not visible in this image.
        * *counting*: a single density map that sums to the object count, rescaled to ``[0, 1]``.
        * *continuous maps* (depth, normals): normalise to ``[0, 1]`` with a fixed, documented
          range, and set ``M = 0`` wherever the sensor gave no reading.
        """
        path = self._find(self.label_dir, stem)
        if path.endswith('.npy'):
            label = torch.from_numpy(np.load(path).astype(np.float32))
            if label.ndim == 2:
                label = label[None]
        else:
            label = self.toten(Image.open(path).convert('L'))
        if label.max() > 1:
            label = label / 255.0
        if label.shape[0] != self.n_channels:
            raise ValueError(
                f'Label for "{stem}" has {label.shape[0]} channel(s) but n_tasks={self.n_channels}. '
                f'Override load_label(), or re-register with the right n_tasks.'
            )
        mask = torch.ones_like(label)
        return label, mask

    # ------------------------------------------------------------- augmentation

    def postprocess(self, X, Y, M):
        if not self.eval_mode:
            if self.randomjitter and torch.rand(1).item() > 0.5:
                X = self.jitter(X)
            if self.randomflip and torch.rand(1).item() > 0.5:
                X, Y, M = (torch.flip(t, dims=[-1]) for t in (X, Y, M))

        if self.resize:
            # support examples: squash the whole image into the model's input size
            X, Y, M = self.resizer(X), self.resizer(Y), self.resizer(M)
        else:
            # queries and training crops: take a crop so nothing is distorted
            X, Y, M = crop_arrays(
                X, Y, M,
                base_size=X.size()[-2:],
                crop_size=self.img_size,
                random=(not self.eval_mode),
            )

        if self.precision == 'bf16':
            X, Y, M = X.to(torch.bfloat16), Y.to(torch.bfloat16), M.to(torch.bfloat16)
        return X, Y, M

    def __getitem__(self, idx):
        stem = self.stems[idx % len(self.stems)]
        X = self.load_image(stem)
        Y, M = self.load_label(stem)

        # bring image and label to a common resolution before cropping
        X = self.base_resizer(X)
        if Y.shape[-2:] != X.shape[-2:]:
            Y = Resize(X.shape[-2:], interpolation=Image.NEAREST)(Y)
            M = Resize(X.shape[-2:], interpolation=Image.NEAREST)(M)

        X, Y, M = self.postprocess(X, Y, M)
        if self.eval_mode:
            return X, Y, M, torch.zeros(1)  # aux slot; unused by CustomLearner
        return X, Y, M
