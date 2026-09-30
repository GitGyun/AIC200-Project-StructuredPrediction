#!/usr/bin/env python3
"""Build the student-facing checkpoint bundle from a Chameleon checkpoint release.

The upstream release groups fine-tuned checkpoints by chameleon's internal dataset names
(``ap10k``, ``davis2017``, ...) and ships two tasks this project does not use. This script copies
the subset the mini-project needs into descriptive, student-facing folder names, ready to drop in
the shared Google Drive folder that the notebook mounts.

Usage::

    python tools/prepare_checkpoint_bundle.py \
        --source /path/to/chameleon_checkpoints \
        --dest   /path/to/AIC200-StructuredPrediction/checkpoints

    # keep every fine-tuned checkpoint instead of just the notebook defaults
    python tools/prepare_checkpoint_bundle.py --source ... --dest ... --all

Result::

    dest/
    ├── metatrained/generalist.ckpt
    └── finetuned/
        ├── animal_keypoint_detection/Giraffe.ckpt
        ├── video_object_tracking/judo.ckpt
        ├── medical_lesion_segmentation/0.ckpt
        ├── object_counting/best.ckpt
        └── cell_instance_segmentation/best.ckpt
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chameleon'))
from starter_utils import TASKS  # noqa: E402

# A few extras per task so students have something to swap in without a second download.
EXTRA_CHECKPOINTS = {
    'animal_keypoint_detection': ['Cat', 'Elephant', 'Hippo', 'Horse'],
    'video_object_tracking': ['india', 'kite-surf', 'libby', 'motocross-jump'],
    'medical_lesion_segmentation': [],
    'object_counting': [],
    'cell_instance_segmentation': [],
}


def source_dir(source, task):
    """Find the task's folder in the source release, under either naming scheme."""
    for folder in (TASKS[task]['dataset'], task):
        path = os.path.join(source, 'finetuned', folder)
        if os.path.isdir(path):
            return path
        path = os.path.join(source, 'vtmv2', folder)
        if os.path.isdir(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', required=True, help='directory holding the checkpoint release')
    parser.add_argument('--dest', required=True, help='bundle directory to create')
    parser.add_argument('--all', action='store_true',
                        help='copy every fine-tuned checkpoint, not just the curated subset')
    parser.add_argument('--dry-run', action='store_true', help='print what would be copied')
    args = parser.parse_args()

    copied = skipped = 0

    # ---- meta-trained ----
    mt_src = os.path.join(args.source, 'metatrained', 'generalist.ckpt')
    if not os.path.exists(mt_src):
        sys.exit(f'error: meta-trained checkpoint not found at {mt_src}')
    mt_dst = os.path.join(args.dest, 'metatrained', 'generalist.ckpt')
    size_gb = os.path.getsize(mt_src) / 1024 ** 3
    print(f'metatrained/generalist.ckpt  ({size_gb:.2f} GB)')
    if not args.dry_run:
        os.makedirs(os.path.dirname(mt_dst), exist_ok=True)
        shutil.copy2(mt_src, mt_dst)
    copied += 1

    # ---- fine-tuned ----
    for task, spec in TASKS.items():
        src = source_dir(args.source, task)
        if src is None:
            print(f'  ! no source folder for {task}; skipping')
            skipped += 1
            continue

        available = sorted(os.path.splitext(f)[0] for f in os.listdir(src) if f.endswith('.ckpt'))
        if args.all:
            wanted = available
        else:
            wanted = [spec['default_ckpt']] + EXTRA_CHECKPOINTS.get(task, [])
            wanted = [n for n in wanted if n in available]
            if spec['default_ckpt'] not in available:
                print(f'  ! default checkpoint "{spec["default_ckpt"]}" missing from {src}')
                skipped += 1
                continue

        dst_dir = os.path.join(args.dest, 'finetuned', task)
        print(f'{task}/  <- {os.path.relpath(src, args.source)}  ({len(wanted)} file(s))')
        for name in wanted:
            print(f'    {name}.ckpt')
            if not args.dry_run:
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy2(os.path.join(src, f'{name}.ckpt'),
                             os.path.join(dst_dir, f'{name}.ckpt'))
            copied += 1

    verb = 'would copy' if args.dry_run else 'copied'
    print(f'\n{verb} {copied} file(s); {skipped} task(s) skipped.')
    if not args.dry_run:
        print(f'bundle ready at {args.dest}')
        print('Upload it to the shared Google Drive folder and point CHECKPOINT_ROOT at it.')


if __name__ == '__main__':
    main()
