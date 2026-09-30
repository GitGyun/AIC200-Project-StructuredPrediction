# Changes to the vendored Chameleon code

This directory is a copy of the [official Chameleon repository](https://github.com/GitGyun/chameleon)
(ECCV 2024 oral). Everything below is a change we made for the AIC200 mini-project. Each one is
marked in the source with a `[AIC200 PATCH]` comment, so you can find them with:

```bash
grep -rn "AIC200 PATCH" chameleon/
```

Nothing else was modified. The model, the matching module, the encoders, the learners and the
dataset classes are all upstream code.

---

## 1. `train/optim.py`, `train/trainer.py` — DeepSpeed is now an optional import

Upstream imports `deepspeed` unconditionally. DeepSpeed is only used by **stage 0 (meta-training)**,
which this project never runs, and installing it on Colab means a multi-minute source build.

Both imports are now wrapped in `try/except ImportError`. In `train/optim.py`,
`optim_dict['cpuadam']` becomes `None` when DeepSpeed is absent — selecting `--optimizer cpuadam`
without DeepSpeed installed will fail, which is correct. The single use site in `train/trainer.py`
(`deepspeed.comm.barrier()`) is already behind an `n_devices > 1` guard.

## 2. `args.py` — configs resolve relative to this package

Upstream builds config paths relative to the current working directory
(`'downstream/ap10k/configs/train_config.yaml'`), which only works if you launch `main.py` from the
repository root. The notebook imports `args` from wherever Colab happens to be, so paths are now
resolved against `PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))`.

`data_paths.yaml` is still looked up in the working directory first, then in the package root, so
existing command-line workflows behave exactly as before.

## 3. `args.py`, `downstream/learner_factory.py`, `model/model_factory.py` — a `custom` task

Adds `custom` to `DOWNSTREAM_DATASETS` and `DOWNSTREAM_TASKS`, adds a `--n_tasks` argument, and
teaches both factories to dispatch it. The Dataset class, Learner class and label-channel count come from
`downstream/custom/registry.py`, so a student can register their own classes from a notebook cell
instead of editing the library:

```python
from downstream.custom import register_custom, CustomDenseDataset, CustomLearner

class MyDataset(CustomDenseDataset):
    def load_label(self, stem):
        ...

register_custom(dataset_cls=MyDataset, learner_cls=CustomLearner, n_tasks=1)
```

The label-channel count is resolved as `--n_tasks` (or `n_tasks=` in `build_config`) first, then
the value given to `register_custom`. The custom YAML configs intentionally do not set it, since a
YAML default would silently win over the registry.

## 4. `downstream/custom/` — new

A folder-based `Dataset` and a `Learner` template for student-defined tasks, plus `train_config.yaml`
and `test_config.yaml`. Not part of upstream.

## 5. `starter_utils.py` — new

Notebook plumbing: task registry, checkpoint discovery, config building without an `experiments/`
tree, and matplotlib rendering of predictions. Not part of upstream.

### 5b. `task_vis.py` — new

The notebook's per-task figures: input channels, then the raw dense label / prediction
(heatmaps, soft masks, densities, flows), then the decoded task output (skeletons, mask overlays,
counted points with the count, cell boundaries). It reuses upstream drawing and decoding tools
(`vis_animal_keypoints`, `postprocess_semseg`, `preprocess_kpmap`, `flow_vis`, `compute_masks`).
`starter_utils.predict_queries` feeds it the pre-`postprocess_final` outputs and reports the peak
GPU memory of support encoding and inference against the T4 budget (14.7 GB).

## 6. `requirements-colab.txt` — new

The Colab-installable subset of `requirements.txt`. Drops:

| dropped | why |
| --- | --- |
| `lightning_habana` | Intel Gaudi only; irrelevant on a GPU runtime |
| `plyfile` | only read LINEMOD's 3-D meshes (see patch 11) |
| `deepspeed` | stage-0 meta-training only (see patch 1) |
| `torch==2.3.1` pin | Colab ships its own CUDA-matched torch; re-pinning forces a slow reinstall |
| `xtcocotools` | no Python 3.12 wheel; only needed for keypoint-AP evaluation (patch 13) |

Changed: `timm==0.6.12` → `timm==0.6.13`. 0.6.12 fails to import on Python ≥ 3.11 (a mutable dataclass
default in `timm.models.maxxvit`); 0.6.13 is the release that fixes it, with the same model code.
The notebook's pip cell also pins NumPy to the version the runtime has already loaded, so no
dependency can upgrade it under a running kernel.

`requirements.txt` is left untouched for anyone reproducing the paper.

## 7. `dataset/meta_info/` — pruned

Upstream ships ~135 MB of dataset index files. The ones for the **meta-training** corpora
(`taskonomy`, `coco`, `midair`, `openimages`, `deepfashion`, `freihand`, `mpii`, `nyud`, `sintel`,
`kitti`) are only read by stage 0 and have been removed so the repository clones quickly in Colab.

Kept: `ap10k/`, `cellpose/`, `davis2017/` — the indices the downstream tasks actually load.
They are `.pth` files, so the repository `.gitignore` has an explicit exception for
`chameleon/dataset/meta_info/**/*.pth`; without it they silently drop out of the repo.

These indices matter for reproducibility: `ap10k/*_class_dict_skip_crowd.pth` fixes which
annotations belong to each species (and in which order), and `cellpose/train_idxs_perm.pth` fixes
which training images form the support set. If they are missing the loaders regenerate them with a
fresh random shuffle, and the support set no longer matches the one the released checkpoints were
fine-tuned with.

To run meta-training, clone the upstream repository instead.

## 8. `downstream/ap10k/dataset.py`, `downstream/cellpose/dataset.py` — `meta_info` resolves relative to this package

Upstream opens `dataset/meta_info/...` relative to the current working directory, the same problem
as patch 2. From a notebook whose working directory is not `chameleon/`, AP-10K silently
regenerated its class index in the wrong place, and Cellpose crashed trying to `torch.save` into a
directory that does not exist. Both paths are now joined onto a module-level `PACKAGE_ROOT`.

## 9. `downstream/cellpose/dataset.py` — flows cast to float32

Cellpose's precomputed `NNN_flows.npy` are float64. Upstream always runs in bf16, and
`prepare_support_data` casts the support set to bf16, which hides this. On a T4 we run in fp32,
where nothing casts, so the float64 flows reached the label encoder's first conv and crashed
(`Input type (double) and bias type (float) should be the same`). The loader now casts them to
float32 as it reads them.

## 10. `downstream/demo.py`, `downstream/learner_factory.py` — the flat demo-data layout

New. The demo subsets use one layout for every task, `<dataset>/images/N.*` + `<dataset>/labels/N.*`
with no train/test split (the first `shot` files are the support set). `demo.py` subclasses each
official Dataset, overriding only file discovery and loading, and reuses the official
preprocessing unchanged. `get_downstream_learner` returns the matching demo learner whenever the
dataset folder has that layout, so full datasets are unaffected. The AP-10K demo learner disables
the COCO evaluator, which needs the full annotation JSON.

## 11. LINEMOD removed

The project uses five of Chameleon's six downstream tasks; LINEMOD (6-DoF pose) is excluded, so
its code is gone: `downstream/linemod/`, `scripts/linemod/`, the 6-DoF pose section of
`dataset/utils.py` (mesh loading, PnP, ADD score), the `linemod` branches in `learner_factory.py`
and `model_factory.py`, `linemod` / `pose_6d` / `segment_semantic` from the `args.py` choices, and
the LINEMOD-only `--coord_path` argument (plus its `coord_path: none` line in every task's YAML).
That also removes the only use of `plyfile`, which is dropped from both requirements files. To
reproduce the LINEMOD results, use the upstream repository.

## 12. `downstream/base_learner.py`, `args.py` — fine-tuning without validation

The notebook fine-tunes with a plain loop and never validates. `starter_utils.build_config`
therefore always passes `--no_eval` for stage 1, which (upstream behaviour) skips building the
validation support set and sizes the training loader by `n_steps` instead of `val_iter`. The
remaining tie was the learners' evaluators, registered unconditionally in `BaseLearner.__init__`:
AP-10K's opens `ap10k-val-split1.json`. They are now skipped when `stage == 1 and no_eval`, so
fine-tuning needs no validation data at all (checked on full AP-10K with only the training
annotations present). `n_schedule_steps` defaults to `n_steps` in stages 0 and 1, so the notebook
sets neither `val_iter` nor `n_schedule_steps`. The Lightning validation hooks in `train/trainer.py`
are still there for `main.py`, but the notebook does not use them.

## 13. `dataset/xtcoco_api_wrapper.py` — `xtcocotools` is optional

`xtcocotools` has no wheel for Python 3.12, which Colab now runs, and its source build fails inside
pip's isolated build environment (`setup.py egg_info` imports numpy and Cython). It is only used
for COCO keypoint-AP evaluation (`SilentXTCOCO`, `SilentXTCOCOeval`). Those two classes moved from
`dataset/coco_api_wrapper.py` to the new `dataset/xtcoco_api_wrapper.py`, and `AP10KLearner` imports its
evaluator inside `register_evaluator` instead of at module level. Annotation loading
(`SilentCOCO`) uses `pycocotools` as before. `xtcocotools` is dropped from `requirements-colab.txt`
and the notebook's pip cell; to compute AP-10K AP, install it with
`pip install cython && pip install --no-build-isolation xtcocotools`.

For the same reason `train/trainer.py` now imports `MetaTrainLearner` inside `setup()`, only for stage
0: `meta_train/learner.py` imports the keypoint evaluator (`xtcocotools`) and, through
`meta_train/evaluator.py`, an unguarded `import deepspeed` that patch 1 did not cover.

## 14. `torch.load(..., weights_only=False)` for the released files

PyTorch 2.6 changed `torch.load` to default to `weights_only=True`, which refuses the released
checkpoints (their config is an `EasyDict`) and the `meta_info` indices (NumPy arrays). Colab runs
a newer PyTorch, so every load on the notebook's path now passes `weights_only=False` explicitly:
`train/train_utils.py` (checkpoints), `starter_utils.py`, `downstream/ap10k/dataset.py`,
`downstream/cellpose/dataset.py`, `downstream/isic2018/dataset.py`, `model/transformers/helpers.py`
(and `tools/extract_demo_data.py`). These files come from this repository or the course's
checkpoint bundle; only load checkpoints this way if you trust their source. Stage-0-only loaders
(`dataset/base.py`, `meta_train/`, `preprocess_checkpoints.py`) are unchanged.
