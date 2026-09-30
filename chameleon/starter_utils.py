"""Helpers for the AIC200 structured-prediction starter notebook.

This module is *not* part of the official Chameleon release. It exists so the notebook can stay
readable: everything here is plumbing (building configs, finding checkpoints, drawing figures),
while the parts worth reading -- the input and label format, the fine-tuning loop, the inference
call -- live in the notebook itself.

The design goal is that a whole experiment lives inside one Colab session: no ``experiments/``
directory tree, no TensorBoard, no PyTorch Lightning ``Trainer``. Configs are built in memory and
scratch files go to a single temporary directory.
"""

import os
import tempfile

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml

# ---------------------------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------------------------
# The five downstream tasks of the Chameleon paper that this project uses (LINEMOD 6-DoF pose is
# excluded). `dataset` is the name chameleon itself uses; `default_ckpt` is the checkpoint the
# notebook demos. Other checkpoints shipped for the same task are discovered at runtime by
# `list_checkpoints`, so the notebook never goes stale when the bundle changes.

TASKS = {
    'animal_keypoint_detection': dict(
        dataset='ap10k',
        title='Animal Keypoint Detection',
        blurb='Locate 17 body joints of one animal species (AP-10K).',
        output='17 keypoint heatmaps, one per joint',
        default_ckpt='Giraffe',
    ),
    'video_object_tracking': dict(
        dataset='davis2017',
        title='Video Object Segmentation',
        blurb='Propagate a first-frame mask through the rest of a video (DAVIS-2017).',
        output='one binary mask per tracked instance',
        default_ckpt='judo',
    ),
    'medical_lesion_segmentation': dict(
        dataset='isic2018',
        title='Medical Lesion Segmentation',
        blurb='Segment skin lesions in dermoscopy images (ISIC-2018).',
        output='1 binary mask',
        default_ckpt='0',
    ),
    'object_counting': dict(
        dataset='fsc147',
        title='Exemplar-Guided Object Counting',
        blurb='Count arbitrary objects given a few exemplar boxes (FSC-147).',
        output='1 density map, whose sum is the object count',
        default_ckpt='best',
    ),
    'cell_instance_segmentation': dict(
        dataset='cellpose',
        title='Cell Instance Segmentation',
        blurb='Separate touching cells in microscopy images (Cellpose).',
        output='2 flow channels + 1 foreground mask',
        default_ckpt='best',
    ),
}

# Fine-tuned checkpoints live in `<CHECKPOINT_ROOT>/finetuned/<folder>/<name>.ckpt`. The bundle has
# been distributed under both the chameleon dataset name and the descriptive task name, so both are
# accepted; `prepare_checkpoint_bundle.py` produces the descriptive form.
_CKPT_FOLDER_ALIASES = {task: [spec['dataset'], task] for task, spec in TASKS.items()}

# Keys copied out of a fine-tuned checkpoint's own config so stage 2 reproduces how it was trained.
_INHERITED_KEYS = (
    'class_name', 'shot', 'eval_shot', 'img_size', 'base_size', 'support_idx',
    'chunk_size', 'channel_chunk_size', 'autocrop', 'autocrop_minoverlap',
    'head_tuning', 'label_decoder_tuning', 'relpos_tuning',
    'input_embed_tuning', 'output_embed_tuning', 'separate_alpha',
)


# ---------------------------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------------------------

def describe_gpu():
    """Print the GPU the notebook will run on."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            'No GPU detected. In Colab: Runtime -> Change runtime type -> T4 GPU, then reconnect.'
        )
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f'GPU    : {torch.cuda.get_device_name(0)}')
    print(f'Memory : {total_gb:.1f} GB')


def write_data_paths(data_root, extra=None, chameleon_root=None):
    """Write ``data_paths.yaml``, the file chameleon reads to locate each dataset."""
    chameleon_root = chameleon_root or os.path.dirname(os.path.abspath(__file__))
    path_dict = {name: os.path.join(data_root, name) for name in
                 ('ap10k', 'davis2017', 'isic2018', 'fsc147', 'cellpose')}
    path_dict.update(extra or {})
    out = os.path.join(chameleon_root, 'data_paths.yaml')
    with open(out, 'w') as f:
        yaml.safe_dump(path_dict, f, default_flow_style=False)
    print(f'wrote {out}')
    return path_dict


def scratch_dir(name='aic200'):
    """A throwaway directory for the bits of chameleon that insist on writing to disk."""
    path = os.path.join(tempfile.gettempdir(), name)
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------------------------
# Checkpoints and configs
# ---------------------------------------------------------------------------------------------

def find_meta_trained(checkpoint_root):
    """Path of the meta-trained (generalist) checkpoint."""
    path = os.path.join(checkpoint_root, 'metatrained', 'generalist.ckpt')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'No meta-trained checkpoint at {path}. Check that the shared course folder is added to '
            f'your Drive under its original name (README.md, Getting Started, step 4).'
        )
    return path


def _task_ckpt_dir(checkpoint_root, task):
    if task not in TASKS:
        raise KeyError(f'Unknown task "{task}". Choose from {list(TASKS)}.')
    for folder in _CKPT_FOLDER_ALIASES[task]:
        path = os.path.join(checkpoint_root, 'finetuned', folder)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        f'No checkpoint folder for task "{task}" under {os.path.join(checkpoint_root, "finetuned")}. '
        f'Looked for: {_CKPT_FOLDER_ALIASES[task]}'
    )


def list_checkpoints(checkpoint_root, task):
    """Every fine-tuned checkpoint shipped for one task, by name (no extension)."""
    directory = _task_ckpt_dir(checkpoint_root, task)
    return sorted(os.path.splitext(f)[0] for f in os.listdir(directory) if f.endswith('.ckpt'))


def find_finetuned(checkpoint_root, task, name=None):
    """Path of one fine-tuned checkpoint, e.g. task ``animal_keypoint_detection``, name ``Giraffe``."""
    directory = _task_ckpt_dir(checkpoint_root, task)
    name = name or TASKS[task]['default_ckpt']
    path = os.path.join(directory, f'{name}.ckpt')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found. Available for this task: {list_checkpoints(checkpoint_root, task)}'
        )
    return path


def read_ckpt_config(ft_ckpt_path):
    """The config that a fine-tuned checkpoint was produced with."""
    ckpt = torch.load(ft_ckpt_path, map_location='cpu', weights_only=False)
    return ckpt['hyper_parameters']['config']


def derive_stage2_overrides(ft_ckpt_path, verbose=True):
    """Build the stage-2 overrides that reproduce how a checkpoint was fine-tuned.

    Every checkpoint records the config it was produced with. Reading it back is what guarantees
    the support set is built at the same resolution and shot count that the fine-tuned biases
    expect -- rather than whatever happens to be in ``test_config.yaml``, which is only a default
    and has not always matched the distributed checkpoints.
    """
    cfg = read_ckpt_config(ft_ckpt_path)
    overrides = {}
    for key in _INHERITED_KEYS:
        value = cfg.get(key) if hasattr(cfg, 'get') else getattr(cfg, key, None)
        if value is None:
            continue
        if key in ('img_size', 'base_size') and isinstance(value, (list, tuple)):
            value = list(value)
        overrides[key] = value

    # eval_shot <= 0 means "use `shot` support examples"
    if overrides.get('eval_shot', 0) is not None and overrides.get('eval_shot', 0) <= 0:
        overrides.pop('eval_shot', None)

    if verbose:
        print(f'settings recovered from {os.path.basename(ft_ckpt_path)}:')
        for key in ('class_name', 'img_size', 'base_size', 'shot', 'eval_shot',
                    'chunk_size', 'channel_chunk_size', 'autocrop'):
            if key in overrides:
                print(f'  {key:<20} = {overrides[key]}')
    return overrides


def _as_flag(key, value):
    """Render one override as CLI tokens for ``parse_args(shell_script=...)``."""
    if isinstance(value, bool):
        return [f'--{key}', str(value)]
    if isinstance(value, (list, tuple)):
        return [f'--{key}'] + [str(v) for v in value]
    return [f'--{key}', str(value)]


def build_config(stage, dataset, result_dir=None, flags=(), **overrides):
    """Build a chameleon config in memory.

    Wraps ``args.parse_args`` so every default from the argparse definition and the task's YAML is
    present, then applies the notebook's overrides. Unlike ``main.py`` this creates no
    ``experiments/`` tree, no logger and no checkpoint callbacks -- everything stays in this
    session.

    ``flags`` names switch-style arguments that take no value (``debug_mode``, ...); everything in
    ``overrides`` is passed as ``--key value``.

    Stage 1 always runs with ``--no_eval``: the notebook's fine-tuning loop never validates, so no
    validation split, evaluator or validation support set is built, and the training loader is
    sized by ``n_steps`` alone.

    Everything runs in fp32: the released configs say ``bf16``, which a Colab T4 cannot do.

    Note: ``parse_args(shell_script=...)`` splits on single spaces, so no value may contain one.
    """
    from args import parse_args

    tokens = ['--stage', str(stage), '--dataset', dataset,
              '--precision', 'fp32', '--single_gpu']
    if stage == 1 and 'no_eval' not in flags:
        flags = tuple(flags) + ('no_eval',)
    tokens += [f'--{name}' for name in flags]
    for key, value in overrides.items():
        if value is None:
            continue
        tokens += _as_flag(key, value)

    for token in tokens:
        if ' ' in token:
            raise ValueError(f'config value "{token}" contains a space; parse_args cannot take it')

    config = parse_args(shell_script=' '.join(tokens))

    # main.py normally assigns these after building the experiment directories. We are not
    # building any, so point them at scratch space -- BaseLearner.__init__ does makedirs on
    # result_dir, and some learners write temp JSON there.
    config.result_dir = result_dir or os.path.join(scratch_dir(), f'{dataset}_stage{stage}')
    config.ckpt_dir = config.result_dir
    os.makedirs(config.result_dir, exist_ok=True)

    # single-process, no DDP: the learners then skip every all_gather path
    config.single_gpu = True
    config.strategy = 'ddp'
    config.num_workers = min(getattr(config, 'num_workers', 1), 2)
    return config


def load_chameleon(config, mt_ckpt, ft_ckpt=None, verbose=True):
    """Build the model, load weights, and prepare its support set.

    ``load_model`` merges three configs -- meta-trained, then fine-tuned, then ours -- so our
    overrides win. It is also what promotes ``n_input_images`` to 2 (the meta-trained model was
    trained with stereo inputs), which is why the model must be built through it rather than by
    hand.
    """
    from train.train_utils import load_model

    config.load_mt_path = mt_ckpt
    if ft_ckpt is not None:
        config.load_ft_path = ft_ckpt

    model, config, *_ = load_model(config, verbose=verbose)
    # the released checkpoints were trained in bf16; run everything in fp32 regardless
    config.precision = model.config.precision = 'fp32'
    model = model.float().cuda()
    # builds the learner and loads the support set (encoding happens lazily at first inference);
    # Lightning's hook signature requires a stage name
    model.setup('test' if config.stage == 2 else 'fit')
    return model, config


def finetuned_parameter_names(model):
    """The state-dict keys that stage-1 fine-tuning actually changes.

    Mirrors what ``train_utils.configure_experiment`` hands to its checkpoint plugin, which is why
    an official task checkpoint is ~20 MB instead of ~3 GB: only these tensors are stored, and
    stage 2 layers them back on top of the meta-trained weights.
    """
    config = model.config
    names = [f'model.{name}' for name in model.model.bias_parameter_names()]
    names += [f'model.matching_module.alpha.{n}'
              for n, _ in model.model.matching_module.alpha.named_parameters()]
    names += [f'model.matching_module.layernorm.{n}'
              for n, _ in model.model.matching_module.layernorm.named_parameters()]
    if getattr(config, 'head_tuning', False):
        names += [f'model.label_decoder.head.{n}'
                  for n, _ in model.model.label_decoder.head.named_parameters()]
    if getattr(config, 'label_decoder_tuning', False):
        names += [f'model.label_decoder.{n}'
                  for n, _ in model.model.label_decoder.named_parameters()]
    if getattr(config, 'relpos_tuning', False):
        names += [f'model.image_encoder.{n}' for n in model.model.image_encoder.relpos_parameter_names()]
        if getattr(model.model.image_encoder.backbone, 'pos_embed', None) is not None:
            names += ['model.image_encoder.backbone.pos_embed']
    if getattr(config, 'input_embed_tuning', False):
        names += [f'model.image_encoder.backbone.patch_embed.{n}'
                  for n, _ in model.model.image_encoder.backbone.patch_embed.named_parameters()]
    if getattr(config, 'output_embed_tuning', False):
        names += [f'model.label_encoder.backbone.patch_embed.{n}'
                  for n, _ in model.model.label_encoder.backbone.patch_embed.named_parameters()]
    return names


def save_finetuned(model, path):
    """Save a fine-tuned checkpoint in the format stage 2 expects.

    ``train_utils.load_finetuned_ckpt`` reads ``state_dict`` and ``hyper_parameters['config']``, so
    a checkpoint written here is interchangeable with the official ones -- including being readable
    by ``derive_stage2_overrides``.
    """
    names = set(finetuned_parameter_names(model))
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items() if k in names}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({'state_dict': state_dict, 'hyper_parameters': {'config': model.config}}, path)
    size_mb = os.path.getsize(path) / 1024 ** 2
    print(f'saved {len(state_dict)} tensors ({size_mb:.1f} MB) to {path}')
    return path


def free_gpu():
    """Collect garbage and empty the CUDA cache, so the next task starts from a clean slate.

    Call this *after* ``del``-ing the model in your own scope -- deleting a reference inside this
    function would not release the caller's.
    """
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        used = torch.cuda.memory_allocated() / 1024 ** 3
        print(f'GPU memory now allocated: {used:.2f} GB')


def trainable_parameter_report(model):
    """Print which parameters stage-1 fine-tuning actually updates."""
    from train.optim import get_optimizer

    optimizer, _ = get_optimizer(model.config, model.model)
    n_trainable = sum(p.numel() for g in optimizer.param_groups for p in g['params'])
    n_total = sum(p.numel() for p in model.model.parameters())
    print(f'trainable : {n_trainable:,} parameters ({100 * n_trainable / n_total:.2f}% of the model)')
    print(f'total     : {n_total:,} parameters')
    print('Chameleon fine-tunes only bias terms (BitFit), the matching temperature alpha, and one '
          'LayerNorm -- which is why each task checkpoint is ~20 MB rather than ~3 GB.')
    return optimizer


# ---------------------------------------------------------------------------------------------
# GPU memory
# ---------------------------------------------------------------------------------------------

# Usable memory of a Colab T4 as PyTorch reports it (15,101 MiB).
T4_BUDGET_GB = 14.7


def _gb(n_bytes):
    return n_bytes / 1024 ** 3


def memory_line(label, peak_bytes):
    peak = _gb(peak_bytes)
    flag = 'fits a T4' if peak <= T4_BUDGET_GB else 'EXCEEDS a T4 -- lower the memory settings'
    return f'[GPU] {label}: peak {peak:.2f} GB / {T4_BUDGET_GB} GB ({flag})'


# ---------------------------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------------------------

@torch.no_grad()
def predict_queries(model, n_query, support_verbose=False):
    """Predict the first ``n_query`` test images, one sample dict per query, on the CPU.

    Each sample holds ``X, Y, M, aux`` from the loader at native resolution and ``raw``, the
    model's dense output *before* the learner's final thresholding / argmax (``postprocess_final``)
    -- heatmaps, logits, densities, flows -- so a figure can show both the raw maps and the decoded
    task output. Peak GPU memory is reported separately for support encoding and for inference.
    """
    learner, task = model.learner, model.config.task
    final, learner.postprocess_final = learner.postprocess_final, (lambda y: y)
    samples, peaks = [], {}
    try:
        for batch in learner.get_test_loader():
            X = batch[0].cuda()
            aux = batch[3] if len(batch) > 3 else None
            if not model.model.has_encoded_support:
                torch.cuda.reset_peak_memory_stats()
                with torch.autocast('cuda', dtype=torch.float32):   # as inference() does
                    model.encode_support(X, task, support_verbose)
                peaks['support encoding'] = torch.cuda.max_memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            raw = model.inference(X, task).float().cpu()
            peaks['inference'] = max(peaks.get('inference', 0), torch.cuda.max_memory_allocated())
            for i in range(len(X)):
                samples.append(dict(X=batch[0][i].float(), Y=batch[1][i].float(), M=batch[2][i].float(),
                                    aux=_index_aux(aux, i), raw=raw[i]))
            if len(samples) >= n_query:
                break
    finally:
        learner.postprocess_final = final
    for label, peak in peaks.items():
        print(memory_line(label, peak))
    return samples[:n_query]


@torch.no_grad()
def predict_image(model, image):
    """Run a loaded stage-2 model on one new RGB image (a PIL image or a path), outside any dataset.

    This is the building block for applications: the support set is the one the model was loaded
    with, the query is whatever image you pass. Returns ``(raw, pred)`` at the image's own
    resolution -- ``raw`` is the dense output before ``postprocess_final`` (heatmaps, logits),
    ``pred`` after it (e.g. a binary or instance-id mask).

    Single-image tasks only: FSC-147 and Cellpose take a 6-channel ``X`` (photo + exemplar map, or
    two stains) that has to be built the way their Dataset classes do.
    """
    import torchvision.transforms.functional as TF
    from PIL import Image

    config, learner = model.config, model.learner
    if config.dataset in ('fsc147', 'cellpose'):
        raise ValueError(f'{config.dataset} needs a 6-channel input; build X as its Dataset does')
    if isinstance(image, str):
        image = Image.open(image)
    image = image.convert('RGB')
    W, H = image.size
    X = TF.to_tensor(image.resize((config.base_size[1], config.base_size[0]), Image.BILINEAR))[None].cuda()

    final, learner.postprocess_final = learner.postprocess_final, (lambda y: y)
    try:
        raw = model.inference(X, config.task).float()
    finally:
        learner.postprocess_final = final
    pred = final(raw)
    raw = F.interpolate(raw, (H, W), mode='bilinear', align_corners=False)
    pred = pred[:, None] if pred.ndim == 3 else pred
    pred = F.interpolate(pred.float(), (H, W), mode='nearest')
    return raw[0].cpu(), pred[0].cpu()


def _index_aux(aux, i):
    if isinstance(aux, torch.Tensor):
        return aux[i]
    if isinstance(aux, (list, tuple)):
        return type(aux)(_index_aux(a, i) for a in aux)
    if isinstance(aux, dict):
        return {k: _index_aux(v, i) for k, v in aux.items()}
    return aux


def show_predictions(model, samples, title=None, save_path=None):
    """Input channels, then raw dense label / prediction, then the decoded task output."""
    from task_vis import show_predictions as _show
    _show(model, samples, title=title, save_path=save_path)


# ---------------------------------------------------------------------------------------------
# Figures for Steps 2 and 4
# ---------------------------------------------------------------------------------------------

def show_channels(tensor, titles=None, title=None, cols=6, cmap='inferno'):
    """Display each channel of a ``(C, H, W)`` label tensor separately.

    Useful for seeing what a multi-channel label actually contains -- e.g. the 17 keypoint
    heatmaps that make up one AP-10K annotation.
    """
    tensor = tensor.float().cpu()
    n = tensor.shape[0]
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(1.7 * cols, 1.8 * rows))
    axes = axes.ravel() if n > 1 else [axes]
    for i in range(rows * cols):
        axes[i].axis('off')
        if i < n:
            axes[i].imshow(tensor[i], cmap=cmap, vmin=0, vmax=max(1e-8, float(tensor[i].max())))
            if titles:
                axes[i].set_title(titles[i], fontsize=7)
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    plt.show()


def show_images(images, titles=None, title=None, cols=5, scale=2.4):
    """Display a list of ``(3, H, W)`` tensors in a row."""
    images = [im.float().cpu() for im in images]
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(scale * cols, scale * rows))
    axes = axes.ravel() if n > 1 or rows > 1 else [axes]
    for i in range(rows * cols):
        axes[i].axis('off')
        if i < n:
            axes[i].imshow(images[i].permute(1, 2, 0).numpy().clip(0, 1))
            if titles:
                axes[i].set_title(titles[i], fontsize=9)
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    plt.show()


def plot_loss(losses, title='Fine-tuning loss'):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(losses, linewidth=1.2)
    ax.set_xlabel('step')
    ax.set_ylabel('loss')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------------------------
# Asset checking
# ---------------------------------------------------------------------------------------------

# What each task's demo data must contain, relative to DEMO_DATA_ROOT/<dataset>: the flat layout
# read by downstream/demo.py (images/N.*, labels/N.*; the first `shot` files are the support set).
REQUIRED_DATA = {
    'ap10k': ['images', 'labels'],
    'davis2017': ['images', 'labels'],
    'isic2018': ['images', 'labels'],
    'fsc147': ['images', 'labels', 'exemplars.json'],
    'cellpose': ['images', 'labels'],
}


def dataset_class(config):
    """The Dataset class chameleon uses for ``config.dataset``: the demo loader for the flat
    ``images/`` + ``labels/`` layout, the official one otherwise."""
    from downstream.demo import get_dataset_cls
    return get_dataset_cls(config.dataset, config.path_dict[config.dataset])


def check_assets(checkpoint_root, data_root):
    """Print a per-file report of which checkpoints and demo datasets are present."""
    ok = True

    print('CHECKPOINTS')
    try:
        print(f'  ✓ meta-trained  {find_meta_trained(checkpoint_root)}')
    except FileNotFoundError as exc:
        print(f'  ✗ meta-trained  {exc}')
        ok = False
    for task in TASKS:
        try:
            find_finetuned(checkpoint_root, task)
            names = list_checkpoints(checkpoint_root, task)
            print(f'  ✓ {task:<30} {len(names)} checkpoint(s): {", ".join(names[:6])}'
                  + (' ...' if len(names) > 6 else ''))
        except FileNotFoundError as exc:
            print(f'  ✗ {task:<30} {exc}')
            ok = False

    print('\nDEMO DATA')
    for task, spec in TASKS.items():
        dataset = spec['dataset']
        root = os.path.join(data_root, dataset)
        missing = [rel for rel in REQUIRED_DATA[dataset] if not os.path.exists(os.path.join(root, rel))]
        if missing:
            print(f'  ✗ {dataset:<12} missing: {", ".join(missing)}')
            ok = False
        else:
            n = len(os.listdir(os.path.join(root, 'images')))
            print(f'  ✓ {dataset:<12} {n} images')

    print('\nAll assets present.' if ok else
          '\nSome assets are missing -- check that the shared course folder is added to your Drive '
          'under its original name (README.md, Getting Started, step 4).')
    return ok
