"""Task-specific figures for the starter notebook (not part of upstream Chameleon).

Every figure has the same structure, one column per query:

1. the **input channels** (``X`` is ``(3M, H, W)``: one row per input image),
2. the **raw dense label and prediction** -- exactly the tensors the model is trained on and
   outputs (heatmaps, soft masks, density maps, flow fields), with channels that are invalid for
   that image (``M = 0``, e.g. unannotated joints) masked out,
3. the **task-level label and prediction** -- what those dense maps *mean* (a skeleton, a mask,
   a count, cell instances), decoded with the same post-processing the evaluation code uses.

The decoding and drawing reuse upstream tools wherever they exist: ``dense_to_sparse`` +
``vis_animal_keypoints`` (AP-10K), the learners' ``postprocess_final`` + ``postprocess_semseg``
(DAVIS, ISIC), ``preprocess_kpmap`` (FSC-147 points; its mode count is the count the evaluator
reports), ``flow_vis`` + ``compute_masks`` (Cellpose).
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from matplotlib.colors import to_rgb
from PIL import Image, ImageDraw

TILE = 256  # every panel is resized to TILE x TILE for the grid

# instance colours, in the order train.visualize.postprocess_semseg assigns them
SEMSEG_COLORS = ('red', 'blue', 'yellow', 'magenta', 'green', 'indigo', 'darkorange', 'cyan',
                 'pink', 'yellowgreen', 'black', 'darkgreen', 'brown', 'gray', 'purple', 'darkviolet')


# ---------------------------------------------------------------------------------------------
# small drawing helpers; every panel is a float (3, h, w) tensor in [0, 1]
# ---------------------------------------------------------------------------------------------

def tile(img):
    img = img.float()
    if img.ndim == 2:
        img = img[None]
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    return F.interpolate(img[None], (TILE, TILE), mode='bilinear', align_corners=False)[0].clip(0, 1)


def heat(x, cmap='inferno'):
    """A (h, w) map in [0, 1] as a colour image."""
    x = x.float().clip(0, 1).cpu().numpy()
    return torch.from_numpy(colormaps[cmap](x)[..., :3]).permute(2, 0, 1).float()


def instance_colors(probs):
    """(K, h, w) per-instance probabilities -> colour mix on black (instance k gets colour k)."""
    colors = torch.tensor([to_rgb(SEMSEG_COLORS[k % len(SEMSEG_COLORS)]) for k in range(len(probs))])
    return torch.einsum('khw,kc->chw', probs.float(), colors).clip(0, 1)


def draw(img, fn):
    """Draw with PIL on a TILE-sized copy of ``img``; ``fn(ImageDraw, scale)``."""
    pil = Image.fromarray((tile(img).permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    fn(ImageDraw.Draw(pil), TILE)
    return torch.from_numpy(np.array(pil)).permute(2, 0, 1).float() / 255


# ---------------------------------------------------------------------------------------------
# per-task renderers: sample -> (rows, annotations)
#   rows:        list of (row label, panel)
#   annotations: {row label: text drawn in red bold at the top middle of that panel}
# ---------------------------------------------------------------------------------------------

def render_ap10k(learner, s):
    from dataset.utils import dense_to_sparse
    from downstream.ap10k.utils import vis_animal_keypoints

    img, Y, M, raw = s['X'][:3], s['Y'], s['M'], s['raw']
    visibility = s['aux'][0][:, 2].float()                 # annotated joints (2 = visible)

    def skeleton(dense):
        kps = dense_to_sparse(dense.float()).transpose(0, 1).numpy()
        kps[2] = visibility.numpy()                        # upstream: draw annotated joints only
        # dimmed background as in AP10KLearner.postprocess_vis, so the skeleton stands out
        canvas = np.ascontiguousarray((img * 128).byte().permute(1, 2, 0).numpy())
        return torch.from_numpy(vis_animal_keypoints(canvas, kps, lth=2, crad=3)).permute(2, 0, 1).float() / 255

    return [
        ('image', img),
        ('label: 17 heatmaps (max)', heat((Y * M).max(0).values)),
        ('prediction: 17 heatmaps (max)', heat((raw * M).max(0).values)),   # M: unannotated joints
        ('label: skeleton', skeleton(Y)),
        ('prediction: skeleton', skeleton(raw)),
    ], {}


def render_segmentation(learner, s):
    """DAVIS (K instance channels, softmax with an implicit background) and ISIC (1 channel)."""
    from train.visualize import postprocess_semseg

    img, Y, M, raw = s['X'][:3], s['Y'].float(), s['M'].float(), s['raw'].float()
    if len(raw) == 1:
        soft = raw.sigmoid() * M                            # ISIC: lesion probability
        raw_label, raw_pred = heat(Y[0]), heat(soft[0])
        name = 'lesion probability'
    else:
        soft = torch.cat([torch.zeros_like(raw[:1]), raw]).softmax(0)[1:] * M   # DAVIS: per-instance
        raw_label, raw_pred = instance_colors(Y), instance_colors(soft)
        name = 'instance probabilities'
    pred_ids = learner.postprocess_final(raw[None])         # (1, h, w): the thresholded / argmax mask
    return [
        ('image', img),
        ('label: masks' if len(raw) > 1 else 'label: mask', raw_label),
        (f'prediction: {name}', raw_pred),
        ('label: overlay', postprocess_semseg(Y[None], img[None])[0]),
        ('prediction: overlay', postprocess_semseg(pred_ids.cpu(), img[None])[0]),
    ], {}


def render_fsc147(learner, s):
    from downstream.fsc147.utils import preprocess_kpmap

    img, exemplar, Y, M, raw = s['X'][:3], s['X'][3:6], s['Y'].float(), s['M'].float(), s['raw'].float()
    mode_value, boxes = s['aux']                            # density peak before scaling; 3 boxes
    H, W = Y.shape[-2:]

    def with_boxes(d, n):
        for x1, y1, x2, y2 in boxes.tolist():               # normalised to [0, 1]
            d.rectangle([x1 * n, y1 * n, x2 * n, y2 * n], outline=(255, 0, 0), width=2)

    def with_points(points):
        def fn(d, n):
            for r, c in points.tolist():
                x, y = (c + 0.5) * n / W, (r + 0.5) * n / H
                d.ellipse([x - 2.5, y - 2.5, x + 2.5, y + 2.5], fill=(255, 255, 0), outline=(0, 0, 0))
        return fn

    gt_points = preprocess_kpmap(Y[0].clip(0, 1), threshold=0.2)
    pred_points = preprocess_kpmap(raw[0].clip(0, 1), threshold=0.2)
    gt_count = int(round(float((Y * mode_value).sum())))   # the density map sums to the count
    pred_count = len(pred_points)                           # FSC147Learner counts density modes
    return [
        ('image + exemplar boxes', draw(img, with_boxes)),
        ('exemplar channel (augmented)', exemplar),
        ('label: density', heat(Y[0])),
        ('prediction: density', heat((raw * M)[0])),
        # darkened background (upstream blends it at 50%) so the points stand out
        ('label: points', draw(img * 0.5, with_points(gt_points))),
        ('prediction: points', draw(img * 0.5, with_points(pred_points))),
    ], {'label: points': f'count: {gt_count}', 'prediction: points': f'count: {pred_count}'}


def render_cellpose(learner, s):
    import flow_vis
    from skimage.segmentation import find_boundaries
    from downstream.cellpose.utils import compute_masks

    X, Y, M, raw = s['X'], s['Y'].float(), s['M'].float(), s['raw'].float()
    cyto, nuclei = X[0], X[3]
    H, W = Y.shape[-2:]
    stretch = lambda c: (c / c.flatten().quantile(0.995).clamp(min=1e-6)).clip(0, 1)
    # the stored RG image (nuclei in red, cytoplasm in green), contrast-stretched for display
    rg = torch.stack([stretch(nuclei), stretch(cyto), torch.zeros_like(cyto)])

    def flows(y):
        """flow colour wheel, faded to white where the cell probability is low"""
        color = flow_vis.flow_to_color((y[:2] * 2 - 1).clip(-1, 1).permute(1, 2, 0).numpy())
        color = torch.from_numpy(color / 255).permute(2, 0, 1).float()
        p = y[2:3].clip(0, 1)
        return p * color + (1 - p)

    def boundaries(mask):
        edge = torch.from_numpy(find_boundaries(mask, mode='thick'))
        out = rg.clone()
        out[:, edge] = torch.tensor([1., 0., 0.])[:, None]
        return out

    gt_mask = s['aux']['full_mask'][0, :H, :W].numpy().astype(np.int32)
    flow = (raw[:2] * 2 - 1).numpy()
    pred_mask, _ = compute_masks(5 * flow, raw[2].numpy(), cellprob_threshold=0.5, flow_threshold=0.,
                                 resize=(H, W), use_gpu=torch.cuda.is_available())
    return [
        ('cytoplasm', cyto),
        ('nuclei', nuclei),
        ('label: flows x cell prob.', flows(Y)),
        ('prediction: flows x cell prob.', flows(raw * M)),   # masked pixels fade to white
        ('label: cells', boundaries(gt_mask)),
        ('prediction: cells', boundaries(pred_mask.astype(np.int32))),
    ], {'label: cells': f'cells: {len(np.unique(gt_mask)) - 1}',
        'prediction: cells': f'cells: {len(np.unique(pred_mask)) - 1}'}


def render_generic(learner, s):
    """Custom tasks: raw maps, then whatever the learner's postprocess_vis draws."""
    X, Y, M, raw = s['X'], s['Y'].float(), s['M'].float(), s['raw'].float()
    rows = [(f'input {i + 1}' if len(X) > 3 else 'input', X[3 * i:3 * i + 3]) for i in range(len(X) // 3)]
    collapse = lambda t: t.max(0).values if len(t) > 1 else t[0]
    final = learner.postprocess_final(raw[None])
    rows += [('label: raw', heat(collapse(Y * M))), ('prediction: raw', heat(collapse(raw * M))),
             ('label', learner.postprocess_vis(Y[None], X[None, :3])[0]),
             ('prediction', learner.postprocess_vis(final.cpu(), X[None, :3])[0])]
    return rows, {}


RENDERERS = {'ap10k': render_ap10k, 'davis2017': render_segmentation,
             'isic2018': render_segmentation, 'fsc147': render_fsc147, 'cellpose': render_cellpose}


# ---------------------------------------------------------------------------------------------

def show_predictions(model, samples, title=None, save_path=None, scale=2.2):
    """Render ``samples`` (from ``starter_utils.predict_queries``) as one figure."""
    render = RENDERERS.get(model.config.dataset, render_generic)
    columns = [render(model.learner, s) for s in samples]
    labels = [label for label, _ in columns[0][0]]
    n_rows, n_cols = len(labels), len(columns)

    label_w, top = 2.4, (0.45 if title else 0.05)             # inches for row labels / title
    fig_w, fig_h = label_w + scale * n_cols, top + scale * n_rows + 0.05
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(left=label_w / fig_w, right=1 - 0.05 / fig_w, bottom=0.05 / fig_h,
                        top=1 - top / fig_h, wspace=0.03, hspace=0.03)
    for c, (rows, notes) in enumerate(columns):
        for r, (label, panel) in enumerate(rows):
            ax = axes[r, c]
            ax.imshow(tile(panel).permute(1, 2, 0).numpy())
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, rotation=0, ha='right', va='center', fontsize=10)
            if label in notes:
                ax.text(0.5, 0.97, notes[label], transform=ax.transAxes, ha='center', va='top',
                        color='red', fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.75, edgecolor='none'))
    if title:
        fig.suptitle(title, fontsize=13, y=1 - 0.12 / fig_h, va='top')
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()
