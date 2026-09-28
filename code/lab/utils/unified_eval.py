"""Unified evaluation core shared by the multi-seed, temperature-scaling and
OER-threshold-sensitivity experiments.

Why this module exists
----------------------
The earlier evaluation scripts (``evaluate_calibration.py``,
``evaluate_oer_threshold.py``) collected classifier logits only for the
*double* pulse branch and derived the label as ``(dt > 0).long()``, which is
identically 1 for every sample of ``PulsePrepare``.  The resulting task is
single-class, so accuracy/ECE/OER were computed on a degenerate problem.

``control/saefit.py::test_step`` defines the classification task the model was
actually trained and reported on: singles (``x1 / m1``) carry label 0, doubles
(``xd / m``) carry label 1, concatenated into one balanced batch.  This module
reproduces that definition exactly and is the single source of truth for the
canonical OER used in all follow-up experiments.
"""

import numpy as np
import torch
import torch.nn.functional as F

from lab.utils.perturbations import PERTURBATION_CONFIGS, apply_perturbation

__all__ = [
    'seed_eval_dataset',
    'collect_balanced_predictions',
    'compute_oer',
    'threshold_sweep',
    'risk_coverage_curve',
    'compute_aurc',
    'bootstrap_ci',
    'summarize_predictions',
    'PERTURBATION_CONFIGS',
]


def seed_eval_dataset(seed=12345):
    """Pin the RNG that ``PulsePrepare`` uses to synthesize the eval set.

    ``PulsePrepare`` draws its t1/t2 shifts and amplitude scales with bare
    ``np.random.*`` calls, so without this the test set is re-synthesized
    differently on every call and two evaluations of the *same* checkpoint
    disagree at the ~5e-4 level -- comparable to the Stage1/Stage2 effect being
    measured.  Call this immediately before constructing any eval dataloader so
    all checkpoints see a byte-identical test set.

    The real-trace train/val/test split is unaffected: it is controlled
    separately by ``dataset_options['seed']`` via ``train_test_split``.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


def _perturb(x, spec):
    """Apply a perturbation given either a config name or an explicit dict.

    A dict lets callers sweep severity continuously, e.g.
    ``{'noise_std': 0.02, 'baseline_range': (-0.02, 0.02)}``.
    """
    if isinstance(spec, str):
        return apply_perturbation(x, spec)
    from lab.utils.perturbations import add_combined_perturbation
    return add_combined_perturbation(
        x,
        noise_std=spec.get('noise_std', 0.0),
        baseline_range=tuple(spec.get('baseline_range', (0.0, 0.0))),
    )


@torch.no_grad()
def collect_balanced_predictions(model, dataloader, device='cuda',
                                 perturbation=None, max_batches=None):
    """Collect classifier logits on the balanced single/double task.

    Mirrors ``control/saefit.py::test_step``: for every batch the single pulse
    ``x1 / m1`` is labelled 0 and the double pulse ``xd / m`` is labelled 1.

    Args:
        model: a ``control.saefit.SAEFit`` LightningModule (or anything exposing
            ``.model`` as the underlying ``lab.models.net.SAEFit``).
        dataloader: loader yielding ``(x1, x2, xd, m1, m2, m, dt)``.
        device: torch device string.
        perturbation: name from ``PERTURBATION_CONFIGS``, an explicit
            ``{'noise_std': ..., 'baseline_range': ...}`` dict for a severity
            sweep, or ``None`` for clean waveforms.  Applied *after*
            normalization, i.e. to the same tensor the network consumes -- the
            earlier scripts perturbed the raw un-normalized waveform, which made
            the nominal noise_std ~1e4x larger in effective units.
        max_batches: optional cap, for smoke tests.

    Returns:
        dict with ``logits`` (2N, 2), ``labels`` (2N,), ``group`` (2N,) where
        group 0 = single branch and 1 = double branch, ``predictions`` (2N,),
        ``confidences`` (2N,) and ``dt`` (N,) timing separation of the doubles.
    """
    net = getattr(model, 'model', model)
    net = net.to(device).eval()

    # Pin the perturbation draws too, so the same condition is byte-identical
    # across checkpoints and across the uncalibrated/calibrated passes.
    if perturbation is not None:
        torch.manual_seed(20260925)
        if device.startswith('cuda'):
            torch.cuda.manual_seed_all(20260925)

    logits_single, logits_double, dts = [], [], []

    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        x1, _x2, xd, m1, _m2, m, dt = batch
        x1 = (x1.to(device) / m1.to(device)[:, None])
        xd = (xd.to(device) / m.to(device)[:, None])

        if perturbation is not None:
            x1 = _perturb(x1, perturbation)
            xd = _perturb(xd, perturbation)

        logits_single.append(net(x1, control=False)[1].float().cpu())
        logits_double.append(net(xd, control=False)[1].float().cpu())
        dts.append(dt.cpu())

    ls = torch.cat(logits_single, dim=0)
    ld = torch.cat(logits_double, dim=0)

    logits = torch.cat((ls, ld), dim=0)
    labels = torch.cat((torch.zeros(len(ls)), torch.ones(len(ld)))).long()
    group = labels.clone()

    probs = torch.softmax(logits, dim=-1)
    confidences, predictions = probs.max(dim=-1)

    return {
        'logits': logits,
        'labels': labels,
        'group': group,
        'predictions': predictions,
        'confidences': confidences,
        'dt': torch.cat(dts, dim=0),
    }


def compute_oer(confidences, predictions, labels, threshold=0.85,
                mode='conditional', inclusive=True):
    """Canonical Overconfident Error Rate.

    Two reported forms, both on the balanced single/double task:

    ``conditional`` (default, the headline definition)
        ``OER(tau) = P(yhat != y | conf >= tau)`` -- the error rate among
        predictions the model is willing to commit to.  Undefined when no
        sample clears the threshold; reported as ``nan`` in that case rather
        than silently collapsing to 0.0.

    ``absolute``
        ``OER_abs(tau) = P(yhat != y AND conf >= tau)`` -- the fraction of the
        *whole* test set that is both wrong and confident.  Monotonically
        non-increasing in tau, which makes it the safer form for threshold
        sweeps.

    Args:
        inclusive: use ``conf >= tau`` (True, default) or the legacy strict
            ``conf > tau`` (False).  Kept explicit because the legacy scripts
            used strict comparison.
    """
    mask = confidences >= threshold if inclusive else confidences > threshold
    n_total = int(len(confidences))
    n_sel = int(mask.sum().item())
    wrong = (predictions != labels)
    n_wrong_sel = int((wrong & mask).sum().item())

    if mode == 'conditional':
        oer = float('nan') if n_sel == 0 else n_wrong_sel / n_sel
    elif mode == 'absolute':
        oer = n_wrong_sel / n_total
    else:
        raise ValueError(f"unknown mode: {mode}")

    return {
        'threshold': float(threshold),
        'mode': mode,
        'inclusive': bool(inclusive),
        'oer': oer,
        'coverage': n_sel / n_total,
        'rejection_rate': 1.0 - n_sel / n_total,
        'n_selected': n_sel,
        'n_errors_selected': n_wrong_sel,
        'n_total': n_total,
        'error_rate_all': float(wrong.float().mean().item()),
    }


def threshold_sweep(confidences, predictions, labels, thresholds,
                    inclusive=True, n_boot=0, seed=0):
    """Evaluate both OER forms over a grid of confidence thresholds."""
    rows = []
    for tau in sorted(float(t) for t in thresholds):
        cond = compute_oer(confidences, predictions, labels, tau,
                           mode='conditional', inclusive=inclusive)
        absol = compute_oer(confidences, predictions, labels, tau,
                            mode='absolute', inclusive=inclusive)
        row = {
            'threshold': tau,
            'oer_conditional': cond['oer'],
            'oer_absolute': absol['oer'],
            'coverage': cond['coverage'],
            'rejection_rate': cond['rejection_rate'],
            'n_selected': cond['n_selected'],
            'n_errors_selected': cond['n_errors_selected'],
        }
        if n_boot > 0:
            row['oer_conditional_ci95'] = bootstrap_ci(
                confidences, predictions, labels, tau,
                mode='conditional', inclusive=inclusive,
                n_boot=n_boot, seed=seed)
        rows.append(row)
    return rows


def risk_coverage_curve(confidences, predictions, labels, n_points=50):
    """Selective risk as a function of coverage.

    Samples are ranked by confidence and the top ``coverage`` fraction is
    retained; risk is the error rate of that retained set.
    """
    order = torch.argsort(confidences, descending=True)
    wrong = (predictions != labels).float()[order]
    cum_wrong = torch.cumsum(wrong, dim=0)
    n = len(confidences)

    rows = []
    for coverage in np.linspace(1.0, 1.0 / n, n_points):
        k = max(1, int(round(n * float(coverage))))
        risk = float(cum_wrong[k - 1].item() / k)
        rows.append({
            'coverage': k / n,
            'risk': risk,
            'accuracy': 1.0 - risk,
            'threshold': float(confidences[order[k - 1]].item()),
            'n_samples': k,
        })
    return rows


def compute_aurc(confidences, predictions, labels):
    """Area under the risk-coverage curve, evaluated at every coverage level.

    Lower is better.  Also returns the optimal-oracle AURC and the normalized
    excess-AURC ``E-AURC = AURC - AURC_optimal``, which removes the dependence
    on the base error rate and so is comparable across seeds and shifts.
    """
    order = torch.argsort(confidences, descending=True)
    wrong = (predictions != labels).float()[order]
    n = len(wrong)
    ks = torch.arange(1, n + 1, dtype=torch.float64)
    risks = torch.cumsum(wrong.double(), dim=0) / ks
    aurc = float(risks.mean().item())

    n_wrong = int(wrong.sum().item())
    oracle = torch.cat((torch.zeros(n - n_wrong), torch.ones(n_wrong))).double()
    risks_opt = torch.cumsum(oracle, dim=0) / ks
    aurc_opt = float(risks_opt.mean().item())

    return {
        'aurc': aurc,
        'aurc_optimal': aurc_opt,
        'e_aurc': aurc - aurc_opt,
        'error_rate': n_wrong / n,
    }


def bootstrap_ci(confidences, predictions, labels, threshold,
                 mode='conditional', inclusive=True, n_boot=1000,
                 seed=0, alpha=0.05):
    """Percentile bootstrap CI for an OER value.

    Needed because at high thresholds the selected subset contains very few
    errors, so a point estimate alone is not interpretable.
    """
    g = torch.Generator().manual_seed(seed)
    n = len(confidences)
    mask = confidences >= threshold if inclusive else confidences > threshold
    wrong = (predictions != labels)

    stats = []
    for _ in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=g)
        m = mask[idx]
        w = wrong[idx] & m
        n_sel = int(m.sum().item())
        if mode == 'conditional':
            if n_sel == 0:
                continue
            stats.append(int(w.sum().item()) / n_sel)
        else:
            stats.append(int(w.sum().item()) / n)

    if not stats:
        return {'lower': float('nan'), 'upper': float('nan'), 'n_boot': 0}
    arr = np.asarray(stats)
    return {
        'lower': float(np.quantile(arr, alpha / 2)),
        'upper': float(np.quantile(arr, 1 - alpha / 2)),
        'n_boot': len(stats),
    }


def summarize_predictions(data, threshold=0.85, n_bins=15):
    """Accuracy / ECE / NLL / OER plus per-branch accuracy for one prediction set."""
    from lab.utils.calibration import compute_ece

    logits, labels = data['logits'], data['labels']
    predictions, confidences = data['predictions'], data['confidences']
    group = data['group']

    out = {
        'n_samples': int(len(labels)),
        'accuracy': float((predictions == labels).float().mean().item()),
        'ece': float(compute_ece(confidences, predictions, labels, n_bins=n_bins)),
        'nll': float(F.cross_entropy(logits, labels).item()),
        'mean_confidence': float(confidences.mean().item()),
        f'oer_conditional@{threshold}': compute_oer(
            confidences, predictions, labels, threshold, 'conditional')['oer'],
        f'oer_absolute@{threshold}': compute_oer(
            confidences, predictions, labels, threshold, 'absolute')['oer'],
    }
    for g_id, name in ((0, 'single'), (1, 'double')):
        sel = group == g_id
        if sel.any():
            out[f'accuracy_{name}'] = float(
                (predictions[sel] == labels[sel]).float().mean().item())
    out.update(compute_aurc(confidences, predictions, labels))
    return out
