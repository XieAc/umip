"""Turn a checkpoint's saved policy config into one the current code accepts.

Checkpoints from before the 2026-10-22 cleanup carry encoder options that no
longer exist.  Those trained with the removed pipeline (no ImageNet input
normalization, or a frozen random spatial_embedding) are rejected: their
weights expect inputs the current encoder no longer produces.
"""
import copy

REMOVED_ENCODER_KEYS = ('imagenet_norm', 'input_norm', 'freeze_scope')


class ObsoleteCheckpointError(ValueError):
    pass


def policy_cfg_for_loading(policy_cfg: dict) -> dict:
    cfg = copy.deepcopy(policy_cfg)
    enc = cfg['obs_encoder']
    if 'imagenet_norm' in enc:  # saved before the cleanup
        if enc.get('input_norm') != 'imagenet':
            raise ObsoleteCheckpointError(
                'Checkpoint was trained without ImageNet input normalization (removed pipeline); '
                'retrain with train_xac_v2.')
        if enc.get('frozen') and enc.get('freeze_scope', 'all') == 'all':
            raise ObsoleteCheckpointError(
                'Checkpoint was trained with a frozen random spatial_embedding (removed pipeline); '
                'retrain with train_xac_v2.')
    for key in REMOVED_ENCODER_KEYS:
        enc.pop(key, None)
    return cfg
