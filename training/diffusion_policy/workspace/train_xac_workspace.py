"""Single-GPU XAC training with EMA, action-error model selection and resumable checkpoints.

Every ``training.action_eval_every`` epochs the EMA policy samples actions on
the held-out sessions; ``best.ckpt`` is the lowest ``val_action_score``
(sampled action error relative to a hold-still baseline).  The diffusion
noise ``val_loss`` is logged every epoch but never used for decisions: it is
a poor proxy for action quality.  Training always runs the full schedule.
"""
import copy
import hashlib
import json
import random
from contextlib import nullcontext
from pathlib import Path
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.common.action_metrics import evaluate_policy, score as action_score


class TrainXacWorkspace(BaseWorkspace):
    def run(self):
        cfg = self.cfg
        torch.manual_seed(cfg.training.seed)
        np.random.seed(cfg.training.seed)
        random.seed(cfg.training.seed)
        torch.set_num_threads(cfg.training.cpu_threads)
        device = torch.device(cfg.training.device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable: install a PyTorch build supporting the training GPU')
        bf16_amp = bool(cfg.training.get('bf16_amp', True))
        if bf16_amp and device.type != 'cuda':
            print(f'BF16 AMP requested for {device}, but this training path enables AMP only on CUDA; using FP32', flush=True)
            bf16_amp = False
        elif bf16_amp and not torch.cuda.is_bf16_supported():
            print('BF16 AMP requested, but the CUDA device does not support BF16; using FP32', flush=True)
            bf16_amp = False

        def autocast_context():
            if bf16_amp:
                return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
            return nullcontext()

        action_eval_every = int(cfg.training.action_eval_every)
        if action_eval_every <= 0:
            raise ValueError('training.action_eval_every must be positive')
        save_every = int(cfg.training.get('save_every', 0) or 0)
        n_exec = int(cfg.n_action_steps)

        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        val_dataset = dataset.get_validation_dataset()
        train_loader = DataLoader(dataset, **cfg.dataloader)
        val_loader = DataLoader(val_dataset, **cfg.val_dataloader)
        if not len(train_loader) or not len(val_loader):
            raise ValueError('Both train and validation must be nonempty; use at least two source sessions')
        model = hydra.utils.instantiate(cfg.policy)
        model.set_normalizer(dataset.get_normalizer())
        model.to(device)
        ema_model = copy.deepcopy(model)
        ema = hydra.utils.instantiate(cfg.ema, model=ema_model)
        optimizer = self._make_optimizer(cfg, model)
        print('trainable params: ' + ', '.join(
            f"group{i} lr={g['lr']:.2g} n={sum(p.numel() for p in g['params'])/1e6:.2f}M"
            for i, g in enumerate(optimizer.param_groups)), flush=True)
        epochs = 2 if cfg.training.debug else cfg.training.num_epochs
        max_steps = 2 if cfg.training.debug else cfg.training.max_train_steps
        steps_per_epoch = min(len(train_loader), max_steps or len(train_loader))
        scheduler = get_scheduler(cfg.training.lr_scheduler, optimizer=optimizer,
            num_warmup_steps=min(cfg.training.lr_warmup_steps, steps_per_epoch*epochs//10),
            num_training_steps=steps_per_epoch*epochs)
        contract = {
            'dataset': str(Path(cfg.task.dataset_path).resolve()),
            'dataset_sha256': self._dataset_hash(cfg.task.dataset_path),
            'shape_meta': OmegaConf.to_container(cfg.shape_meta, resolve=True),
            'pose_repr': OmegaConf.to_container(cfg.task.pose_repr, resolve=True),
            'frequency_hz': cfg.task.dataset_frequeny,
            'sessions': dataset.source_sessions,
            'validation_mask': dataset.val_mask.tolist(),
            'gripper_samples': dataset.gripper_samples,
            'policy': OmegaConf.to_container(cfg.policy, resolve=True),
        }
        ckpt_dir = out/'checkpoints'
        ckpt_dir.mkdir(exist_ok=True)
        last = ckpt_dir/'last.ckpt'
        start_epoch, global_step, best = 0, 0, float('inf')
        if last.exists():
            if not cfg.training.resume:
                raise FileExistsError('Output already has checkpoint: use training.resume=True or a new output directory')
            payload = torch.load(last, map_location=device, weights_only=False)
            if 'best_metric_value' not in payload:
                raise ValueError('Checkpoint is from the removed training pipeline; start a new experiment')
            if payload['contract'] != contract:
                raise ValueError('Dataset/config/calibration changed; start a new experiment')
            if payload['cfg']['training']['lr_scheduler'] != cfg.training.lr_scheduler:
                raise ValueError('Scheduler changed; use the saved scheduler or a new output directory')
            model.load_state_dict(payload['model'])
            ema_model.load_state_dict(payload['ema_model'])
            optimizer.load_state_dict(payload['optimizer'])
            scheduler.load_state_dict(payload['scheduler'])
            start_epoch, global_step = payload['epoch']+1, payload['global_step']
            best = payload['best_metric_value']
            ema.optimization_step = payload['ema_step']
            torch.set_rng_state(payload['torch_rng'].cpu())
            np.random.set_state(payload['numpy_rng'])
            random.setstate(payload['python_rng'])
            if device.type == 'cuda':
                torch.cuda.set_rng_state_all([x.cpu() for x in payload['cuda_rng']])
        (out/'data_contract.json').write_text(json.dumps(contract, ensure_ascii=False, indent=2))
        torch.save(model.normalizer.state_dict(), out/'normalizer.pt')
        precision = 'bf16-mixed' if bf16_amp else 'fp32'
        print(f'train_windows={len(dataset)} val_windows={len(val_dataset)} device={device} precision={precision}', flush=True)
        for epoch in range(start_epoch, epochs):
            model.train()
            total, count = 0.0, 0
            for i, batch in enumerate(train_loader):
                batch = dict_apply(batch, lambda x: x.to(device))
                optimizer.zero_grad(set_to_none=True)
                with autocast_context():
                    loss = model.compute_loss(batch)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                ema.step(model)
                total += loss.item()*len(batch['action'])
                count += len(batch['action'])
                global_step += 1
                if max_steps and i+1 >= max_steps: break
            ema_model.eval()
            val_total, val_count = 0.0, 0
            # Logged only. Fixed diffusion noise keeps it comparable across epochs.
            with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
                torch.manual_seed(cfg.training.seed+1)
                with torch.no_grad():
                    for i, batch in enumerate(val_loader):
                        batch = dict_apply(batch, lambda x: x.to(device))
                        with autocast_context():
                            loss = ema_model.compute_loss(batch)
                        val_total += loss.item()*len(batch['action'])
                        val_count += len(batch['action'])
                        if cfg.training.debug and i >= 1: break
            val_loss = val_total/val_count
            if not np.isfinite(val_loss): raise RuntimeError('Nonfinite validation loss')
            metrics = dict(epoch=epoch, global_step=global_step, train_loss=total/count, val_loss=val_loss,
                           lr=float(optimizer.param_groups[0]['lr']))
            improved = False
            if (epoch + 1) % action_eval_every == 0 or epoch + 1 == epochs:
                # fp32 on purpose: DDIM sampling is sensitive to bf16 rounding.
                act, hold, _ = evaluate_policy(ema_model, val_dataset, device, n_exec,
                                               batch_size=cfg.val_dataloader.batch_size,
                                               seed=cfg.training.seed + 2, num_workers=0)
                metrics['val_action_score'] = action_score(act, hold)
                metrics.update({f'val_{k}': v for k, v in act.items()})
                metrics.update({f'hold_{k}': v for k, v in hold.items() if k.endswith('_exec')})
                if metrics['val_action_score'] < best:
                    best, improved = metrics['val_action_score'], True
            print(json.dumps(metrics), flush=True)
            with (out/'metrics.jsonl').open('a') as stream: stream.write(json.dumps(metrics)+'\n')
            payload = dict(cfg=OmegaConf.to_container(cfg, resolve=True), contract=contract,
                model=model.state_dict(), ema_model=ema_model.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), epoch=epoch, global_step=global_step,
                best_metric='val_action_score', best_metric_value=best, metrics=metrics,
                ema_step=ema.optimization_step, torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                python_rng=random.getstate(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
            torch.save(payload, ckpt_dir/'last.tmp')
            (ckpt_dir/'last.tmp').replace(last)
            completed_epoch = epoch + 1
            if save_every and completed_epoch % save_every == 0:
                torch.save(payload, ckpt_dir/f'epoch_{completed_epoch:05d}.ckpt')
            if improved:
                print(f'New best val_action_score={best:.6g} at epoch {epoch}', flush=True)
                torch.save(payload, ckpt_dir/'best.tmp')
                (ckpt_dir/'best.tmp').replace(ckpt_dir/'best.ckpt')

    @staticmethod
    def _make_optimizer(cfg, model):
        """One group for the diffusion head; an optional lower-LR group for the
        image backbone (training.obs_encoder_lr) when it is fine-tuned."""
        trainable = [p for p in model.parameters() if p.requires_grad]
        encoder_lr = cfg.training.get('obs_encoder_lr', None)
        if encoder_lr is None:
            return hydra.utils.instantiate(cfg.optimizer, params=trainable)
        backbone = [p for p in model.obs_encoder.key_model_map.parameters() if p.requires_grad]
        backbone_ids = {id(p) for p in backbone}
        head = [p for p in trainable if id(p) not in backbone_ids]
        groups = [{'params': head}]
        if backbone:
            groups.append({'params': backbone, 'lr': float(encoder_lr)})
        kwargs = OmegaConf.to_container(cfg.optimizer, resolve=True)
        if kwargs.pop('_target_') != 'torch.optim.AdamW':
            raise ValueError('obs_encoder_lr param groups support torch.optim.AdamW only')
        return torch.optim.AdamW(groups, **kwargs)

    @staticmethod
    def _dataset_hash(path):
        digest = hashlib.sha256()
        with open(path, 'rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()
