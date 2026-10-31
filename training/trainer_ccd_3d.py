"""Dataset-agnostic trainer for spatiotemporal CCD3D models.

Expected tensor convention
--------------------------
``clean`` and ``noisy`` have shape ``(B, C, T, H, W)``.  A loader may yield

* ``(clean, noisy, sigma)``;
* ``{"clean": clean, "noisy": noisy, "sigma": sigma}``; or
* ``(clean, sigma)`` / ``{"clean": clean, "sigma": sigma}``, in which case
  this trainer synthesizes AWGN when ``synthesize_noise_if_missing=True``.

For synthesized noise, one sigma is used per sample while independent Gaussian
draws are generated at every channel, time, and spatial location.  The trainer
does not construct datasets or impose any file format.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


BatchAdapter = Callable[[Any, torch.device], Tuple[Tensor, Tensor, Tensor]]
EpochCallback = Callable[[Dict[str, float]], None]


@dataclass
class TrainerConfig:
    epochs: int = 100
    device: str = "cuda"
    seed: int = 0

    optimizer: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)

    # Optional differential learning rates.  ``None`` selects a documented
    # multiple of ``lr`` matching the earlier CCD training convention.
    lr_conv: Optional[float] = None
    lr_q: Optional[float] = None
    lr_Q: Optional[float] = None
    lr_tau: Optional[float] = None
    lr_gamma: Optional[float] = None
    lr_activation: Optional[float] = None
    lr_scaling: Optional[float] = None
    lr_bias: Optional[float] = None

    loss: str = "mse"
    amp: bool = True
    grad_clip_norm: Optional[float] = None

    scheduler: str = "multistep"
    milestones: Tuple[int, ...] = (50, 75, 90)
    scheduler_gamma: float = 0.5
    cosine_eta_min: float = 1e-6

    synthesize_noise_if_missing: bool = True
    data_range: float = 1.0

    validate_every: int = 1
    checkpoint_every: int = 1
    checkpoint_dir: str = "checkpoints_ccd_3d"
    resume_from: Optional[str] = None
    keep_last: bool = True


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _canonical_sigma_for_noise(sigma: Any, clean: Tensor) -> Tensor:
    """Broadcast sigma to ``(B,1,T,H,W)`` for AWGN synthesis."""
    if clean.ndim != 5:
        raise ValueError(
            "Expected clean data with shape (B,C,T,H,W), "
            f"received {tuple(clean.shape)}."
        )
    batch, _, time, height, width = clean.shape
    sigma_tensor = torch.as_tensor(
        sigma, device=clean.device, dtype=clean.dtype
    )
    if sigma_tensor.numel() in (1, batch) and sigma_tensor.ndim < 5:
        sigma_tensor = sigma_tensor.reshape(-1, 1, 1, 1, 1)
    elif sigma_tensor.ndim != 5:
        raise ValueError(
            "sigma must be scalar, one scalar per sample, or (B,1,T,H,W)."
        )
    if sigma_tensor.shape[1] != 1:
        raise ValueError("sigma must have one channel.")
    try:
        return sigma_tensor.expand(batch, 1, time, height, width)
    except RuntimeError as exc:
        raise ValueError(
            f"sigma shape {tuple(sigma_tensor.shape)} cannot broadcast to "
            f"{(batch, 1, time, height, width)}."
        ) from exc


def add_spatiotemporal_awgn(clean: Tensor, sigma: Any) -> Tensor:
    """Add independent zero-mean Gaussian samples using one sigma map.

    Independence is across channels, time, height, and width.  Supplying one
    scalar per batch item therefore gives the usual AWGN model with a common
    noise level over the complete spatiotemporal sample.
    """
    sigma_map = _canonical_sigma_for_noise(sigma, clean)
    return clean + torch.randn_like(clean) * sigma_map


def _mapping_get(batch: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in batch:
            return batch[name]
    return None


def default_batch_adapter(
    batch: Any,
    device: torch.device,
    synthesize_noise_if_missing: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Normalize common loader outputs to ``(clean, noisy, sigma)``."""
    clean = noisy = sigma = None

    if isinstance(batch, Mapping):
        clean = _mapping_get(batch, ("clean", "target", "ground_truth", "gt"))
        noisy = _mapping_get(batch, ("noisy", "input", "observation"))
        sigma = _mapping_get(batch, ("sigma", "noise_level", "std"))
    elif isinstance(batch, (tuple, list)):
        if len(batch) == 3:
            clean, noisy, sigma = batch
        elif len(batch) == 2:
            clean, sigma = batch
        else:
            raise ValueError(
                "A sequence batch must be (clean,noisy,sigma) or (clean,sigma)."
            )
    else:
        raise TypeError(
            "Unsupported batch type. Supply a mapping or tuple/list."
        )

    if clean is None or sigma is None:
        raise ValueError("Every batch must provide clean data and sigma.")

    clean = torch.as_tensor(clean).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    sigma = torch.as_tensor(sigma).to(
        device=device, dtype=clean.dtype, non_blocking=True
    )

    if clean.ndim != 5:
        raise ValueError(
            "clean must have shape (B,C,T,H,W), "
            f"received {tuple(clean.shape)}."
        )

    if noisy is None:
        if not synthesize_noise_if_missing:
            raise ValueError(
                "The batch did not provide noisy data and synthesis is disabled."
            )
        noisy = add_spatiotemporal_awgn(clean, sigma)
    else:
        noisy = torch.as_tensor(noisy).to(
            device=device, dtype=clean.dtype, non_blocking=True
        )

    if noisy.shape != clean.shape:
        raise ValueError(
            f"noisy shape {tuple(noisy.shape)} does not match clean shape "
            f"{tuple(clean.shape)}."
        )
    return clean, noisy, sigma


def _unique_trainable(parameters: Iterable[nn.Parameter]) -> List[nn.Parameter]:
    output: List[nn.Parameter] = []
    seen = set()
    for parameter in parameters:
        if parameter is None or not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        output.append(parameter)
    return output


def _activation_parameters(module: Optional[nn.Module]) -> List[nn.Parameter]:
    if module is None:
        return []
    parameters: List[nn.Parameter] = []
    for name in ("coefficients_vect", "coefficients"):
        value = getattr(module, name, None)
        if isinstance(value, nn.Parameter):
            parameters.append(value)
    return _unique_trainable(parameters)


def build_optimizer(model: nn.Module, config: TrainerConfig) -> Optimizer:
    """Build non-overlapping CCD parameter groups with differential LRs."""
    base_lr = float(config.lr)
    learning_rates = {
        "conv": config.lr_conv if config.lr_conv is not None else base_lr / 10.0,
        "q": config.lr_q if config.lr_q is not None else base_lr / 10.0,
        "Q": config.lr_Q if config.lr_Q is not None else base_lr / 10.0,
        # Current lab trainer uses the base LR for tau/gamma.
        "tau": config.lr_tau if config.lr_tau is not None else base_lr,
        "gamma": config.lr_gamma if config.lr_gamma is not None else base_lr,
        "activation": (
            config.lr_activation
            if config.lr_activation is not None else base_lr / 100.0
        ),
        "scaling": (
            config.lr_scaling
            if config.lr_scaling is not None else base_lr / 10.0
        ),
        "bias": config.lr_bias if config.lr_bias is not None else base_lr,
        "remaining": base_lr,
    }

    groups: List[Dict[str, Any]] = []
    assigned = set()

    def add_group(name: str, parameters: Iterable[nn.Parameter]) -> None:
        params = [
            p for p in _unique_trainable(parameters) if id(p) not in assigned
        ]
        if not params:
            return
        assigned.update(id(p) for p in params)
        groups.append({"params": params, "lr": learning_rates[name], "name": name})

    conv_layer = getattr(model, "conv_layer", None)
    convs = getattr(conv_layer, "conv_layers", ())

    raw_kernels: List[nn.Parameter] = []
    q_parameters: List[nn.Parameter] = []
    for conv in convs:
        parametrizations = getattr(conv, "parametrizations", None)
        weight_parametrizations = (
            getattr(parametrizations, "weight", None)
            if parametrizations is not None else None
        )
        original = (
            getattr(weight_parametrizations, "original", None)
            if weight_parametrizations is not None else None
        )
        if isinstance(original, nn.Parameter):
            raw_kernels.append(original)
        else:
            weight = getattr(conv, "weight", None)
            if isinstance(weight, nn.Parameter):
                raw_kernels.append(weight)
        q = getattr(conv, "q", None)
        if isinstance(q, nn.Parameter):
            q_parameters.append(q)

    add_group("conv", raw_kernels)
    add_group("q", q_parameters)

    Q_obj = getattr(model, "Q", None)
    if isinstance(Q_obj, nn.ParameterList):
        add_group("Q", list(Q_obj.parameters()))
    elif isinstance(Q_obj, nn.Parameter):
        add_group("Q", [Q_obj])

    add_group("activation", _activation_parameters(getattr(model, "activation", None)))
    add_group("scaling", _activation_parameters(getattr(model, "spline_scaling", None)))

    tau = getattr(model, "tau", None)
    gamma = getattr(model, "gamma", None)
    bias = getattr(model, "bias", None)
    add_group("tau", [tau] if isinstance(tau, nn.Parameter) else [])
    add_group("gamma", [gamma] if isinstance(gamma, nn.Parameter) else [])
    add_group("bias", [bias] if isinstance(bias, nn.Parameter) else [])
    add_group("remaining", [p for p in model.parameters() if id(p) not in assigned])

    optimizer_name = config.optimizer.lower()
    common = {
        "lr": base_lr,
        "betas": config.betas,
        "weight_decay": config.weight_decay,
    }
    if optimizer_name == "adam":
        return torch.optim.Adam(groups, **common)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(groups, **common)
    raise ValueError("optimizer must be 'adam' or 'adamw'.")


def build_scheduler(optimizer: Optimizer, config: TrainerConfig):
    name = config.scheduler.lower()
    if name == "none":
        return None
    if name == "multistep":
        milestones = [m for m in config.milestones if 0 < m < config.epochs]
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=config.scheduler_gamma
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, config.epochs),
            eta_min=config.cosine_eta_min,
        )
    raise ValueError("scheduler must be 'none', 'multistep', or 'cosine'.")


def build_loss(name: str) -> nn.Module:
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "l1":
        return nn.L1Loss()
    raise ValueError("loss must be 'mse' or 'l1'.")


def batch_psnr(prediction: Tensor, target: Tensor, data_range: float) -> Tensor:
    mse = (prediction - target).flatten(1).square().mean(dim=1)
    eps = torch.finfo(mse.dtype).eps
    return 10.0 * torch.log10((data_range ** 2) / mse.clamp_min(eps))


class CCD3DTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        batch_adapter: Optional[BatchAdapter] = None,
        on_epoch_end: Optional[EpochCallback] = None,
    ) -> None:
        if config.epochs < 1:
            raise ValueError("epochs must be positive.")
        if config.data_range <= 0:
            raise ValueError("data_range must be positive.")
        if config.validate_every < 1:
            raise ValueError("validate_every must be positive.")
        if config.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be positive.")

        seed_everything(config.seed)
        self.config = config
        self.device = self._resolve_device(config.device)
        self.model = model.to(self.device)
        self.model.project_parameters()

        self.optimizer = build_optimizer(self.model, config)
        self.scheduler = build_scheduler(self.optimizer, config)
        self.criterion = build_loss(config.loss)
        self.batch_adapter = batch_adapter
        self.on_epoch_end = on_epoch_end

        self.amp_enabled = bool(config.amp and self.device.type == "cuda")
        self.scaler = self._make_grad_scaler()

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.start_epoch = 0
        self.best_val_psnr = -math.inf
        self.history: List[Dict[str, float]] = []

        if config.resume_from is not None:
            self.load_checkpoint(config.resume_from)

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return device

    def _make_grad_scaler(self):
        try:
            return torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

    def _autocast(self):
        if not self.amp_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def _adapt_batch(self, batch: Any) -> Tuple[Tensor, Tensor, Tensor]:
        if self.batch_adapter is not None:
            clean, noisy, sigma = self.batch_adapter(batch, self.device)
            return clean, noisy, sigma
        return default_batch_adapter(
            batch,
            self.device,
            synthesize_noise_if_missing=self.config.synthesize_noise_if_missing,
        )

    def _run_epoch(self, loader: DataLoader, training: bool) -> Dict[str, float]:
        self.model.train(training)
        total_loss = 0.0
        total_psnr = 0.0
        total_samples = 0

        grad_context = torch.enable_grad() if training else torch.no_grad()
        with grad_context:
            for batch in loader:
                clean, noisy, sigma = self._adapt_batch(batch)
                batch_size = clean.shape[0]

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                with self._autocast():
                    output = self.model(noisy, sigma)
                    loss = self.criterion(output, clean)

                if training:
                    self.scaler.scale(loss).backward()
                    if self.config.grad_clip_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.grad_clip_norm
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.model.project_parameters()

                with torch.no_grad():
                    psnr = batch_psnr(
                        output.float(), clean.float(), self.config.data_range
                    )
                total_loss += float(loss.detach()) * batch_size
                total_psnr += float(psnr.sum())
                total_samples += batch_size

        if total_samples == 0:
            raise RuntimeError("The data loader produced no samples.")
        return {
            "loss": total_loss / total_samples,
            "psnr": total_psnr / total_samples,
        }

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        return self._run_epoch(loader, training=True)

    def validate(self, loader: DataLoader) -> Dict[str, float]:
        return self._run_epoch(loader, training=False)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> List[Dict[str, float]]:
        for epoch in range(self.start_epoch, self.config.epochs):
            train_metrics = self.train_epoch(train_loader)
            metrics: Dict[str, float] = {
                "epoch": float(epoch + 1),
                "train_loss": train_metrics["loss"],
                "train_psnr": train_metrics["psnr"],
            }

            should_validate = (
                val_loader is not None
                and (epoch + 1) % self.config.validate_every == 0
            )
            is_best = False
            if should_validate:
                val_metrics = self.validate(val_loader)
                metrics.update(
                    val_loss=val_metrics["loss"],
                    val_psnr=val_metrics["psnr"],
                )
                if val_metrics["psnr"] > self.best_val_psnr:
                    self.best_val_psnr = val_metrics["psnr"]
                    is_best = True

            if self.scheduler is not None:
                self.scheduler.step()

            metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
            metrics["tau"] = float(self.model.effective_tau.detach())
            if hasattr(self.model, "contraction_factor"):
                metrics["kappa"] = float(
                    self.model.contraction_factor().detach()
                )
            if hasattr(self.model, "effective_gamma"):
                metrics["gamma"] = float(
                    self.model.effective_gamma.detach()
                )

            self.history.append(metrics)
            if self.on_epoch_end is not None:
                self.on_epoch_end(dict(metrics))

            if (epoch + 1) % self.config.checkpoint_every == 0 or is_best:
                self.save_checkpoint(epoch + 1, is_best=is_best)

        self._write_history()
        return list(self.history)

    def _checkpoint_state(self, epoch: int) -> Dict[str, Any]:
        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_psnr": self.best_val_psnr,
            "history": self.history,
            "config": asdict(self.config),
        }

    def _atomic_torch_save(self, state: Dict[str, Any], path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        os.replace(temporary, path)

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        state = self._checkpoint_state(epoch)
        epoch_path = self.checkpoint_dir / f"model_epoch_{epoch:04d}.pth"
        self._atomic_torch_save(state, epoch_path)

        if self.config.keep_last:
            self._atomic_torch_save(
                state, self.checkpoint_dir / "model_last.pth"
            )
        if is_best:
            self._atomic_torch_save(
                state, self.checkpoint_dir / "model_best.pth"
            )

    def load_checkpoint(self, path: str) -> None:
        try:
            checkpoint = torch.load(
                path, map_location="cpu", weights_only=False
            )
        except TypeError:
            # ``weights_only`` is unavailable in older supported PyTorch
            # releases. This file is a trusted trainer-created checkpoint.
            checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.project_parameters()
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scheduler_state = checkpoint.get("scheduler_state_dict")
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)
        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state is not None:
            self.scaler.load_state_dict(scaler_state)

        self.start_epoch = int(checkpoint.get("epoch", 0))
        self.best_val_psnr = float(
            checkpoint.get("best_val_psnr", -math.inf)
        )
        self.history = list(checkpoint.get("history", []))

    def _write_history(self) -> None:
        path = self.checkpoint_dir / "history.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        os.replace(temporary, path)


__all__ = [
    "CCD3DTrainer",
    "TrainerConfig",
    "add_spatiotemporal_awgn",
    "batch_psnr",
    "build_optimizer",
    "default_batch_adapter",
]
