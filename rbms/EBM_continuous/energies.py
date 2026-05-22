from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def _normalize_hidden_dims(
    hidden_dims: list[int] | tuple[int, ...] | None = None,
    hidden_dim: int = 256,
    num_layers: int = 1,
) -> list[int]:
    if hidden_dims is None:
        hidden_dims = [hidden_dim] * num_layers
    hidden_dims = [int(dim) for dim in hidden_dims]
    if len(hidden_dims) == 0:
        raise ValueError("hidden_dims must contain at least one hidden layer size.")
    if any(dim <= 0 for dim in hidden_dims):
        raise ValueError(f"hidden_dims must be positive, got {hidden_dims}.")
    return hidden_dims


def _init_mlp_layers(
    modules: torch.nn.Sequential,
) -> None:
    """Initialize MLP weights while keeping hidden biases neutral."""

    for module in modules:
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


def _init_cnn_layers(*modules: torch.nn.Module) -> None:
    """Initialize CNN affine layers while keeping biases neutral."""

    for parent in modules:
        for module in parent.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)


def _infer_square_image_shape(num_visibles: int) -> tuple[int, int]:
    image_side = int(num_visibles**0.5)
    if image_side * image_side != num_visibles:
        raise ValueError(
            "CNNEnergy expects square flattened images unless image_shape is provided. "
            f"Got num_visibles={num_visibles}."
        )
    return image_side, image_side


def _rescale_final_linear_to_target_std(
    modules: torch.nn.Sequential,
    data: Tensor,
    target_std: float = 0.05,
    weights: Tensor | None = None,
    batch_size: int = 4096,
    eps: float = 1e-12,
) -> float:
    """Rescale only the final Linear layer so Std_D(f_theta(x)) ~= target_std."""

    linear_layers = [module for module in modules if isinstance(module, torch.nn.Linear)]
    if len(linear_layers) == 0:
        raise ValueError("Cannot calibrate an MLP without Linear layers.")

    final_layer = linear_layers[-1]
    device = final_layer.weight.device
    dtype = final_layer.weight.dtype
    data = data.to(device=device, dtype=dtype)
    if weights is not None:
        weights = weights.to(device=device, dtype=dtype).view(-1)

    outputs = []
    weight_chunks = []
    with torch.no_grad():
        for start in range(0, data.shape[0], batch_size):
            stop = min(start + batch_size, data.shape[0])
            outputs.append(modules(data[start:stop]).view(-1))
            if weights is not None:
                weight_chunks.append(weights[start:stop])

        values = torch.cat(outputs)
        if weights is None:
            current_std = values.std(unbiased=False)
        else:
            sample_weights = torch.cat(weight_chunks)
            norm_weights = sample_weights / sample_weights.sum()
            mean = (values * norm_weights).sum()
            current_std = ((values - mean).square() * norm_weights).sum().sqrt()

        if torch.isfinite(current_std) and current_std > eps:
            scale = torch.as_tensor(target_std, device=device, dtype=dtype) / current_std
            final_layer.weight.mul_(scale)
            return float(scale.detach().cpu())

    return 1.0


class GaussianBaseEnergy(torch.nn.Module):
    """Independent Gaussian reference energy for continuous visibles."""

    def __init__(self, data_mean: Tensor, data_std: Tensor, std_floor: float = 0.2):
        super().__init__()
        self.num_visibles = data_mean.shape[0]
        self.std_floor = float(std_floor)
        self.register_buffer("data_mean", data_mean.clone())
        self.register_buffer("data_std", data_std.clone().clamp_min(self.std_floor))

    def forward(self, x: Tensor) -> Tensor:
        z = (x - self.data_mean) / self.data_std
        return 0.5 * z.square().sum(dim=1)

    @property
    def log_z(self) -> Tensor:
        log_two_pi = torch.log(
            torch.tensor(2.0 * torch.pi, device=self.data_std.device, dtype=self.data_std.dtype)
        )
        return 0.5 * self.num_visibles * log_two_pi + torch.log(self.data_std).sum()


class MLPEnergy(torch.nn.Module):
    """Continuous visible-state energy represented by an MLP plus Gaussian tails.

    A learnable visible field h contributes the linear term -x^T h.
    """

    def __init__(
        self,
        num_visibles: int,
        hidden_dims: list[int] | tuple[int, ...] | None = None,
        hidden_dim: int = 256,
        num_layers: int = 1,
        data_mean: Tensor | None = None,
        data_std: Tensor | None = None,
        base_std_floor: float = 0.02,
        visible_field: Tensor | None = None,
        output_bias: bool = False,
    ):
        super().__init__()
        self.num_visibles = num_visibles
        self.hidden_dims = _normalize_hidden_dims(
            hidden_dims=hidden_dims,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.hidden_dim = self.hidden_dims[-1]
        self.num_layers = len(self.hidden_dims)

        if data_mean is None:
            data_mean = torch.zeros(num_visibles)
        if data_std is None:
            data_std = torch.ones(num_visibles)
        if visible_field is None:
            visible_field = data_mean.clone()
        self.visible_field = torch.nn.Parameter(visible_field.clone())

        self.base_std_floor = float(base_std_floor)
        self.base = GaussianBaseEnergy(
            data_mean=data_mean,
            data_std=data_std,
            std_floor=self.base_std_floor,
        )

        layers = []
        in_dim = num_visibles
        for out_dim in self.hidden_dims:
            layers.append(torch.nn.Linear(in_dim, out_dim))
            layers.append(torch.nn.SiLU())
            in_dim = out_dim
        layers.append(torch.nn.Linear(in_dim, 1, bias=output_bias))

        self.net = torch.nn.Sequential(*layers)
        _init_mlp_layers(self.net)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).view(-1) + self.base(x) - x @ self.visible_field

    def calibrate_final_layer(
        self,
        data: Tensor,
        weights: Tensor | None = None,
        target_std: float = 0.05,
        batch_size: int = 4096,
    ) -> float:
        return _rescale_final_linear_to_target_std(
            modules=self.net,
            data=data,
            weights=weights,
            target_std=target_std,
            batch_size=batch_size,
        )


class CNNEnergy(torch.nn.Module):
    """Continuous flattened-image energy represented by a small CNN.

    A learnable visible field h contributes the linear term -x^T h.
    """

    def __init__(
        self,
        num_visibles: int,
        hidden_dims: list[int] | tuple[int, ...] | None = None,
        hidden_dim: int = 32,
        num_layers: int = 2,
        image_shape: tuple[int, int] | list[int] | None = None,
        kernel_size: int = 3,
        data_mean: Tensor | None = None,
        data_std: Tensor | None = None,
        base_std_floor: float = 0.02,
        visible_field: Tensor | None = None,
        output_bias: bool = False,
    ):
        super().__init__()
        self.num_visibles = int(num_visibles)
        self.hidden_dims = _normalize_hidden_dims(
            hidden_dims=hidden_dims,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.hidden_dim = self.hidden_dims[-1]
        self.num_layers = len(self.hidden_dims)
        self.kernel_size = int(kernel_size)
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}.")

        if image_shape is None:
            image_shape = _infer_square_image_shape(self.num_visibles)
        image_shape = tuple(int(dim) for dim in image_shape)
        if len(image_shape) != 2 or any(dim <= 0 for dim in image_shape):
            raise ValueError(f"image_shape must be a positive (height, width), got {image_shape}.")
        if image_shape[0] * image_shape[1] != self.num_visibles:
            raise ValueError(
                f"image_shape={image_shape} is incompatible with num_visibles={self.num_visibles}."
            )
        self.image_shape = image_shape
        self.register_buffer(
            "_image_shape",
            torch.tensor(image_shape, dtype=torch.int64),
        )

        if data_mean is None:
            data_mean = torch.zeros(self.num_visibles)
        if data_std is None:
            data_std = torch.ones(self.num_visibles)
        if visible_field is None:
            visible_field = data_mean.clone()
        self.visible_field = torch.nn.Parameter(visible_field.clone())
        
        self.base_std_floor = float(base_std_floor)
        self.base = GaussianBaseEnergy(
            data_mean=data_mean,
            data_std=data_std,
            std_floor=self.base_std_floor,
        )

        conv_layers = []
        in_channels = 1
        for out_channels in self.hidden_dims:
            conv_layers.append(
                torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                )
            )
            conv_layers.append(torch.nn.SiLU())
            in_channels = out_channels

        self.conv = torch.nn.Sequential(*conv_layers)
        self.pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.head = torch.nn.Linear(in_channels, 1, bias=output_bias)
        _init_cnn_layers(self.conv, self.head)

    def _score(self, x: Tensor) -> Tensor:
        image = x.view(x.shape[0], 1, *self.image_shape)
        features = self.pool(self.conv(image)).flatten(start_dim=1)
        return self.head(features).view(-1)

    def forward(self, x: Tensor) -> Tensor:
        return self._score(x) + self.base(x) - x @ self.visible_field

    def calibrate_final_layer(
        self,
        data: Tensor,
        weights: Tensor | None = None,
        target_std: float = 0.05,
        batch_size: int = 4096,
        eps: float = 1e-12,
    ) -> float:
        device = self.head.weight.device
        dtype = self.head.weight.dtype
        data = data.to(device=device, dtype=dtype)
        if weights is not None:
            weights = weights.to(device=device, dtype=dtype).view(-1)

        outputs = []
        weight_chunks = []
        with torch.no_grad():
            for start in range(0, data.shape[0], batch_size):
                stop = min(start + batch_size, data.shape[0])
                outputs.append(self._score(data[start:stop]))
                if weights is not None:
                    weight_chunks.append(weights[start:stop])

            values = torch.cat(outputs)
            if weights is None:
                current_std = values.std(unbiased=False)
            else:
                sample_weights = torch.cat(weight_chunks)
                norm_weights = sample_weights / sample_weights.sum()
                mean = (values * norm_weights).sum()
                current_std = ((values - mean).square() * norm_weights).sum().sqrt()

            if torch.isfinite(current_std) and current_std > eps:
                scale = torch.as_tensor(target_std, device=device, dtype=dtype) / current_std
                self.head.weight.mul_(scale)
                return float(scale.detach().cpu())

        return 1.0


ENERGY_MAP: dict[str, type[torch.nn.Module]] = {
    "mlp": MLPEnergy,
    "cnn": CNNEnergy,
    "gaussian": GaussianBaseEnergy,
}


def get_gaussian_base_from_data(
    data: Tensor,
    weights: Tensor | None = None,
    eps: float = 1e-4,
) -> tuple[Tensor, Tensor]:
    """Return weighted mean and standard deviation for continuous data."""

    if weights is None:
        mean = data.mean(dim=0)
        var = data.var(dim=0, unbiased=False)
    else:
        weights = weights.to(device=data.device, dtype=data.dtype).view(-1)
        norm_weights = weights / weights.sum()
        mean = (data * norm_weights[:, None]).sum(dim=0)
        var = ((data - mean).square() * norm_weights[:, None]).sum(dim=0)

    return mean, var.sqrt().clamp_min(eps)


def build_energy(
    energy_type: str,
    num_visibles: int,
    device: torch.device | str,
    dtype: torch.dtype,
    **energy_kwargs,
) -> torch.nn.Module:
    if energy_type not in ENERGY_MAP:
        raise ValueError(
            f"Unknown continuous EBM energy type '{energy_type}'. "
            f"Available energy types: {list(ENERGY_MAP.keys())}."
        )

    if energy_type == "gaussian":
        energy = GaussianBaseEnergy(
            data_mean=energy_kwargs["data_mean"],
            data_std=energy_kwargs["data_std"],
            std_floor=energy_kwargs.get("base_std_floor", 0.02),
        )
    else:
        energy = ENERGY_MAP[energy_type](
            num_visibles=num_visibles,
            **energy_kwargs,
        )
    return energy.to(device=device, dtype=dtype)


def restore_energy(
    named_params: dict[str, np.ndarray],
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.nn.Module:
    energy_type = identify_energy_type(named_params)

    match energy_type:
        case "gaussian":
            energy = restore_gaussian_energy(named_params)
        case "mlp":
            energy = restore_mlp_energy(named_params)
        case "cnn":
            energy = restore_cnn_energy(named_params)
        case _:
            raise ValueError(
                f"Cannot restore unknown continuous energy type '{energy_type}'. "
                f"Available parameter keys: {list(named_params.keys())}."
            )

    state_dict = {
        name: torch.as_tensor(array, device=device, dtype=dtype)
        for name, array in named_params.items()
    }
    if "visible_field" in energy.state_dict() and "visible_field" not in state_dict:
        state_dict["visible_field"] = torch.zeros(
            energy.num_visibles,
            device=device,
            dtype=dtype,
        )
    energy.load_state_dict(state_dict)
    return energy.to(device=device, dtype=dtype)


def identify_energy_type(named_params: dict[str, np.ndarray]) -> str:
    keys = set(named_params)

    match keys:
        case keys if any(name.startswith("conv.") for name in keys):
            return "cnn"
        case keys if any(name.startswith("net.") for name in keys):
            return "mlp"
        case keys if {"data_mean", "data_std"} <= keys:
            return "gaussian"
        case _:
            raise ValueError(
                "Could not identify continuous EBM energy type from saved parameters. "
                f"Available keys: {list(named_params.keys())}."
            )


def restore_gaussian_energy(named_params: dict[str, np.ndarray]) -> GaussianBaseEnergy:
    return GaussianBaseEnergy(
        data_mean=torch.as_tensor(named_params["data_mean"]),
        data_std=torch.as_tensor(named_params["data_std"]),
    )


def restore_mlp_energy(named_params: dict[str, np.ndarray]) -> MLPEnergy:
    weight_keys = sorted(
        [name for name in named_params if name.startswith("net.") and name.endswith(".weight")],
        key=lambda name: int(name.split(".")[1]),
    )
    if len(weight_keys) == 0:
        raise ValueError("Cannot restore MLPEnergy without weight tensors.")

    first_weight = named_params[weight_keys[0]]
    num_visibles = first_weight.shape[1]
    hidden_dims = [named_params[key].shape[0] for key in weight_keys[:-1]]
    final_bias_key = weight_keys[-1].replace(".weight", ".bias")

    return MLPEnergy(
        num_visibles=num_visibles,
        hidden_dims=hidden_dims,
        data_mean=torch.as_tensor(named_params["base.data_mean"]),
        data_std=torch.as_tensor(named_params["base.data_std"]),
        output_bias=final_bias_key in named_params,
    )


def restore_cnn_energy(named_params: dict[str, np.ndarray]) -> CNNEnergy:
    conv_weight_keys = sorted(
        [
            name
            for name in named_params
            if name.startswith("conv.") and name.endswith(".weight")
        ],
        key=lambda name: int(name.split(".")[1]),
    )
    if len(conv_weight_keys) == 0:
        raise ValueError("Cannot restore CNNEnergy without convolution weight tensors.")

    channels = [named_params[key].shape[0] for key in conv_weight_keys]
    kernel_size = named_params[conv_weight_keys[0]].shape[-1]
    if "_image_shape" in named_params:
        image_shape = tuple(int(dim) for dim in named_params["_image_shape"])
    else:
        image_shape = None

    if "base.data_mean" in named_params:
        data_mean = torch.as_tensor(named_params["base.data_mean"])
        data_std = torch.as_tensor(named_params["base.data_std"])
        num_visibles = data_mean.shape[0]
    elif image_shape is not None:
        data_mean = None
        data_std = None
        num_visibles = image_shape[0] * image_shape[1]
    else:
        raise ValueError("Cannot restore CNNEnergy without base stats or saved image shape.")

    return CNNEnergy(
        num_visibles=num_visibles,
        hidden_dims=channels,
        image_shape=image_shape,
        kernel_size=kernel_size,
        data_mean=data_mean,
        data_std=data_std,
        output_bias="head.bias" in named_params,
    )
