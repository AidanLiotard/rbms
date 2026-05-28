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


class _ResidualDownsampleBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        downsample: bool,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.downsample = bool(downsample)
        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            torch.nn.SiLU(),
            torch.nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
        )
        if self.downsample:
            self.main_pool = torch.nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)
            self.skip = torch.nn.Sequential(
                torch.nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True),
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            )
        elif in_channels != out_channels:
            self.main_pool = torch.nn.Identity()
            self.skip = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.main_pool = torch.nn.Identity()
            self.skip = torch.nn.Identity()
        self.act = torch.nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.skip(x)
        out = self.main(x)
        out = self.main_pool(out)
        return self.act(out + residual)


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


class MLPEnergy(torch.nn.Module):
    """Continuous visible-state energy: Gaussian base + visible field + MLP residual."""
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
        self.num_visibles = int(num_visibles)
        self.hidden_dims = _normalize_hidden_dims(
            hidden_dims=hidden_dims,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.hidden_dim = self.hidden_dims[-1]
        self.num_layers = len(self.hidden_dims)
        self.base_std_floor = float(base_std_floor)

        if data_mean is None:
            data_mean = torch.zeros(num_visibles)
        if data_std is None:
            data_std = torch.ones(num_visibles)
        if visible_field is None:
            visible_field = torch.zeros(
                num_visibles,
                device=data_mean.device,
                dtype=data_mean.dtype,
            )

        self.register_buffer("data_mean", data_mean.clone())
        self.register_buffer("data_std", data_std.clone().clamp_min(self.base_std_floor))
        self.visible_field = torch.nn.Parameter(visible_field.clone())

        layers = []
        in_dim = num_visibles
        for out_dim in self.hidden_dims:
            layers.append(torch.nn.Linear(in_dim, out_dim))
            layers.append(torch.nn.SiLU())
            in_dim = out_dim
        layers.append(torch.nn.Linear(in_dim, 1, bias=output_bias))

        self.net = torch.nn.Sequential(*layers)
        _init_mlp_layers(self.net)

    @property
    def visible_std(self) -> Tensor:
        return self.data_std

    def forward(self, x: Tensor) -> Tensor:
        return self.E_beta(x)

    def E_visible_gaussian(self, x: Tensor) -> Tensor:
        z = (x - self.data_mean) / self.data_std
        return 0.5 * z.square().sum(dim=1)

    def E_gauss(self, x: Tensor) -> Tensor:
        return self.E_visible_gaussian(x)

    def E_visible_field(self, x: Tensor) -> Tensor:
        return -x @ self.visible_field

    def E_nn(self, x: Tensor) -> Tensor:
        return self.net(x).view(-1)

    def E_beta(self, x: Tensor, beta: float = 1.0) -> Tensor:
        return self.E_visible_gaussian(x) + self.E_visible_field(x) + beta * self.E_nn(x)

    @property
    def ref_log_z(self) -> Tensor:
        std = self.data_std
        log_two_pi = torch.log(
            torch.as_tensor(2.0 * torch.pi, device=std.device, dtype=std.dtype)
        )
        log_z_gauss = 0.5 * self.num_visibles * log_two_pi + torch.log(std).sum()
        field_shift = torch.dot(self.visible_field, self.data_mean) + 0.5 * (
            std * self.visible_field
        ).square().sum()
        return log_z_gauss + field_shift

    def sample_independent(self, num_samples: int) -> Tensor:
        std = self.data_std
        mean = self.data_mean + std.square() * self.visible_field
        eps = torch.randn(
            num_samples,
            self.num_visibles,
            device=std.device,
            dtype=std.dtype,
        )
        return mean.view(1, -1) + std.view(1, -1) * eps

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
    """Continuous flattened-image energy: Gaussian base + visible field + CNN residual."""

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
        architecture: str = "residual",
    ):
        super().__init__()

        self.num_visibles = int(num_visibles)
        self.architecture = architecture
        self.hidden_dims = _normalize_hidden_dims(
            hidden_dims=hidden_dims,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.hidden_dim = self.hidden_dims[-1]
        self.num_layers = len(self.hidden_dims)
        self.kernel_size = int(kernel_size)
        self.base_std_floor = float(base_std_floor)

        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}.")

        if image_shape is None:
            image_shape = _infer_square_image_shape(self.num_visibles)

        image_shape = tuple(int(dim) for dim in image_shape)

        if len(image_shape) != 2 or any(dim <= 0 for dim in image_shape):
            raise ValueError(f"image_shape must be a positive (height, width), got {image_shape}.")

        if image_shape[0] * image_shape[1] != self.num_visibles:
            raise ValueError(
                f"image_shape={image_shape} is incompatible with "
                f"num_visibles={self.num_visibles}."
            )

        self.image_shape = image_shape
        self.register_buffer("_image_shape", torch.tensor(image_shape, dtype=torch.int64))

        if data_mean is None:
            data_mean = torch.zeros(self.num_visibles)

        if data_std is None:
            data_std = torch.ones(self.num_visibles)

        if visible_field is None:
            visible_field = torch.zeros(
                self.num_visibles,
                device=data_mean.device,
                dtype=data_mean.dtype,
            )

        self.register_buffer("data_mean", data_mean.clone())
        self.register_buffer("data_std", data_std.clone().clamp_min(self.base_std_floor))

        self.visible_field = torch.nn.Parameter(visible_field.clone())

        if architecture == "legacy":
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

        else:
            self.stem = torch.nn.Conv2d(
                1,
                self.hidden_dims[0],
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
            )

            blocks = []
            in_channels = self.hidden_dims[0]

            for idx, out_channels in enumerate(self.hidden_dims):
                blocks.append(
                    _ResidualDownsampleBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=self.kernel_size,
                        downsample=idx < len(self.hidden_dims) - 1,
                    )
                )
                in_channels = out_channels

            self.blocks = torch.nn.ModuleList(blocks)

            with torch.no_grad():
                dummy = torch.zeros(1, 1, *self.image_shape)
                dummy_features = self.stem(dummy)

                for block in self.blocks:
                    dummy_features = block(dummy_features)

                flattened_dim = dummy_features.flatten(start_dim=1).shape[1]

            head_layers: list[torch.nn.Module] = []
            head_hidden_dims = self.hidden_dims[1:] or [self.hidden_dims[0]]
            head_in_dim = flattened_dim

            for head_out_dim in head_hidden_dims:
                head_layers.append(torch.nn.Linear(head_in_dim, head_out_dim))
                head_layers.append(torch.nn.SiLU())
                head_in_dim = head_out_dim

            head_layers.append(torch.nn.Linear(head_in_dim, 1, bias=output_bias))

            self.head = torch.nn.Sequential(*head_layers)

            _init_cnn_layers(self.stem, self.blocks, self.head)

    @property
    def visible_std(self) -> Tensor:
        return self.data_std

    def _score(self, x: Tensor) -> Tensor:
        image = x.view(x.shape[0], 1, *self.image_shape)

        if self.architecture == "legacy":
            features = self.pool(self.conv(image)).flatten(start_dim=1)
        else:
            features = self.stem(image)

            for block in self.blocks:
                features = block(features)

            features = features.flatten(start_dim=1)

        return self.head(features).view(-1)

    def forward(self, x: Tensor) -> Tensor:
        return self.E_beta(x)

    def E_visible_gaussian(self, x: Tensor) -> Tensor:
        z = (x - self.data_mean) / self.data_std
        return 0.5 * z.square().sum(dim=1)

    def E_gauss(self, x: Tensor) -> Tensor:
        return self.E_visible_gaussian(x)

    def E_visible_field(self, x: Tensor) -> Tensor:
        return -x @ self.visible_field

    def E_nn(self, x: Tensor) -> Tensor:
        return self._score(x)

    def E_beta(self, x: Tensor, beta: float = 1.0) -> Tensor:
        return (
            self.E_visible_gaussian(x)
            + self.E_visible_field(x)
            + beta * self.E_nn(x)
        )

    @property
    def ref_log_z(self) -> Tensor:
        std = self.data_std
        log_two_pi = torch.log(
            torch.as_tensor(2.0 * torch.pi, device=std.device, dtype=std.dtype)
        )
        log_z_gauss = 0.5 * self.num_visibles * log_two_pi + torch.log(std).sum()
        field_shift = torch.dot(self.visible_field, self.data_mean) + 0.5 * (
            std * self.visible_field
        ).square().sum()
        return log_z_gauss + field_shift

    def sample_independent(self, num_samples: int) -> Tensor:
        std = self.data_std
        mean = self.data_mean + std.square() * self.visible_field

        eps = torch.randn(
            num_samples,
            self.num_visibles,
            device=std.device,
            dtype=std.dtype,
        )

        return mean.view(1, -1) + std.view(1, -1) * eps

    def calibrate_final_layer(
        self,
        data: Tensor,
        weights: Tensor | None = None,
        target_std: float = 0.05,
        batch_size: int = 4096,
        eps: float = 1e-12,
    ) -> float:
        linear_layers = [
            module for module in self.head.modules()
            if isinstance(module, torch.nn.Linear)
        ]
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
                final_layer.weight.mul_(scale)
                return float(scale.detach().cpu())

        return 1.0
    

@torch.compile
class RBMEnergy(torch.nn.Module):
    """Fixed-variance Gaussian-visible / Bernoulli-hidden RBM.
    """

    def __init__(
        self,
        num_visibles: int,
        num_hiddens: int = 256,
        log_visible_std: Tensor | None = None,
        visible_field: Tensor | None = None,
        hidden_bias: Tensor | None = None,
        weight_scale: float = 1e-3,
        min_std: float = 1e-4,
        **_,
    ):
        super().__init__()

        self.num_visibles = int(num_visibles)
        self.num_hiddens = int(num_hiddens)
        self.min_std = float(min_std)

        if log_visible_std is None:
            log_visible_std = torch.zeros(num_visibles)

        if visible_field is None:
            visible_field = torch.zeros(
                num_visibles,
                device=log_visible_std.device,
                dtype=log_visible_std.dtype,
            )

        if hidden_bias is None:
            hidden_bias = torch.zeros(
                num_hiddens,
                device=log_visible_std.device,
                dtype=log_visible_std.dtype,
            )

        self.register_buffer("log_visible_std", log_visible_std.clone())

        self.visible_field = torch.nn.Parameter(visible_field.clone())
        self.hidden_bias = torch.nn.Parameter(hidden_bias.clone())

        self.weight = torch.nn.Parameter(
            weight_scale
            * torch.randn(
                self.num_hiddens,
                self.num_visibles,
                device=log_visible_std.device,
                dtype=log_visible_std.dtype,
            )
        )

    @property
    def visible_std(self) -> Tensor:
        return torch.nn.functional.softplus(self.log_visible_std) + self.min_std

    def forward(self, x: Tensor) -> Tensor:
        return self.E_beta(x)

    def E_visible_gaussian(self, x: Tensor) -> Tensor:
        std = self.visible_std
        z = x / std
        return 0.5 * z.square().sum(dim=1)

    def E_visible_field(self, x: Tensor) -> Tensor:
        return -x @ self.visible_field

    def E_nn(self, x: Tensor) -> Tensor:
        hidden_pre_activation = x @ self.weight.T + self.hidden_bias
        return -torch.nn.functional.softplus(hidden_pre_activation).sum(dim=1)

    def E_beta(self, x: Tensor, beta: float = 1.0) -> Tensor:
        return (
            self.E_visible_gaussian(x)
            + self.E_visible_field(x)
            + beta * self.E_nn(x)
        )

    @property
    def ref_log_z(self) -> Tensor:
        std = self.visible_std
        d = torch.as_tensor(
            self.num_visibles,
            device=std.device,
            dtype=std.dtype,
        )
        log_two_pi = torch.log(
            torch.as_tensor(2.0 * torch.pi, device=std.device, dtype=std.dtype)
        )
        return (
            0.5 * d * log_two_pi
            + torch.log(std).sum()
            + 0.5 * (std * self.visible_field).square().sum()
        )

    def sample_independent(self, num_samples: int) -> Tensor:
        std = self.visible_std
        mean = std.square() * self.visible_field

        eps = torch.randn(
            num_samples,
            self.num_visibles,
            device=std.device,
            dtype=std.dtype,
        )

        return mean.view(1, -1) + std.view(1, -1) * eps
    

ENERGY_MAP: dict[str, type[torch.nn.Module]] = {
    "mlp": MLPEnergy,
    "cnn": CNNEnergy,
    "rbm": RBMEnergy,
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
    else:
        if energy_type == "rbm":
            hidden_dims = energy_kwargs.pop("hidden_dims", None)
            if hidden_dims is not None:
                energy_kwargs["num_hiddens"] = int(hidden_dims[0])

            energy_kwargs.pop("data_mean", None)

            data_std = energy_kwargs.pop("data_std", None)
            base_std_floor = energy_kwargs.pop("base_std_floor", 1e-4)

            if data_std is not None:
                data_std = torch.clamp(data_std, min=base_std_floor)
                energy_kwargs["log_visible_std"] = torch.log(torch.expm1(data_std))

            energy_kwargs["min_std"] = base_std_floor

            energy = RBMEnergy(num_visibles=num_visibles, **energy_kwargs)
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
        case "mlp":
            energy = restore_mlp_energy(named_params)
        case "cnn":
            energy = restore_cnn_energy(named_params)
        case "rbm":
            energy = restore_rbm_energy(named_params)
        case _:
            raise ValueError(
                f"Cannot restore unknown continuous energy type '{energy_type}'. "
                f"Available parameter keys: {list(named_params.keys())}."
            )

    state_dict = {
        name: torch.as_tensor(array, device=device, dtype=dtype)
        for name, array in named_params.items()
    }
    energy.load_state_dict(state_dict)
    return energy.to(device=device, dtype=dtype)


def identify_energy_type(named_params: dict[str, np.ndarray]) -> str:
    keys = set(named_params)

    match keys:
        case keys if any(
            name.startswith("conv.") or name.startswith("stem.") or name.startswith("blocks.")
            for name in keys
        ):
            return "cnn"
        case keys if any(name.startswith("net.") for name in keys):
            return "mlp"
        case keys if {"log_visible_std", "visible_field", "hidden_bias", "weight"} <= keys:
            return "rbm"
        case _:
            raise ValueError(
                "Could not identify continuous EBM energy type from saved parameters. "
                f"Available keys: {list(named_params.keys())}."
            )


def restore_rbm_energy(named_params):
    return RBMEnergy(
        num_visibles=named_params["log_visible_std"].shape[0],
        num_hiddens=named_params["hidden_bias"].shape[0],
        log_visible_std=torch.as_tensor(named_params["log_visible_std"]),
        visible_field=torch.as_tensor(named_params["visible_field"]),
        hidden_bias=torch.as_tensor(named_params["hidden_bias"]),
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
        data_mean=torch.as_tensor(named_params["data_mean"]),
        data_std=torch.as_tensor(named_params["data_std"]),
        visible_field=torch.as_tensor(named_params["visible_field"]),
        output_bias=final_bias_key in named_params,
    )


def restore_cnn_energy(named_params: dict[str, np.ndarray]) -> CNNEnergy:
    legacy_conv_weight_keys = sorted(
        [
            name
            for name in named_params
            if name.startswith("conv.") and name.endswith(".weight")
        ],
        key=lambda name: int(name.split(".")[1]),
    )
    new_block_weight_keys = sorted(
        [
            name
            for name in named_params
            if name.startswith("blocks.") and name.endswith("main.0.weight")
        ],
        key=lambda name: int(name.split(".")[1]),
    )
    if len(new_block_weight_keys) > 0:
        channels = [named_params[key].shape[0] for key in new_block_weight_keys]
        kernel_size = named_params[new_block_weight_keys[0]].shape[-1]
        architecture = "residual"
        head_weight_keys = sorted(
            [
                name
                for name in named_params
                if name.startswith("head.") and name.endswith(".weight")
            ],
            key=lambda name: int(name.split(".")[1]),
        )
        final_head_bias = head_weight_keys[-1].replace(".weight", ".bias")
        output_bias = final_head_bias in named_params
    elif len(legacy_conv_weight_keys) > 0:
        channels = [named_params[key].shape[0] for key in legacy_conv_weight_keys]
        kernel_size = named_params[legacy_conv_weight_keys[0]].shape[-1]
        architecture = "legacy"
        output_bias = "head.bias" in named_params
    else:
        raise ValueError("Cannot restore CNNEnergy without convolution weight tensors.")

    if "_image_shape" in named_params:
        image_shape = tuple(int(dim) for dim in named_params["_image_shape"])
    else:
        image_shape = None

    if "data_mean" in named_params:
        data_mean = torch.as_tensor(named_params["data_mean"])
        data_std = torch.as_tensor(named_params["data_std"])
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
        visible_field=torch.as_tensor(named_params["visible_field"]),
        output_bias=output_bias,
        architecture=architecture,
    )
