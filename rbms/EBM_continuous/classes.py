from __future__ import annotations

import copy

import numpy as np
import torch
from torch import Tensor

from rbms.classes import EBM
from rbms.EBM_continuous.implement import _sample_state_hmc, _sample_state_nuts


class _ModelEnergyProxy(torch.nn.Module):
    def __init__(self, model: "CEBM"):
        super().__init__()
        self.model = model
        self.base = getattr(model.energy, "base", model.energy)

    def forward(self, v: Tensor) -> Tensor:
        return self.model.compute_energy_visibles(v)


class CEBM(EBM):
    """Continuous visible-state energy-based model."""

    name: str
    device: torch.device | str | None
    visible_type: str = "continuous"
    flags: list[str]
    energy: torch.nn.Module

    def __init__(
        self,
        energy: torch.nn.Module,
        num_visibles: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        beta: float = 1.0,
        interpolation_beta: float | None = None,
    ):
        first_param = next(energy.parameters(), None)

        if device is None:
            device = torch.device("cpu") if first_param is None else first_param.device
        if dtype is None:
            dtype = torch.get_default_dtype() if first_param is None else first_param.dtype

        self.device = device
        self.dtype = dtype
        self.energy = energy.to(device=self.device, dtype=self.dtype)
        self._num_visibles = num_visibles
        self.name = "CEBM"
        self.flags = []
        self.last_acceptance: Tensor | None = None
        self.last_tree_depth: Tensor | None = None
        self.last_step_size: Tensor | None = None
        self.interpolation_beta = float(beta if interpolation_beta is None else interpolation_beta)


    @property
    def beta(self) -> float:
        """Backward-compatible alias for the CEBM interpolation coefficient.

        Prefer `interpolation_beta` in new code, because `beta` is also used
        as the sampler inverse-temperature argument in `sample_state`.
        """
        return self.interpolation_beta

    @beta.setter
    def beta(self, value: float) -> None:
        self.interpolation_beta = float(value)

    def __add__(self, other: EBM) -> EBM:
        return self.interpolated_model(self.interpolation_beta + other.interpolation_beta)
    
    def __mul__(self, other: float) -> EBM:
        return self.interpolated_model(self.interpolation_beta * other)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EBM):
            return False
        other_params = other.named_parameters()
        for k, v in self.named_parameters().items():
            if not np.equal(other_params[k], v).all():
                return False
        if hasattr(other, "interpolation_beta") and not np.isclose(self.interpolation_beta, other.interpolation_beta):
            return False
        return True

    def compute_energy_visibles(self, v: Tensor, beta: float | None = None) -> Tensor:
        v = v.to(device=self.device, dtype=self.dtype)
        if hasattr(self.energy, "E_beta"):
            if beta is None:
                beta = self.interpolation_beta
            return self.energy.E_beta(v, beta=beta).view(-1)
        return self.energy(v).view(-1)

    def init_chains(
        self,
        num_samples: int,
        weights: Tensor | None = None,
        start_v: Tensor | None = None,
    ) -> dict[str, Tensor]:
        data_mean, data_std = self._get_base_stats()
        if num_samples <= 0:
            if start_v is not None:
                num_samples = start_v.shape[0]
            else:
                raise ValueError(f"Got negative num_samples arg: {num_samples}")

        if start_v is None:
            # The analytic reference distribution for a beta=0 CEBM is
            #   E_0(x) = 0.5 * ||(x - mu) / sigma||^2 - b^T x,
            # hence x ~ N(mu + sigma^2 * b, diag(sigma^2)).
            # For beta != 0 this is only an initialization distribution.
            init_mean = data_mean
            visible_field = self._get_visible_field()
            if visible_field is not None:
                init_mean = data_mean + data_std.square() * visible_field
            visible = init_mean.view(1, -1) + data_std.view(1, -1) * torch.randn(
                size=(num_samples, self.num_visibles),
                device=self.device,
                dtype=self.dtype,
            )
        else:
            visible = start_v.to(device=self.device, dtype=self.dtype)
        mean_visible = visible

        if weights is None:
            weights = torch.ones(
                visible.shape[0],
                device=visible.device,
                dtype=visible.dtype,
            )
        else:
            weights = weights.to(device=self.device, dtype=self.dtype).view(-1)

        return {
            "visible": visible,
            "visible_mag": mean_visible,
            "weights": weights,
        }

    def compute_gradient(
        self,
        data: dict[str, Tensor],
        chains: dict[str, Tensor],
        centered: bool = True,
    ) -> None:
        v_data = data["visible"].to(device=self.device, dtype=self.dtype)
        v_chain = chains["visible"].to(device=self.device, dtype=self.dtype)
        w_data = data["weights"].to(device=self.device, dtype=self.dtype).view(-1)
        w_chain = chains["weights"].to(device=self.device, dtype=self.dtype).view(-1)

        data_weights = w_data / w_data.sum()
        chain_weights = w_chain / w_chain.sum()

        data_energy = self.compute_energy_visibles(v_data)
        chain_energy = self.compute_energy_visibles(v_chain)

        objective = -(data_energy * data_weights).sum() + (
            chain_energy * chain_weights
        ).sum()

        self.energy.zero_grad(set_to_none=True)
        objective.backward()

    def parameters(self) -> list[Tensor]:
        return list(self.energy.parameters())

    def named_parameters(self) -> dict[str, np.ndarray]:
        named_params = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in self.energy.state_dict().items()
        }
        named_params["interpolation_beta"] = np.asarray(self.interpolation_beta)
        return named_params

    @staticmethod
    def set_named_parameters(
        named_params: dict[str, np.ndarray],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> EBM:
        from rbms.EBM_continuous.energies import restore_energy

        named_params = dict(named_params)
        beta = float(named_params.pop("interpolation_beta", named_params.pop("beta", 1.0)))
        energy = restore_energy(
            named_params=named_params,
            device=device,
            dtype=dtype,
        )

        return CEBM(
            energy=energy,
            num_visibles=energy.num_visibles,
            device=device,
            dtype=dtype,
            interpolation_beta=beta,
        )

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> CEBM:
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype

        self.energy = self.energy.to(device=self.device, dtype=self.dtype)
        return self

    def clone(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> EBM:
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype

        return CEBM(
            energy=copy.deepcopy(self.energy),
            num_visibles=self.num_visibles,
            device=device,
            dtype=dtype,
            interpolation_beta=self.interpolation_beta,
        )

    def interpolated_model(self, beta: float) -> "CEBM":
        """Return the same CEBM energy family at interpolation value beta.

        The intended decomposition is
            E_beta(x) = E_gauss(x) + E_visible_field(x) + beta * E_nn(x).
        interpolation_beta=0 is the independent CEBM reference, interpolation_beta=1 is the full model.
        """

        new = self.clone(device=self.device, dtype=self.dtype)
        new.interpolation_beta = float(beta)
        return new

    @staticmethod
    def init_parameters(
        num_visibles: int,
        dataset,
        device: torch.device | str,
        dtype: torch.dtype,
        var_init: float = 1e-4,
    ) -> EBM:
        from rbms.EBM_continuous.energies import MLPEnergy, get_gaussian_base_from_data

        data_mean, data_std = get_gaussian_base_from_data(
            data=dataset.data,
            weights=dataset.weights,
        )
        energy = MLPEnergy(
            num_visibles=num_visibles,
            data_mean=data_mean,
            data_std=data_std,
        )
        return CEBM(energy=energy, num_visibles=num_visibles, device=device, dtype=dtype)

    @property
    def num_visibles(self) -> int:
        return self._num_visibles

    @property
    def ref_log_z(self) -> float:
        """Analytic log partition function of the beta=0 reference model.

        This is not the logZ of the full beta=1 CEBM. It is the reference
        normalizer used for beta-ladder AIS/PTT estimates.
        """

        return float(self.ref_log_z_beta0().detach().cpu())

    def ref_log_z_beta0(self) -> Tensor:
        """Return log Z for E_0(x)=E_gauss(x)-visible_field^T x.

        With diagonal Gaussian base parameters mu, sigma and field b,
            log Z_0 = D/2 log(2*pi) + sum(log sigma)
                      + b^T mu + 1/2 sum((sigma*b)^2).
        If no visible field exists, b=0 and this reduces to the Gaussian logZ.
        """

        data_mean, data_std = self._get_base_stats()
        data_mean = data_mean.to(device=self.device, dtype=self.dtype).view(-1)
        data_std = data_std.to(device=self.device, dtype=self.dtype).view(-1)
        visible_field = self._get_visible_field()
        if visible_field is None:
            visible_field = torch.zeros_like(data_mean)
        else:
            visible_field = visible_field.to(device=self.device, dtype=self.dtype).view(-1)

        log_two_pi = torch.log(
            torch.tensor(2.0 * torch.pi, device=self.device, dtype=self.dtype)
        )
        log_z_gauss = 0.5 * data_mean.numel() * log_two_pi + torch.log(data_std).sum()
        field_shift = torch.dot(visible_field, data_mean) + 0.5 * (data_std * visible_field).square().sum()
        return log_z_gauss + field_shift

    def independent_model(self) -> EBM:
        """Return the independent CEBM: beta=0 with visible field kept.

        This mirrors the independent RBM convention: local fields/biases remain,
        while the interaction/neural residual term is removed.
        """

        return self.interpolated_model(beta=0.0)

    def sample_state(
        self,
        chains: dict[str, Tensor],
        n_steps: int,
        beta: float = 1.0,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Sample the model for n_steps.

        Args:
            chains: The starting position of the chains.
            n_steps: The number of sampling steps.
            beta: The inverse temperature. Defaults to 1.0.
            kernel: The Markov kernel to use for sampling. Defaults to "hmc".
            kernel_params: Parameters forwarded to the Markov kernel.

        Returns:
            The updated chains after n_steps of sampling.
        """
        kernel: str | None = kwargs.pop("kernel", None)
        kernel_params: dict | None = kwargs.pop("kernel_params", {})

        if kernel_params is None:
            kernel_params = {}
        kernel_params = {**kernel_params, **kwargs}

        if kernel is None:
            kernel = "hmc"

        new_chains = {
            "visible": chains["visible"].clone(),
            "weights": chains["weights"].clone(),
        }

        tempered_energy = _ModelEnergyProxy(self)

        match kernel:
            case "hmc":
                sampled = _sample_state_hmc(
                    energy=tempered_energy,
                    chains=new_chains,
                    n_steps=n_steps,
                    beta=beta,
                    **kernel_params,
                )
                self.last_acceptance = sampled.get("acceptance")
                self.last_tree_depth = sampled.get("nuts_tree_depth")
                self.last_step_size = sampled.get("step_size")
                return sampled
            case "nuts":
                sampled = _sample_state_nuts(
                    energy=tempered_energy,
                    chains=new_chains,
                    n_steps=n_steps,
                    beta=beta,
                    **kernel_params,
                )
                self.last_acceptance = sampled.get("acceptance")
                self.last_tree_depth = sampled.get("nuts_tree_depth")
                self.last_step_size = sampled.get("step_size")
                return sampled
            case _:
                raise NotImplementedError(f"Unknown CEBM sampling kernel: {kernel}.")

    def get_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        if self.last_acceptance is not None:
            metrics["hmc_acceptance"] = float(self.last_acceptance.detach().cpu())
        if self.last_tree_depth is not None:
            metrics["nuts_tree_depth"] = float(self.last_tree_depth.detach().cpu())
        if self.last_step_size is not None:
            metrics["hmc_step_size"] = float(self.last_step_size.detach().cpu())
        return metrics

    def pre_grad_update(self) -> None:
        pass

    def post_grad_update(self) -> None:
        pass

    @property
    def effective_number_variables(self) -> float:
        return self.num_visibles

    def compute_base_energy(self, v: Tensor) -> Tensor:
        v = v.to(device=self.device, dtype=self.dtype)
        if hasattr(self.energy, "E_gauss"):
            return self.energy.E_gauss(v).view(-1)
        return torch.zeros(v.shape[0], device=v.device, dtype=v.dtype)

    def compute_visible_field_energy(self, v: Tensor) -> Tensor:
        v = v.to(device=self.device, dtype=self.dtype)
        if hasattr(self.energy, "E_visible_field"):
            return self.energy.E_visible_field(v).view(-1)
        visible_field = self._get_visible_field()
        if visible_field is None:
            return torch.zeros(v.shape[0], device=v.device, dtype=v.dtype)
        return -(v @ visible_field.to(device=v.device, dtype=v.dtype).view(-1))

    def compute_neural_energy(self, v: Tensor) -> Tensor:
        v = v.to(device=self.device, dtype=self.dtype)
        if hasattr(self.energy, "E_nn"):
            return self.energy.E_nn(v).view(-1)
        return self.compute_energy_visibles(v) - self.compute_base_energy(v) - self.compute_visible_field_energy(v)

    def _get_visible_field(self) -> Tensor | None:
        if hasattr(self.energy, "visible_field"):
            return self.energy.visible_field
        return None

    def _get_base_stats(self) -> tuple[Tensor, Tensor]:
        if hasattr(self.energy, "base"):
            return self.energy.base.data_mean, self.energy.base.data_std
        if hasattr(self.energy, "data_mean") and hasattr(self.energy, "data_std"):
            return self.energy.data_mean, self.energy.data_std
        return (
            torch.zeros(self.num_visibles, device=self.device, dtype=self.dtype),
            torch.ones(self.num_visibles, device=self.device, dtype=self.dtype),
        )
