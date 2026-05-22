from __future__ import annotations

import copy

import numpy as np
import torch
from torch import Tensor

from rbms.classes import EBM
from rbms.EBM_continuous.implement import _sample_state_hmc, _sample_state_nuts


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

    def __add__(self, other: EBM) -> EBM:
        raise NotImplementedError("Addition of CEBMs is not implemented yet.")

    def __mul__(self, other: float) -> EBM:
        raise NotImplementedError("Multiplication of CEBMs is not implemented yet.")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EBM):
            return False
        other_params = other.named_parameters()
        for k, v in self.named_parameters().items():
            if not np.equal(other_params[k], v).all():
                return False
        return True

    def compute_energy_visibles(self, v: Tensor) -> Tensor:
        v = v.to(device=self.device, dtype=self.dtype)
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
            visible = data_mean.view(1, -1) + data_std.view(1, -1) * torch.randn(
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

        data_energy = self.energy(v_data).view(-1)
        chain_energy = self.energy(v_chain).view(-1)

        objective = -(data_energy * data_weights).sum() + (
            chain_energy * chain_weights
        ).sum()

        self.energy.zero_grad(set_to_none=True)
        objective.backward()

    def parameters(self) -> list[Tensor]:
        return list(self.energy.parameters())

    def named_parameters(self) -> dict[str, np.ndarray]:
        return {
            name: tensor.detach().cpu().numpy()
            for name, tensor in self.energy.state_dict().items()
        }

    @staticmethod
    def set_named_parameters(
        named_params: dict[str, np.ndarray],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> EBM:
        from rbms.EBM_continuous.energies import restore_energy

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
        )

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
            visible_field=data_mean,
        )
        return CEBM(energy=energy, num_visibles=num_visibles, device=device, dtype=dtype)

    @property
    def num_visibles(self) -> int:
        return self._num_visibles

    @property
    def ref_log_z(self) -> float:
        independent = self.independent_model()
        return independent.energy.log_z.item()

    def independent_model(self) -> EBM:
        from rbms.EBM_continuous.energies import GaussianBaseEnergy

        data_mean, data_std = self._get_base_stats()
        std_floor = getattr(getattr(self.energy, "base", self.energy), "std_floor", 0.2)
        energy = GaussianBaseEnergy(
            data_mean=data_mean.detach().clone(),
            data_std=data_std.detach().clone(),
            std_floor=std_floor,
        )
        return CEBM(
            energy=energy,
            num_visibles=self.num_visibles,
            device=self.device,
            dtype=self.dtype,
        )

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

        match kernel:
            case "hmc":
                sampled = _sample_state_hmc(
                    energy=self.energy,
                    chains=new_chains,
                    n_steps=n_steps,
                    beta=beta,
                    **kernel_params,
                )
                self.last_acceptance = sampled.get("acceptance")
                self.last_tree_depth = sampled.get("nuts_tree_depth")
                return sampled
            case "nuts":
                sampled = _sample_state_nuts(
                    energy=self.energy,
                    chains=new_chains,
                    n_steps=n_steps,
                    beta=beta,
                    **kernel_params,
                )
                self.last_acceptance = sampled.get("acceptance")
                self.last_tree_depth = sampled.get("nuts_tree_depth")
                return sampled
            case _:
                raise NotImplementedError(f"Unknown CEBM sampling kernel: {kernel}.")

    def get_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        if self.last_acceptance is not None:
            metrics["hmc_acceptance"] = float(self.last_acceptance.detach().cpu())
        if self.last_tree_depth is not None:
            metrics["nuts_tree_depth"] = float(self.last_tree_depth.detach().cpu())
        return metrics

    def pre_grad_update(self) -> None:
        pass

    def post_grad_update(self) -> None:
        pass

    @property
    def effective_number_variables(self) -> float:
        return self.num_visibles

    def _get_base_stats(self) -> tuple[Tensor, Tensor]:
        if hasattr(self.energy, "base"):
            return self.energy.base.data_mean, self.energy.base.data_std
        if hasattr(self.energy, "data_mean") and hasattr(self.energy, "data_std"):
            return self.energy.data_mean, self.energy.data_std
        return (
            torch.zeros(self.num_visibles, device=self.device, dtype=self.dtype),
            torch.ones(self.num_visibles, device=self.device, dtype=self.dtype),
        )
