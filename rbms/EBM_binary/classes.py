from __future__ import annotations

import copy

import numpy as np
import torch
from torch import Tensor

from rbms.EBM_binary.energies import IndependentBernoulliEnergy, restore_energy
from rbms.EBM_binary.implement import _sample_state_dmala

from rbms.classes import EBM


class BEBM(EBM):
    """ Parameters of the binary EBM"""

    name: str
    device: torch.device | str | None
    visible_type: str = "bernoulli"
    flags: list[str]
    energy: torch.nn.Module

    def __init__(
        self,
        energy: torch.nn.Module,
        num_visibles: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ): 
        """Initialize the parameters of the binary EBM.

        Args:
            energy (torch.nn.Module): The energy function of the EBM.
            num_visibles (int): The number of visible units.
            device (Optional[torch.device], optional): The device for the parameters.
            dtype (Optional[torch.dtype], optional): The data type for the parameters.
        """
        first_param = next(energy.parameters(), None)

        if device is None:
            if first_param is None:
                device = torch.device("cpu")
            else:
                device = first_param.device

        if dtype is None:
            if first_param is None:
                dtype = torch.get_default_dtype()
            else:
                dtype = first_param.dtype
        self.device = device
        self.dtype = dtype
        self.energy = energy.to(device=self.device, dtype=self.dtype)
        self._num_visibles = num_visibles
        self.name = "BEBM"
        self.flags = []

    def __add__(self, other: EBM) -> EBM:
        """Add the parameters of two EBMs. Useful for interpolation"""
        raise NotImplementedError("Addition of EBMs is not implemented yet.") 

    def __mul__(self, other: float) -> EBM:
        """Multiplies the parameters of the EBM by a float."""
        raise NotImplementedError("Multiplication of EBMs is not implemented yet.")
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EBM):
            return False
        other_params = other.named_parameters()
        for k, v in self.named_parameters().items():
            if not np.equal(other_params[k], v).all():
                return False
        return True

    def compute_energy_visibles(self, v: Tensor) -> Tensor:
        """Returns the marginalized energy of the model computed on the visible configurations

        Args:
            v (Tensor): Visible configurations

        Returns:
            Tensor: The computed energy.
        """
        v = v.to(device=self.device, dtype=self.dtype)
        return self.energy(v).view(-1)

    def init_chains(
        self,
        num_samples: int,
        weights: Tensor | None = None,
        start_v: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Initialize a Markov chain for the EBM.

        Args:
            num_samples (int): The number of samples to initialize.
            start_v (Tensor, optional): The initial visible states. Defaults to None.

        Returns:
            dict[str, Tensor]: The initialized Markov chain.

        Notes:
            - If start_v is specified, its number of samples will override the num_samples argument.
        """
        if num_samples <= 0:
            if start_v is not None:
                num_samples = start_v.shape[0]
            else:
                raise ValueError(f"Got negative num_samples arg: {num_samples}")

        if start_v is None:
            mean_visible = (
                torch.ones(
                    size=(num_samples, self.num_visibles),
                    device=self.device,
                    dtype=self.dtype,
                )
                / 2
            )
            visible = torch.bernoulli(mean_visible)
        else:
            mean_visible = (
                torch.ones_like(start_v, device=self.device, dtype=self.dtype) / 2
            )
            visible = start_v.to(device=self.device, dtype=self.dtype)

        if weights is None:
            weights = torch.ones(
                visible.shape[0],
                device=visible.device,
                dtype=visible.dtype,
            )
        else:
            weights = weights.to(device=self.device, dtype=self.dtype).view(-1)

        return dict(
            visible=visible,
            visible_mag=mean_visible,
            weights=weights
        )
        
    def compute_gradient(
        self,
        data: dict[str, Tensor],
        chains: dict[str, Tensor],
        centered: bool = True,
    ) -> None:
        """Compute the gradient for each of the parameters and attach it.

        Args:
            data (dict[str, Tensor]): The data state.
            chains (dict[str, Tensor]): The parallel chains used for gradient computation.
            centered (bool, optional): Whether to use centered gradients. Defaults to True.
        """
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
        """Returns a list containing the parameters of the BEBM.

        Returns:
            list[Tensor]: The parameters of the BEBM.
        """
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
        energy = restore_energy(
            named_params=named_params,
            device=device,
            dtype=dtype,
        )

        return BEBM(
            energy=energy,
            num_visibles=energy.num_visibles,
            device=device,
            dtype=dtype,
        )

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> BEBM:
        """Move the parameters to the specified device and/or convert them to the specified data type.

        Args:
            device (Optional[torch.device], optional): The device to move the parameters to.
                Defaults to None.
            dtype (Optional[torch.dtype], optional): The data type to convert the parameters to.
                Defaults to None.

        Returns:
            EBM: The modified EBM instance.
        """
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype

        self.energy = self.energy.to(device=self.device, dtype=self.dtype)
        return self

    def clone(
        self, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> EBM:
        """Create a clone of the EBM instance.

        Args:
            device (Optional[torch.device], optional): The device for the cloned parameters.
                Defaults to the current device.
            dtype (Optional[torch.dtype], optional): The data type for the cloned parameters.
                Defaults to the current data type.

        Returns:
            EBM: A new EBM instance with cloned parameters.
        """
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        
        return BEBM(
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
        """Initialize the parameters of the BEBM.

        Args:
            dataset : Training dataset.
            device (torch.device): PyTorch device for the parameters.
            dtype (torch.dtype): PyTorch dtype for the parameters.
            var_init (float, optional): Variance of the weight matrix. Defaults to 1e-4.
        """
        raise NotImplementedError("Initialization of parameters from dataset is not implemented yet.")    
    
    @property
    def num_visibles(self) -> int:
        """Number of visible units"""
        return self._num_visibles

    @property
    def ref_log_z(self) -> float:
        """Reference log partition function with weights set to 0 (except for the visible bias)."""
        return torch.nn.functional.softplus(self.energy.visible_field).sum().item()
    
    def independent_model(self) -> EBM:
        """Independent model where only local fields are preserved."""
        energy = IndependentBernoulliEnergy(
            visible_field=self.energy.visible_field.detach().clone()
        )

        return BEBM(
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
        **kwargs
    ) -> dict[str, Tensor]:
        """Sample the model for n_steps

        Args:
            chains (): The starting position of the chains.
            n_steps (int): The number of sampling steps.
            beta (float, optional): The inverse temperature. Defaults to 1.0
            kernel (Optional[Kernel]): The Markov kernel to use for sampling. Defaults to None.
            kernel_params (Optional[dict]): The parameters for the Markov kernel. Defaults to None.

        Returns:
            dict[str, Tensor]: The updated chains after n_steps of sampling.
        """
        kernel: str | None = kwargs.pop("kernel", None)
        kernel_params: dict | None = kwargs.pop("kernel_params", {})

        if kernel_params is None:
            kernel_params = {}
        kernel_params = {**kernel_params, **kwargs}

        if kernel is None:
            kernel = "dmala"
        
        new_chains = {
            "visible": chains["visible"].clone(),
            "weights": chains["weights"].clone(),
        }

        match kernel:
            case "dmala":
                return _sample_state_dmala(
                    energy=self.energy,
                    chains=new_chains,
                    n_steps=n_steps,
                    beta=beta,
                    **kernel_params,
                )
            case _:
                raise NotImplementedError(f"Seulement dmala as of now.")

    def get_metrics(self, metrics: dict[str, float]) -> dict[str, float]: return metrics

    def pre_grad_update(self) -> None: pass

    def post_grad_update(self) -> None: pass

    @property
    def effective_number_variables(self) -> float: return self.num_visibles
