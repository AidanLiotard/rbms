from __future__ import annotations
 
import math
import torch
from torch import Tensor

def _energy_and_grad(
    energy: torch.nn.Module,
    visible: Tensor,
    beta: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """U(x)=beta E(x), and grad U(x)."""
    x = visible.detach().requires_grad_(True)
    potential = beta * energy(x).view(-1)
    grad = torch.autograd.grad(potential.sum(), x)[0]
    x.requires_grad_(False)
    return potential.detach(), grad.detach()


def _kinetic_energy(momentum: Tensor, mass: Tensor) -> Tensor:
    """K(p)=1/2 sum_i p_i^2 / m_i."""
    return 0.5 * (momentum.square() / mass.view(1, -1)).sum(dim=1)


def _resolve_mass(
    energy: torch.nn.Module,
    visible: Tensor,
    mass: float | Tensor | None = None,
) -> Tensor:
    """Diagonal HMC mass."""
    if mass is None:
        base_std = getattr(getattr(energy, "base", energy), "data_std", None)
        if base_std is None:
            return torch.ones(visible.shape[1], device=visible.device, dtype=visible.dtype)
        return torch.as_tensor(base_std, device=visible.device, dtype=visible.dtype).flatten().square()

    mass = torch.as_tensor(mass, device=visible.device, dtype=visible.dtype).flatten()
    if mass.numel() == 1:
        mass = mass.expand(visible.shape[1])
    return mass

def _sample_state_hmc(
    energy: torch.nn.Module,
    chains: dict[str, Tensor],
    n_steps: int,
    beta: float = 1.0,
    step_size: float = 1e-2,
    num_leapfrog_steps: int = 10,
    mass: float | Tensor | None = None,
    **_,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """HMC for p(x) proportional to exp(-beta E(x))."""
    visible = chains["visible"].detach().clone()
    weights = chains.get("weights", None)

    mass_tensor = _resolve_mass(energy=energy, visible=visible, mass=mass)
    inv_mass = 1.0 / mass_tensor.view(1, -1)
    momentum_std = mass_tensor.sqrt().view(1, -1)

    acceptances = []

    for _ in range(n_steps):
        start_visible = visible.detach()
        start_momentum = momentum_std * torch.randn_like(start_visible)

        current_energy, grad = _energy_and_grad(
            energy=energy,
            visible=start_visible,
            beta=beta,
        )
        current_kinetic = _kinetic_energy(start_momentum, mass_tensor)

        proposal_visible = start_visible
        proposal_momentum = start_momentum - 0.5 * step_size * grad

        for leapfrog_step in range(num_leapfrog_steps):
            proposal_visible = proposal_visible + step_size * proposal_momentum * inv_mass
            proposed_energy, grad = _energy_and_grad(
                energy=energy,
                visible=proposal_visible,
                beta=beta,
            )
            if leapfrog_step != num_leapfrog_steps - 1:
                proposal_momentum = proposal_momentum - step_size * grad

        proposal_momentum = proposal_momentum - 0.5 * step_size * grad
        proposal_momentum = -proposal_momentum

        proposed_kinetic = _kinetic_energy(proposal_momentum, mass_tensor)
        log_acceptance = (
            -proposed_energy
            - proposed_kinetic
            + current_energy
            + current_kinetic
        )

        with torch.no_grad():
            accept = torch.log(torch.rand_like(log_acceptance)) < log_acceptance
            visible = torch.where(accept[:, None], proposal_visible.detach(), start_visible)
            acceptances.append(accept.float().mean())

    sampled = {
        "visible": visible.detach(),
        "visible_mag": visible.detach(),
    }
    if weights is not None:
        sampled["weights"] = weights
    info = {"acceptance": torch.stack(acceptances).mean()}
    torch.cuda.empty_cache()
    return sampled, info


def _sample_state_hmc_adapt(
    energy: torch.nn.Module,
    chains: dict[str, Tensor],
    n_steps: int,
    beta: float = 1.0,
    step_size: float = 1e-2,
    num_leapfrog_steps: int = 3,
    mass: float | Tensor | None = None,
    target_acceptance: float = 0.65,
    adapt_step_size: bool = True,
    adapt_rate: float = 0.05,
    min_step_size: float = 1e-5,
    max_step_size: float = 0.2,
    step_size_target: float | None = None,
    step_size_rate: float | None = None,
    step_size_warmup: int | None = None,
    **_,
):
    """Adaptive HMC for p(x) proportional to exp(-beta E(x))."""

    # Compatibility with CLI/kernel_params names.
    if step_size_target is not None:
        target_acceptance = float(step_size_target)

    if step_size_rate is not None:
        adapt_rate = float(step_size_rate)

    visible = chains["visible"].detach().clone()
    weights = chains.get("weights", None)

    # Optional compatibility if some caller stores step_size in chains.
    if "step_size" in chains:
        step_size = float(chains["step_size"].detach().cpu().item())

    # Clip after every possible source of step_size has been resolved.
    step_size = min(max(float(step_size), min_step_size), max_step_size)

    mass_tensor = _resolve_mass(energy=energy, visible=visible, mass=mass)
    inv_mass = 1.0 / mass_tensor.view(1, -1)
    momentum_std = mass_tensor.sqrt().view(1, -1)

    acceptances = []

    for _ in range(n_steps):
        start_visible = visible.detach()
        start_momentum = momentum_std * torch.randn_like(start_visible)

        current_energy, grad = _energy_and_grad(
            energy=energy,
            visible=start_visible,
            beta=beta,
        )
        current_kinetic = _kinetic_energy(start_momentum, mass_tensor)

        proposal_visible = start_visible
        proposal_momentum = start_momentum - 0.5 * step_size * grad

        for leapfrog_step in range(num_leapfrog_steps):
            proposal_visible = proposal_visible + step_size * proposal_momentum * inv_mass
            proposed_energy, grad = _energy_and_grad(
                energy=energy,
                visible=proposal_visible,
                beta=beta,
            )

            if leapfrog_step != num_leapfrog_steps - 1:
                proposal_momentum = proposal_momentum - step_size * grad

        proposal_momentum = proposal_momentum - 0.5 * step_size * grad
        proposal_momentum = -proposal_momentum

        proposed_kinetic = _kinetic_energy(proposal_momentum, mass_tensor)

        log_acceptance = (
            -proposed_energy
            - proposed_kinetic
            + current_energy
            + current_kinetic
        )

        with torch.no_grad():
            accept = torch.log(torch.rand_like(log_acceptance)) < log_acceptance
            visible = torch.where(
                accept[:, None],
                proposal_visible.detach(),
                start_visible,
            )
            acceptances.append(accept.float().mean())

    mean_acceptance = torch.stack(acceptances).mean()

    if adapt_step_size:
        step_size *= math.exp(
            adapt_rate * (float(mean_acceptance.item()) - target_acceptance)
        )
        step_size = min(max(step_size, min_step_size), max_step_size)

    sampled = {
        "visible": visible.detach(),
        "visible_mag": visible.detach(),
    }

    if weights is not None:
        sampled["weights"] = weights

    info = {
        "acceptance": mean_acceptance,
        "step_size": torch.tensor(
            step_size,
            device=visible.device,
            dtype=visible.dtype,
        ),
    }

    torch.cuda.empty_cache()
    return sampled, info


def _sample_state_mala(
    energy: torch.nn.Module,
    chains: dict[str, Tensor],
    n_steps: int,
    beta: float = 1.0,
    step_size: float = 1e-2,
    **_,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """MALA for p(x) proportional to exp(-beta E(x))."""
    visible = chains["visible"].detach().clone()
    weights = chains.get("weights", None)

    acceptances = []

    for _ in range(n_steps):
        start_visible = visible.detach()
        current_energy, current_grad = _energy_and_grad(
            energy=energy,
            visible=start_visible,
            beta=beta,
        )
        proposal_mean = (
            start_visible
            - 0.5 * step_size**2 * current_grad
        )
        proposal_visible = (
            proposal_mean
            + step_size * torch.randn_like(start_visible)
        )
        proposed_energy, proposed_grad = _energy_and_grad(
            energy=energy,
            visible=proposal_visible,
            beta=beta,
        )
        reverse_mean = (
            proposal_visible
            - 0.5 * step_size**2 * proposed_grad
        )
        forward_diff = proposal_visible - proposal_mean
        reverse_diff = start_visible - reverse_mean

        log_q_forward = -0.5 * torch.sum(
            forward_diff.pow(2) / step_size**2,
            dim=1,
        )
        log_q_reverse = -0.5 * torch.sum(
            reverse_diff.pow(2) / step_size**2,
            dim=1,
        )
        log_acceptance = (
            -proposed_energy
            + current_energy
            + log_q_reverse
            - log_q_forward
        )

        with torch.no_grad():
            accept = torch.log(torch.rand_like(log_acceptance)) < log_acceptance
            visible = torch.where(
                accept[:, None],
                proposal_visible.detach(),
                start_visible,
            )
            acceptances.append(accept.float().mean())

    sampled = {
        "visible": visible.detach(),
        "visible_mag": visible.detach(),
    }

    if weights is not None:
        sampled["weights"] = weights

    info = {"acceptance": torch.stack(acceptances).mean()}
    torch.cuda.empty_cache()

    return sampled, info


def sample_state(
    energy: torch.nn.Module,
    chains: dict[str, Tensor],
    n_steps: int,
    sampler: str = "hmc",
    beta: float = 1.0,
    step_size: float = 1e-2,
    num_leapfrog_steps: int = 10,
    mass: float | Tensor | None = None,
    **kwargs,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Sampler dispatch."""

    if sampler == "hmc":
        return _sample_state_hmc(
            energy=energy,
            chains=chains,
            n_steps=n_steps,
            beta=beta,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            mass=mass,
            **kwargs,
        )

    if sampler == "hmc_adapt":
        return _sample_state_hmc_adapt(
            energy=energy,
            chains=chains,
            n_steps=n_steps,
            beta=beta,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            mass=mass,
            **kwargs,
        )

    if sampler == "mala":
        return _sample_state_mala(
            energy=energy,
            chains=chains,
            n_steps=n_steps,
            beta=beta,
            step_size=step_size,
            **kwargs,
        )

    raise ValueError(f"Unknown sampler: {sampler!r}")