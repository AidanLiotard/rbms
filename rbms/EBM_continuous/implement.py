from __future__ import annotations
 
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
) -> dict[str, Tensor]:
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
        "acceptance": torch.stack(acceptances).mean(),
    }
    if weights is not None:
        sampled["weights"] = weights
    return sampled


def sample_state(
    energy: torch.nn.Module,
    chains: dict[str, Tensor],
    n_steps: int,
    sampler: str = "hmc",
    beta: float = 1.0,
    step_size: float = 1e-2,
    num_leapfrog_steps: int = 10,
    mass: float | Tensor | None = None,
    **_,
) -> dict[str, Tensor]:
    """Sampler dispatch. HMC only."""
    if sampler != "hmc":
        raise NotImplementedError("Only sampler='hmc' is implemented for continuous EBMs.")

    return _sample_state_hmc(
        energy=energy,
        chains=chains,
        n_steps=n_steps,
        beta=beta,
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps,
        mass=mass,
    )
