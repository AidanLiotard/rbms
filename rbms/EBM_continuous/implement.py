import torch
from torch import Tensor


def _energy_and_grad(
    energy: torch.nn.Module,
    visible: Tensor,
    beta: float,
) -> tuple[Tensor, Tensor]:
    visible_grad_input = visible.detach().requires_grad_(True)
    energy_value = beta * energy(visible_grad_input).view(-1)
    grad = torch.autograd.grad(energy_value.sum(), visible_grad_input)[0]
    return energy_value.detach(), grad.detach()


def _sample_state_hmc(
    energy: torch.nn.Module,
    chains: dict[str, Tensor],
    n_steps: int,
    beta: float = 1.0,
    step_size: float = 1e-2,
    num_leapfrog_steps: int = 10,
    mass: float = 1.0,
    clamp: tuple[float, float] | None = None,
) -> dict[str, Tensor]:
    visible = chains["visible"].clone()
    weights = chains["weights"].clone()

    if step_size <= 0:
        raise ValueError(f"HMC step_size must be positive, got {step_size}.")
    if num_leapfrog_steps <= 0:
        raise ValueError(
            f"HMC num_leapfrog_steps must be positive, got {num_leapfrog_steps}."
        )
    if mass <= 0:
        raise ValueError(f"HMC mass must be positive, got {mass}.")

    acceptances = []
    momentum_std = mass ** 0.5

    for _ in range(n_steps):
        start_visible = visible.detach()
        start_momentum = momentum_std * torch.randn_like(start_visible)

        current_energy, grad = _energy_and_grad(energy, start_visible, beta=beta)
        current_kinetic = 0.5 * start_momentum.square().sum(dim=1) / mass

        proposal_visible = start_visible
        proposal_momentum = start_momentum - 0.5 * step_size * grad

        for leapfrog_step in range(num_leapfrog_steps):
            proposal_visible = proposal_visible + step_size * proposal_momentum / mass

            if clamp is not None:
                lo, hi = clamp
                proposal_visible = proposal_visible.clamp(lo, hi)

            proposed_energy, grad = _energy_and_grad(
                energy,
                proposal_visible,
                beta=beta,
            )

            if leapfrog_step != num_leapfrog_steps - 1:
                proposal_momentum = proposal_momentum - step_size * grad

        proposal_momentum = proposal_momentum - 0.5 * step_size * grad
        proposal_momentum = -proposal_momentum

        proposed_kinetic = 0.5 * proposal_momentum.square().sum(dim=1) / mass
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

    return {
        "visible": visible.detach(),
        "visible_mag": visible.detach(),
        "weights": weights,
        "acceptance": torch.stack(acceptances).mean() if acceptances else torch.nan,
    }
