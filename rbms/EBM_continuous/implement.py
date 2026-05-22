from __future__ import annotations

import math

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


def _kinetic_energy(
    momentum: Tensor,
    mass: float,
) -> Tensor:
    return 0.5 * momentum.square().flatten(start_dim=1).sum(dim=1) / mass


def _log_joint(
    potential_energy: Tensor,
    momentum: Tensor,
    mass: float,
) -> Tensor:
    return -potential_energy - _kinetic_energy(momentum, mass)


def _leapfrog_step(
    energy: torch.nn.Module,
    visible: Tensor,
    momentum: Tensor,
    grad: Tensor,
    step_size: float,
    direction: int,
    beta: float,
    mass: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Perform one leapfrog step in direction +1 or -1."""

    eps = float(direction) * step_size

    momentum_half = momentum - 0.5 * eps * grad
    visible_new = visible + eps * momentum_half / mass
    energy_new, grad_new = _energy_and_grad(
        energy=energy,
        visible=visible_new,
        beta=beta,
    )
    momentum_new = momentum_half - 0.5 * eps * grad_new

    return (
        visible_new.detach(),
        momentum_new.detach(),
        energy_new.detach(),
        grad_new.detach(),
    )


def _is_not_u_turn(
    visible_minus: Tensor,
    visible_plus: Tensor,
    momentum_minus: Tensor,
    momentum_plus: Tensor,
) -> Tensor:
    """Batchwise NUTS no-u-turn criterion."""

    delta = visible_plus - visible_minus

    flat_delta = delta.flatten(start_dim=1)
    flat_momentum_minus = momentum_minus.flatten(start_dim=1)
    flat_momentum_plus = momentum_plus.flatten(start_dim=1)

    criterion_minus = (flat_delta * flat_momentum_minus).sum(dim=1) >= 0
    criterion_plus = (flat_delta * flat_momentum_plus).sum(dim=1) >= 0

    return criterion_minus & criterion_plus


def _build_tree_nuts(
    energy: torch.nn.Module,
    visible: Tensor,
    momentum: Tensor,
    grad: Tensor,
    log_u: Tensor,
    direction: int,
    depth: int,
    step_size: float,
    beta: float,
    mass: float,
    initial_log_joint: Tensor,
    max_delta_energy: float,
) -> dict[str, Tensor]:
    """Recursive NUTS tree builder.

    This implementation is vectorized over chains. Some chains may already have
    stopped according to the no-u-turn criterion, but the recursion still builds
    batched tensors. The final candidate-selection masks ensure only valid chains
    are updated.
    """

    if depth == 0:
        (
            visible_prime,
            momentum_prime,
            energy_prime,
            grad_prime,
        ) = _leapfrog_step(
            energy=energy,
            visible=visible,
            momentum=momentum,
            grad=grad,
            step_size=step_size,
            direction=direction,
            beta=beta,
            mass=mass,
        )

        log_joint_prime = _log_joint(
            potential_energy=energy_prime,
            momentum=momentum_prime,
            mass=mass,
        )

        valid = log_u <= log_joint_prime
        not_divergent = (log_u - max_delta_energy) < log_joint_prime

        acceptance_prob = torch.exp(
            torch.minimum(
                torch.zeros_like(log_joint_prime),
                log_joint_prime - initial_log_joint,
            )
        )

        return {
            "visible_minus": visible_prime,
            "momentum_minus": momentum_prime,
            "grad_minus": grad_prime,
            "visible_plus": visible_prime,
            "momentum_plus": momentum_prime,
            "grad_plus": grad_prime,
            "visible_candidate": visible_prime,
            "n_valid": valid.float(),
            "continue_tree": not_divergent,
            "acceptance_sum": acceptance_prob,
            "n_alpha": torch.ones_like(acceptance_prob),
        }

    left = _build_tree_nuts(
        energy=energy,
        visible=visible,
        momentum=momentum,
        grad=grad,
        log_u=log_u,
        direction=direction,
        depth=depth - 1,
        step_size=step_size,
        beta=beta,
        mass=mass,
        initial_log_joint=initial_log_joint,
        max_delta_energy=max_delta_energy,
    )

    if direction == -1:
        right = _build_tree_nuts(
            energy=energy,
            visible=left["visible_minus"],
            momentum=left["momentum_minus"],
            grad=left["grad_minus"],
            log_u=log_u,
            direction=direction,
            depth=depth - 1,
            step_size=step_size,
            beta=beta,
            mass=mass,
            initial_log_joint=initial_log_joint,
            max_delta_energy=max_delta_energy,
        )

        visible_minus = right["visible_minus"]
        momentum_minus = right["momentum_minus"]
        grad_minus = right["grad_minus"]

        visible_plus = left["visible_plus"]
        momentum_plus = left["momentum_plus"]
        grad_plus = left["grad_plus"]

    else:
        right = _build_tree_nuts(
            energy=energy,
            visible=left["visible_plus"],
            momentum=left["momentum_plus"],
            grad=left["grad_plus"],
            log_u=log_u,
            direction=direction,
            depth=depth - 1,
            step_size=step_size,
            beta=beta,
            mass=mass,
            initial_log_joint=initial_log_joint,
            max_delta_energy=max_delta_energy,
        )

        visible_minus = left["visible_minus"]
        momentum_minus = left["momentum_minus"]
        grad_minus = left["grad_minus"]

        visible_plus = right["visible_plus"]
        momentum_plus = right["momentum_plus"]
        grad_plus = right["grad_plus"]

    n_total = left["n_valid"] + right["n_valid"]

    choose_right_prob = right["n_valid"] / torch.clamp(n_total, min=1.0)
    choose_right = torch.rand_like(choose_right_prob) < choose_right_prob

    visible_candidate = torch.where(
        choose_right[:, None],
        right["visible_candidate"],
        left["visible_candidate"],
    )

    not_u_turn = _is_not_u_turn(
        visible_minus=visible_minus,
        visible_plus=visible_plus,
        momentum_minus=momentum_minus,
        momentum_plus=momentum_plus,
    )

    continue_tree = left["continue_tree"] & right["continue_tree"] & not_u_turn

    return {
        "visible_minus": visible_minus,
        "momentum_minus": momentum_minus,
        "grad_minus": grad_minus,
        "visible_plus": visible_plus,
        "momentum_plus": momentum_plus,
        "grad_plus": grad_plus,
        "visible_candidate": visible_candidate,
        "n_valid": n_total,
        "continue_tree": continue_tree,
        "acceptance_sum": left["acceptance_sum"] + right["acceptance_sum"],
        "n_alpha": left["n_alpha"] + right["n_alpha"],
    }


def _sample_state_hmc(
    energy: torch.nn.Module,
    chains: dict[str, Tensor],
    n_steps: int,
    beta: float = 1.0,
    step_size: float = 1e-2,
    num_leapfrog_steps: int = 10,
    mass: float = 1.0,
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
    momentum_std = mass**0.5

    for _ in range(n_steps):
        start_visible = visible.detach()
        start_momentum = momentum_std * torch.randn_like(start_visible)

        current_energy, grad = _energy_and_grad(
            energy=energy,
            visible=start_visible,
            beta=beta,
        )
        current_kinetic = _kinetic_energy(start_momentum, mass=mass)

        proposal_visible = start_visible
        proposal_momentum = start_momentum - 0.5 * step_size * grad

        for leapfrog_step in range(num_leapfrog_steps):
            proposal_visible = proposal_visible + step_size * proposal_momentum / mass

            proposed_energy, grad = _energy_and_grad(
                energy=energy,
                visible=proposal_visible,
                beta=beta,
            )

            if leapfrog_step != num_leapfrog_steps - 1:
                proposal_momentum = proposal_momentum - step_size * grad

        proposal_momentum = proposal_momentum - 0.5 * step_size * grad
        proposal_momentum = -proposal_momentum

        proposed_kinetic = _kinetic_energy(proposal_momentum, mass=mass)
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

    return {
        "visible": visible.detach(),
        "visible_mag": visible.detach(),
        "weights": weights,
        "acceptance": torch.stack(acceptances).mean() if acceptances else torch.nan,
    }


def _sample_state_nuts(
    energy: torch.nn.Module,
    chains: dict[str, Tensor],
    n_steps: int,
    beta: float = 1.0,
    step_size: float = 1e-2,
    num_leapfrog_steps: int = 64,
    mass: float = 1.0,
    max_delta_energy: float = 1000.0,
) -> dict[str, Tensor]:
    """Sample visible chains using No-U-Turn Sampler.

    Args:
        energy:
            Energy module E(x).
        chains:
            Dictionary containing at least "visible" and "weights".
        n_steps:
            Number of NUTS transitions.
        beta:
            Inverse temperature multiplying the energy.
        step_size:
            Leapfrog step size.
        num_leapfrog_steps:
            Approximate maximum number of leapfrog steps. Internally converted
            to max_tree_depth = ceil(log2(num_leapfrog_steps)).
        mass:
            Scalar HMC mass.
        max_delta_energy:
            Divergence threshold. Larger values tolerate larger Hamiltonian error.

    Returns:
        Updated chain dictionary.
    """

    if step_size <= 0:
        raise ValueError(f"NUTS step_size must be positive, got {step_size}.")
    if num_leapfrog_steps <= 0:
        raise ValueError(
            f"NUTS num_leapfrog_steps must be positive, got {num_leapfrog_steps}."
        )
    if mass <= 0:
        raise ValueError(f"NUTS mass must be positive, got {mass}.")

    visible = chains["visible"].clone()
    weights = chains["weights"].clone()

    max_tree_depth = max(1, int(math.ceil(math.log2(num_leapfrog_steps))))
    momentum_std = mass**0.5

    acceptances = []
    tree_depths = []

    for _ in range(n_steps):
        start_visible = visible.detach()
        start_momentum = momentum_std * torch.randn_like(start_visible)

        current_energy, current_grad = _energy_and_grad(
            energy=energy,
            visible=start_visible,
            beta=beta,
        )

        initial_log_joint = _log_joint(
            potential_energy=current_energy,
            momentum=start_momentum,
            mass=mass,
        )

        log_u = initial_log_joint + torch.log(torch.rand_like(initial_log_joint))

        visible_minus = start_visible
        visible_plus = start_visible
        momentum_minus = start_momentum
        momentum_plus = start_momentum
        grad_minus = current_grad
        grad_plus = current_grad

        visible_candidate = start_visible

        n_valid = torch.ones_like(initial_log_joint)
        continue_tree = torch.ones_like(initial_log_joint, dtype=torch.bool)

        acceptance_sum = torch.zeros_like(initial_log_joint)
        n_alpha = torch.zeros_like(initial_log_joint)

        depth_reached = torch.zeros_like(initial_log_joint)

        for depth in range(max_tree_depth):
            direction = -1 if torch.rand(()) < 0.5 else 1

            if direction == -1:
                tree = _build_tree_nuts(
                    energy=energy,
                    visible=visible_minus,
                    momentum=momentum_minus,
                    grad=grad_minus,
                    log_u=log_u,
                    direction=direction,
                    depth=depth,
                    step_size=step_size,
                    beta=beta,
                    mass=mass,
                    initial_log_joint=initial_log_joint,
                    max_delta_energy=max_delta_energy,
                )

                visible_minus = tree["visible_minus"]
                momentum_minus = tree["momentum_minus"]
                grad_minus = tree["grad_minus"]

            else:
                tree = _build_tree_nuts(
                    energy=energy,
                    visible=visible_plus,
                    momentum=momentum_plus,
                    grad=grad_plus,
                    log_u=log_u,
                    direction=direction,
                    depth=depth,
                    step_size=step_size,
                    beta=beta,
                    mass=mass,
                    initial_log_joint=initial_log_joint,
                    max_delta_energy=max_delta_energy,
                )

                visible_plus = tree["visible_plus"]
                momentum_plus = tree["momentum_plus"]
                grad_plus = tree["grad_plus"]

            accept_new_prob = tree["n_valid"] / torch.clamp(
                n_valid + tree["n_valid"],
                min=1.0,
            )
            accept_new = torch.rand_like(accept_new_prob) < accept_new_prob
            accept_new = accept_new & continue_tree

            visible_candidate = torch.where(
                accept_new[:, None],
                tree["visible_candidate"],
                visible_candidate,
            )

            n_valid = n_valid + tree["n_valid"]

            not_u_turn = _is_not_u_turn(
                visible_minus=visible_minus,
                visible_plus=visible_plus,
                momentum_minus=momentum_minus,
                momentum_plus=momentum_plus,
            )

            continue_tree = continue_tree & tree["continue_tree"] & not_u_turn

            acceptance_sum = acceptance_sum + tree["acceptance_sum"]
            n_alpha = n_alpha + tree["n_alpha"]

            depth_reached = torch.where(
                continue_tree,
                torch.full_like(depth_reached, float(depth + 1)),
                depth_reached,
            )

            if not bool(continue_tree.any()):
                break

        visible = visible_candidate.detach()

        transition_acceptance = acceptance_sum / torch.clamp(n_alpha, min=1.0)
        acceptances.append(transition_acceptance.mean())
        tree_depths.append(depth_reached.float().mean())

    return {
        "visible": visible.detach(),
        "visible_mag": visible.detach(),
        "weights": weights,
        "acceptance": torch.stack(acceptances).mean() if acceptances else torch.nan,
        "nuts_tree_depth": torch.stack(tree_depths).mean() if tree_depths else torch.nan,
    }


def sample_state(
    energy: torch.nn.Module,
    chains: dict[str, Tensor],
    n_steps: int,
    sampler: str = "hmc",
    beta: float = 1.0,
    step_size: float = 1e-2,
    num_leapfrog_steps: int = 10,
    mass: float = 1.0,
    max_delta_energy: float = 1000.0,
) -> dict[str, Tensor]:
    """Dispatch visible-state sampling.

    sampler:
        "hmc"  -> fixed-length Hamiltonian Monte Carlo.
        "nuts" -> No-U-Turn Sampler.
    """

    match sampler:
        case "hmc":
            return _sample_state_hmc(
                energy=energy,
                chains=chains,
                n_steps=n_steps,
                beta=beta,
                step_size=step_size,
                num_leapfrog_steps=num_leapfrog_steps,
                mass=mass,
            )

        case "nuts":
            return _sample_state_nuts(
                energy=energy,
                chains=chains,
                n_steps=n_steps,
                beta=beta,
                step_size=step_size,
                num_leapfrog_steps=num_leapfrog_steps,
                mass=mass,
                max_delta_energy=max_delta_energy,
            )

        case _:
            raise ValueError(
                f"Unknown sampler '{sampler}'. Available samplers are: 'hmc', 'nuts'."
            )