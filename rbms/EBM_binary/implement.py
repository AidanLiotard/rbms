import torch
import torch.nn.functional
from torch import Tensor


def _sample_state_dmala(
    energy: torch.nn.Module,
    chains: dict[str, Tensor],
    n_steps: int,
    beta: float = 1.0,
    alpha: float = 0.25,
) -> dict[str, Tensor]:
    
    visible = chains["visible"].clone()
    weights = chains["weights"].clone()

    for _ in range(n_steps):
        visible.requires_grad_(True)

        current_energy = energy(visible).view(-1)
        grad = torch.autograd.grad(
            current_energy.sum(),
            visible,
        )[0]

        forward_logits = (
            -0.5 * beta * grad
            + (2.0 * visible - 1.0) / (2.0 * alpha)
        )

        with torch.no_grad():
            forward_prob = torch.sigmoid(forward_logits)
            proposal = torch.bernoulli(forward_prob)

        proposal_grad_input = proposal.detach().requires_grad_(True)
        proposal_energy = energy(proposal_grad_input).view(-1)
        proposal_grad = torch.autograd.grad(
            proposal_energy.sum(),
            proposal_grad_input,
        )[0]

        reverse_logits = (
            -0.5 * beta * proposal_grad
            + (2.0 * proposal_grad_input - 1.0) / (2.0 * alpha)
        )

        with torch.no_grad():
            log_q_forward = -torch.nn.functional.binary_cross_entropy_with_logits(
                forward_logits.detach(),
                proposal,
                reduction="none",
            ).sum(1)

            log_q_reverse = -torch.nn.functional.binary_cross_entropy_with_logits(
                reverse_logits.detach(),
                visible,
                reduction="none",
            ).sum(1)

            log_acceptance = (
                -beta * proposal_energy.detach()
                + beta * current_energy.detach()
                + log_q_reverse
                - log_q_forward
            )

            accept = torch.log(torch.rand_like(log_acceptance)) < log_acceptance
            visible = torch.where(accept[:, None], proposal, visible)

    return {
        "visible": visible,
        "visible_mag": visible,
        "weights": weights,
    }
