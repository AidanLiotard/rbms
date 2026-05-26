import h5py
import numpy as np
import torch

from rbms.EBM_binary import BEBM, build_energy, get_visible_field_from_data
from rbms.EBM_continuous import (
    CEBM,
    build_energy as build_continuous_energy,
    get_gaussian_base_from_data,
)
from rbms.classes import EBM
from rbms.dataset.dataset_class import RBMDataset
from rbms.io import load_model, save_model
from rbms.map_model import map_model
from rbms.utils import get_saved_updates
from torch import Tensor


def _init_training(
    train_dataset: RBMDataset,
    seed: int,
    train_size: float,
    test_size: float,
    num_hiddens: int,
    hidden_dims: list[int] | None,
    num_chains: int,
    model_type: str,
    energy_type: str,
    filename: str,
    n_save: int,
    n_save_model: int | None,
    n_save_chain: int | None,
    n_save_metric: int | None,
    spacing: str,
    batch_size: int,
    optim: str,
    mult_optim: bool,
    training_type: str,
    learning_rate: float,
    max_lr: float,
    gibbs_steps: int,
    beta: float,
    centered: bool,
    L1: float,
    L2: float,
    normalize_grad: bool,
    max_norm_grad: float,
    subset_labels: list,
    use_weights: bool,
    alphabet: str,
    remove_duplicates: bool,
    dtype: torch.dtype,
    device: torch.device | str,
    flags: list[str],
    base_std_floor: float = 0.2,
    hmc_step_size: float | None = None,
    hmc_step_size_target: float | None = None,
    hmc_step_size_rate: float | None = None,
    hmc_step_size_warmup: int | None = None,
    hmc_num_leapfrog_steps: int | None = None,
    hmc_mass: float | None = None,
    sampling_kernel: str | None = None,
    nuts_max_delta_energy: float | None = None,
    map_model: dict[str, type[EBM]] = map_model,
):
    if model_type is None:
        match train_dataset.variable_type:
            case "bernoulli":
                model_type = "BBRBM"
            case "categorical":
                model_type = "PBRBM"
            case "ising":
                model_type = "IIRBM"
            case _:
                raise NotImplementedError()

    train_dataset.match_model_variable_type(map_model[model_type].visible_type)
    # Setup dataset
    num_visibles = train_dataset.get_num_visibles()

    # Setup model
    if hidden_dims is None:
        hidden_dims = [num_hiddens]

    if model_type == "BEBM":
        visible_field = get_visible_field_from_data(
            data=train_dataset.data,
            weights=train_dataset.weights,
        )

        match energy_type:
            case "mlp" | "mlp_no_w2" | "mlp_silu_no_w2" | "mlp_sigmoid_no_w2":
                energy = build_energy(
                    energy_type=energy_type,
                    num_visibles=num_visibles,
                    device=device,
                    dtype=dtype,
                    hidden_dims=hidden_dims,
                    visible_field=visible_field,
                )

            case "rbm":
                energy = build_energy(
                    energy_type="rbm",
                    num_visibles=num_visibles,
                    device=device,
                    dtype=dtype,
                    hidden_dim=num_hiddens,
                    visible_bias=visible_field,
                )

            case _:
                raise ValueError(f"Unknown BEBM energy type: {energy_type}")

        params = BEBM(
            energy=energy,
            num_visibles=num_visibles,
            device=device,
            dtype=dtype,
        )

    elif model_type == "CEBM":
        data_mean, data_std = get_gaussian_base_from_data(
            data=train_dataset.data,
            weights=train_dataset.weights,
        )

        match energy_type:
            case None | "mlp" | "cnn" | "rbm":
                energy = build_continuous_energy(
                    energy_type="mlp" if energy_type is None else energy_type,
                    num_visibles=num_visibles,
                    device=device,
                    dtype=dtype,
                    hidden_dims=hidden_dims,
                    data_mean=data_mean,
                    data_std=data_std,
                    base_std_floor=base_std_floor,
                )

            case "gaussian":
                energy = build_continuous_energy(
                    energy_type="gaussian",
                    num_visibles=num_visibles,
                    device=device,
                    dtype=dtype,
                    data_mean=data_mean,
                    data_std=data_std,
                    base_std_floor=base_std_floor,
                )

            case _:
                raise ValueError(f"Unknown CEBM energy type: {energy_type}")

        params = CEBM(
            energy=energy,
            num_visibles=num_visibles,
            device=device,
            dtype=dtype,
        )

    else:
        params = map_model[model_type].init_parameters(
            num_hiddens=num_hiddens,
            dataset=train_dataset,
            device=device,
            dtype=dtype,
        )

    if hasattr(params.energy, "calibrate_final_layer"):
        scale = params.energy.calibrate_final_layer(
            data=train_dataset.data,
            weights=train_dataset.weights,
            target_std=0.05,
        )
        print(f"Calibrated final energy layer by scale factor {scale:.6g}")

    sampler_kernel = None
    sampler_kernel_params = {}
    if model_type == "CEBM":
        sampler_kernel = "hmc" if sampling_kernel is None else sampling_kernel
        if hmc_step_size is not None:
            sampler_kernel_params["step_size"] = hmc_step_size
        if hmc_step_size_target is not None:
            sampler_kernel_params["step_size_target"] = hmc_step_size_target
        if hmc_step_size_rate is not None:
            sampler_kernel_params["step_size_rate"] = hmc_step_size_rate
        if hmc_step_size_warmup is not None:
            sampler_kernel_params["step_size_warmup"] = hmc_step_size_warmup
        if hmc_num_leapfrog_steps is not None:
            sampler_kernel_params["num_leapfrog_steps"] = hmc_num_leapfrog_steps
        if hmc_mass is not None:
            sampler_kernel_params["mass"] = hmc_mass
        if nuts_max_delta_energy is not None:
            sampler_kernel_params["max_delta_energy"] = nuts_max_delta_energy

    # Permanent chains
    parallel_chains = params.init_chains(num_samples=num_chains)
    if sampler_kernel is None:
        parallel_chains = params.sample_state(chains=parallel_chains, n_steps=gibbs_steps)
    else:
        parallel_chains = params.sample_state(
            chains=parallel_chains,
            n_steps=gibbs_steps,
            kernel=sampler_kernel,
            kernel_params=sampler_kernel_params,
        )

    # Save hyperparameters
    if mult_optim:
        lr = torch.tensor([learning_rate] * len(params.parameters()))
    else:
        lr = torch.tensor([learning_rate])

    with h5py.File(filename, "w") as file_model:
        hyperparameters = file_model.create_group("hyperparameters")
        hyperparameters["num_visibles"] = num_visibles
        hyperparameters["num_hiddens"] = num_hiddens
        hyperparameters["hidden_dims"] = hidden_dims
        hyperparameters["num_chains"] = num_chains
        hyperparameters["filename"] = str(filename)
        hyperparameters["energy_type"] = np.asarray(energy_type, dtype="T")
        hyperparameters["base_std_floor"] = base_std_floor

    save_model(
        filename=filename,
        params=params,
        chains=parallel_chains,
        num_updates=1,
        time=0.0,
        flags=flags,
        learning_rate=lr,
    )

    with h5py.File(filename, "a") as f:
        dataset = f.create_group("dataset_args")
        if subset_labels is not None:
            dataset["subset_labels"] = subset_labels
        dataset["use_weights"] = use_weights
        dataset["train_size"] = train_size
        dataset["test_size"] = test_size
        dataset["alphabet"] = np.asarray(alphabet, dtype="T")
        dataset["remove_duplicates"] = remove_duplicates
        dataset["seed"] = seed

        grad = f.create_group("grad_args")
        grad["no_center"] = not (centered)
        grad["normalize_grad"] = normalize_grad
        grad["max_norm_grad"] = max_norm_grad
        grad["L1"] = L1
        grad["L2"] = L2

        sampling = f.create_group("sampling_args")
        sampling["gibbs_steps"] = gibbs_steps
        sampling["beta"] = beta
        if hmc_step_size is not None:
            sampling["hmc_step_size"] = hmc_step_size
        if hmc_step_size_target is not None:
            sampling["hmc_step_size_target"] = hmc_step_size_target
        if hmc_step_size_rate is not None:
            sampling["hmc_step_size_rate"] = hmc_step_size_rate
        if hmc_step_size_warmup is not None:
            sampling["hmc_step_size_warmup"] = hmc_step_size_warmup
        if hmc_num_leapfrog_steps is not None:
            sampling["hmc_num_leapfrog_steps"] = hmc_num_leapfrog_steps
        if hmc_mass is not None:
            sampling["hmc_mass"] = hmc_mass
        if sampling_kernel is not None:
            sampling["sampling_kernel"] = np.asarray(sampling_kernel, dtype="T")
        if nuts_max_delta_energy is not None:
            sampling["nuts_max_delta_energy"] = nuts_max_delta_energy

        train_args = f.create_group("train_args")
        train_args["optim"] = np.asarray(optim, dtype="T")
        train_args["batch_size"] = batch_size
        train_args["learning_rate"] = lr
        train_args["training_type"] = np.asarray(training_type, dtype="T")
        train_args["max_lr"] = max_lr

        save_args = f.create_group("save_args")
        save_args["n_save"] = n_save
        if n_save_model is not None:
            save_args["n_save_model"] = n_save_model
        if n_save_chain is not None:
            save_args["n_save_chain"] = n_save_chain
        if n_save_metric is not None:
            save_args["n_save_metric"] = n_save_metric
        save_args["spacing"] = np.asarray(spacing, dtype="T")


def _restore_training(
    filename: str,
    train_dataset: RBMDataset,
    test_dataset: RBMDataset | None,
    num_updates: int,
    target_update: int,
    seed: int,
    train_size: float,
    test_size: float,
    device: str,
    dtype: torch.dtype,
    map_model: dict[str, type[EBM]] = map_model,
) -> tuple[EBM, dict[str, Tensor], int, float, RBMDataset, RBMDataset]:
    # Retrieve the the number of training updates already performed on the model
    print(f"Restoring training from update {target_update}")

    if num_updates <= target_update:
        raise RuntimeError(
            f"The parameter /'num_updates/' ({num_updates}) must be greater than the previous number of updates ({target_update})."
        )

    params, parallel_chains, elapsed_time = load_model(
        filename,
        target_update,
        device=device,
        dtype=dtype,
        restore=True,
        map_model=map_model,
    )

    # Delete all updates after the current one
    saved_updates = get_saved_updates(filename)
    if saved_updates[-1] > target_update:
        to_delete = saved_updates[saved_updates > target_update]
        with h5py.File(filename, "a") as f:
            print("Deleting:")
            for upd in to_delete:
                print(f" - {upd}")
                del f[f"update_{upd}"]

    if test_dataset is None:
        print("Splitting dataset")
        train_dataset, test_dataset = train_dataset.split_train_test(
            rng=np.random.default_rng(seed),
            train_size=train_size,
            test_size=test_size,
        )
        print("Train dataset:")
        print(train_dataset)
        print("Test dataset:")
        print(test_dataset)

    # Initialize gradients for the parameters
    params.init_grad()

    train_dataset.match_model_variable_type(params.visible_type)
    test_dataset.match_model_variable_type(params.visible_type)
    return (
        params,
        parallel_chains,
        target_update,
        elapsed_time,
        train_dataset,
        test_dataset,
    )
