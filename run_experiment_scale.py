import os


def run_experiment(
    scale_of_prior: int,
    is_sgld: bool = True,
    use_prior: bool = True,
    n_samples_in_dataset: int = 1000,
    seed: int = 1,
    fn: str = "filename",
    prior_path: str = "priors/resnet50_ssl_prior/resnet50_ssl_prior",
):

    train_command = (
        "python3 BayesianTransferLearning/prior_run_jobs.py"
        + f" --job=supervised_bayesian_learning "
        + f" --local_dir=downsampled_data/flowers_{n_samples_in_dataset}/"
        + f" --data_dir=downsampled_data/flowers_{n_samples_in_dataset}/"
        + f" --train_dir=train/"
        + f" --val_dir=test/"
        + f" --train_dataset=oxford102flowers"
        + f" --val_dataset=oxford102flowers"
        + f" --encoder=resnet50 --gpus=1 --num_of_labels=102"
        + f" --ignore_wandb --seed={seed}"
    )

    # batchsize --batch_size=16 --epochs=1000
    TRAINING_STEPS = 3
    batch_size = min(16, n_samples_in_dataset)
    epochs = TRAINING_STEPS // (n_samples_in_dataset // batch_size)

    train_command += f" --epochs={epochs} --batch_size={batch_size} --fn={fn}"

    if use_prior:
        train_command += f" --prior_path={prior_path}"

    else:
        train_command += " --prior_type=normal"

    if is_sgld:
        train_command += " --is_sgld"

    train_command += f" --prior_scale={scale_of_prior} --weight_decay=1e-4"

    print(
        f"\nRUNNING EXPERIMENT: \n"
        + f"    seed: {seed}\n"
        + f"    is_sgld: {is_sgld}\n"
        + f"    use_prior: {use_prior}\n"
        + f"    n_samples: {n_samples_in_dataset}\n"
        + f"    epochs: {epochs}\n"
        + f"    batch_size: {batch_size}\n"
        + f"    step_per_epoch: {n_samples_in_dataset//batch_size}\n"
        + f"    prior_scale: {scale_of_prior}\n"
        + f"    weight_decay: 1e-4\n"
    )

    os.system(command=train_command)


if __name__ == "__main__":

    SCALE = list(range(10))
    SEEDS = list(range(5))
    PRIOR_PATH = [
        "priors/resnet50_ssl_prior/resnet50_ssl_prior",
        "priors/resnet50_torchvision/resnet50_torchvision",
    ]

    FILENAME = "results_scale.txt"

    with open(FILENAME, "w") as f:
        f.write(
            ", ".join(
                [
                    "seed",
                    "prior_scale",
                    "prior_type",
                    "prior_rank",
                    "is_bayesian",
                    "dataset_size",
                    "prior",
                    "accuracy",
                    "\n",
                ]
            )
        )

    for scale in SCALE:
        for prior_path in PRIOR_PATH:
            for seed in SEEDS:
                run_experiment(
                    10 ** (scale), seed=seed, fn=FILENAME, prior_path=prior_path
                )
