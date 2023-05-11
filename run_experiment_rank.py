import os


def run_experiment(
    rank_of_prior: int,
    is_sgld: bool = True,
    use_prior: bool = True,
    n_samples_in_dataset: int = 1000,
    seed: int = 1,
    fn: str = "filename",
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
    TRAINING_STEPS = 30
    batch_size = min(16, n_samples_in_dataset)
    epochs = TRAINING_STEPS // (n_samples_in_dataset // batch_size)

    train_command += f" --epochs={epochs} --batch_size={batch_size} --fn={fn}"

    train_command += (
        " --load_prior=True --prior_path=priors/resnet50_ssl_prior/resnet50_ssl_prior"
    )
    train_command += " --is_sgld --prior_scale=1e5 --scale_low_rank=True"

    train_command += (
        f" --number_of_samples={rank_of_prior} --weight_decay=1e-4 --lr=0.01"
    )

    print(
        f"\nRUNNING EXPERIMENT: \n"
        + f"    seed: {seed}\n"
        + f"    is_sgld: {is_sgld}\n"
        + f"    use_prior: {use_prior}\n"
        + f"    n_sampls: {n_samples_in_dataset}\n"
        + f"    epochs: {epochs}\n"
        + f"    batch_size: {batch_size}\n"
        + f"    step_per_epoch: {n_samples_in_dataset//batch_size}\n"
        + f"    number_of_samples: {rank_of_prior}"
    )

    print(train_command)
    os.system(command=train_command)


if __name__ == "__main__":

    RANK = list(range(1, 6))
    SEEDS = range(5)

    FILENAME = "results_rank.txt"

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

    for rank in RANK:
        for seed in SEEDS:
            run_experiment(rank, seed=seed, fn=FILENAME)
