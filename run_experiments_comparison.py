import os


def run_experiment(
    is_sgld: bool,
    use_prior: bool,
    n_samples_in_dataset: int,
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

    TRAINING_STEPS = 30000
    batch_size = min(16, n_samples_in_dataset)
    epochs = TRAINING_STEPS // (n_samples_in_dataset // batch_size)

    train_command += f" --epochs={epochs} --batch_size={batch_size} --weight_decay=1e-4 --lr=0.01 --fn={fn}"

    train_command += " --prior_path=priors/resnet50_ssl_prior/resnet50_ssl_prior"

    train_command += f" --number_of_samples_prior={'5' if use_prior else '1'} --prior_scale=1e5 --scale_low_rank=True"

    if is_sgld:
        train_command += " --is_sgld"

    if not use_prior:
        train_command += " --prior_type=normal"

    print(
        f"\nRUNNING EXPERIMENT: \n"
        + f"    seed: {seed}\n"
        + f"    is_sgld: {is_sgld}\n"
        + f"    use_prior: {use_prior}\n"
        + f"    n_sampls: {n_samples_in_dataset}\n"
        + f"    epochs: {epochs}\n"
        + f"    batch_size: {batch_size}\n"
        + f"    step_per_epoch: {n_samples_in_dataset//batch_size}\n"
    )
    print(train_command)

    os.system(command=train_command)


if __name__ == "__main__":

    DATASET_SIZES = [5, 10, 50, 100, 500, 1000]  # 5, 10
    IS_BAYESIAN = [True, False]
    IS_LEARNED = [True, False]
    SEEDS = list(range(5))

    FILENAME = "results_comparison.txt"

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

    for use_prior in IS_LEARNED:
        for is_sgld in IS_BAYESIAN:
            for ds_size in DATASET_SIZES:
                for seed in SEEDS:
                    run_experiment(is_sgld, use_prior, ds_size, seed, FILENAME)

                break
            break
        break
