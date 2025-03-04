import subprocess
import os

gpuid=int(subprocess.run(["python","scripts/select_gpu.py"],stdout=subprocess.PIPE).stdout.decode('utf-8').strip())

vqvae_alpha=100
vqvae_beta=1
seeds=range(3,6)
codebook=16

env_name="MiniGrid-Reacher-MDP"
# env_name="MDPtakeball"
# env_name="MiniGrid-Reacher-extra-good"
if env_name == "MiniGrid-Reacher-MDP":
    rule_based_dataset_files=[
        "datasets/rule_based/MiniGrid-Reacher-MDP/balanced_20000.pkl",
        "datasets/rule_based/MiniGrid-Reacher-MDP/rightfirst_20000.pkl",
        "datasets/rule_based/MiniGrid-Reacher-MDP/downfirst_20000.pkl",
        "datasets/rule_based/MiniGrid-Reacher-MDP/zigzag1_20000.pkl",
        "datasets/rule_based/MiniGrid-Reacher-MDP/zigzag2_20000.pkl"
    ]
elif env_name == "MDPtakeball":
    rule_based_dataset_files=[
        "datasets/rule_based/MDPtakeball/fixed_0_20000.pkl",
        "datasets/rule_based/MDPtakeball/fixed_1_20000.pkl",
        "datasets/rule_based/MDPtakeball/fixed_2_20000.pkl",
        "datasets/rule_based/MDPtakeball/fixed_3_20000.pkl",
    ]
elif env_name == "MiniGrid-Reacher-extra-good":
    rule_based_dataset_files=[
        "datasets/rule_based/MiniGrid-Reacher-extra-good/batch_8000.pkl",
        "datasets/rule_based/MiniGrid-Reacher-extra-bad/batch_20000.pkl",
        "datasets/rule_based/MiniGrid-Reacher-extra-med/batch_20000.pkl",
    ]
    # vqvae_alpha=10

for seed in seeds:
    command= f"CUDA_VISIBLE_DEVICES={gpuid} python algos/VAE_kmeans.py "
    command+=f"--env {env_name} "
    command+=f"--seed {seed} "
    command+=f"--project VQVAE_D4RL "
    command+=f"--max_updates 1000 "
    command+=f"--rule_based_dataset_files {' '.join(rule_based_dataset_files)} "
    command+=f"--vqvae_codebook {codebook} "
    command+=f"--vqvae_alpha {vqvae_alpha} "
    command+=f"--vqvae_beta {vqvae_beta} "
    command+=f"--algo vqvae "
    command+=f"--learning_rate {2e-4} "
    command+=f"--load_from_rule_based_dataset true "
    print(command)
    os.system(command)