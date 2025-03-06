import subprocess
import os
import random
import math

os.system("clear")
gpuid=int(subprocess.run(["python","scripts/select_gpu.py"],stdout=subprocess.PIPE).stdout.decode('utf-8').strip())

random.seed()
runtimes=32
# runtimes=1
vqvae_alpha=100
vqvae_beta=1
codebook=16
max_updates=400
load_from_rule_based_dataset=True
encoder_attention=False
learning_rate=2e-4
encoder_hidden_dim=32
encoder_attention_features_dim=4
algo="vqvae_modify"
project="vae_v1.2"
batch_size=512
vqvae_modify_use_sigma=True
vqvae_modify_sum_method="sum"

# env_name="MiniGrid-Reacher-MDP"
# env_name="MDPtakeball"
env_name="MiniGrid-Reacher-extra-good"
# env_name="halfcheetah"

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
    # vqvae_alpha=100
else:
    dataset="medium-expert"
    env_name=f"{env_name}-{dataset}-v2"
    load_from_rule_based_dataset=False
    max_updates=2000

def log_uniform(mi,mx):
    return mi*math.exp(random.random()*math.log(mx/mi))
def rand_hypers():
    global seed,codebook,vqvae_beta,vqvae_alpha,encoder_attention,encoder_hidden_dim
    global encoder_attention_features_dim,learning_rate,vqvae_modify_use_sigma, vqvae_modify_sum_method
    seed = random.randint(0,2**30-1)
    codebook = 2**random.randint(3,6)
    vqvae_beta = log_uniform(0.1,10)
    vqvae_alpha = log_uniform(1e-1,1e4)
    encoder_attention = random.choice([True,False])
    if encoder_attention:
        encoder_hidden_dim = random.choice([1,1,1,8,32])
        encoder_attention_features_dim = random.choice([1,1,1,4,16])
        if encoder_attention_features_dim > encoder_hidden_dim:
            encoder_attention_features_dim = 1
    if algo=="vqvae_modify":
        vqvae_modify_use_sigma = random.choice([True])
        vqvae_modify_sum_method = random.choice(["max","sum"])
    learning_rate = log_uniform(1e-5,1e-2)

for _ in range(runtimes):
    rand_hypers()
    command= f"CUDA_VISIBLE_DEVICES={gpuid} python algos/VAE_kmeans.py "
    command+=f"--env {env_name} "
    command+=f"--seed {seed} "
    command+=f"--project {project} "
    command+=f"--max_updates {max_updates} "
    command+=f"--vqvae_codebook {codebook} "
    command+=f"--vqvae_alpha {vqvae_alpha} "
    command+=f"--vqvae_beta {vqvae_beta} "
    command+=f"--algo {algo} "
    command+=f"--learning_rate {learning_rate} "
    if load_from_rule_based_dataset:
        command+=f"--load_from_rule_based_dataset True "
        command+=f"--rule_based_dataset_files {' '.join(rule_based_dataset_files)} "
    if encoder_attention:
        command+=f"--encoder_attention True "
        command+=f"--encoder_attention_features_dim {encoder_attention_features_dim} "
        command+=f"--encoder_hidden_dim {encoder_hidden_dim} "
    if algo=="vqvae_modify":
        if vqvae_modify_use_sigma:
            command+=f"--vqvae_modify_use_sigma True "
        command+=f"--vqvae_modify_sum_method {vqvae_modify_sum_method} "
    command+=f"--batch_size {batch_size} "
    print(command)
    os.system(command)