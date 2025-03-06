import subprocess
import os
import random
import math

os.system("clear")
gpuid=int(subprocess.run(["python","scripts/select_gpu.py"],stdout=subprocess.PIPE).stdout.decode('utf-8').strip())

random.seed()
runtimes=16
vqvae_alpha=100
vqvae_beta=1
codebook=16
max_updates=400
load_from_rule_based_dataset=True
encoder_attention=True
learning_rate=2e-4
encoder_hidden_dim=32
encoder_attention_features_dim=4
algo="vqvae"
batch_size=512

# env_name="MiniGrid-Reacher-MDP"
# env_name="MDPtakeball"
# env_name="MiniGrid-Reacher-extra-good"
env_name="halfcheetah"

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
    global seed,codebook,vqvae_beta,vqvae_alpha,encoder_attention,encoder_hidden_dim,encoder_attention_features_dim,learning_rate
    seed = random.randint(0,2**30-1)
    codebook = 2**random.randint(3,6)
    vqvae_beta = log_uniform(0.1,10)
    vqvae_alpha = log_uniform(1e-1,1e4)
    encoder_attention = random.choice([True,True,False])
    if encoder_attention:
        encoder_hidden_dim = random.choice([1,8,32])
        encoder_attention_features_dim = random.choice([1,4,16])
        if encoder_attention_features_dim > encoder_hidden_dim:
            encoder_attention_features_dim = 1
    learning_rate = log_uniform(1e-5,1e-2)

for _ in range(runtimes):
    rand_hypers()
    command= f"CUDA_VISIBLE_DEVICES={gpuid} python algos/VAE_kmeans.py "
    command+=f"--env {env_name} "
    command+=f"--seed {seed} "
    command+=f"--project {algo} "
    command+=f"--max_updates {max_updates} "
    command+=f"--vqvae_codebook {codebook} "
    command+=f"--vqvae_alpha {vqvae_alpha} "
    command+=f"--vqvae_beta {vqvae_beta} "
    command+=f"--algo {algo} "
    command+=f"--learning_rate {learning_rate} "
    if load_from_rule_based_dataset:
        command+=f"--load_from_rule_based_dataset True "
        command+=f"--rule_based_dataset_files {' '.join(rule_based_dataset_files)} "
    command+=f"--encoder_attention {encoder_attention} "
    command+=f"--encoder_attention_features_dim {encoder_attention_features_dim} "
    command+=f"--encoder_hidden_dim {encoder_hidden_dim} "
    command+=f"--batch_size {batch_size} "
    print(command)
    os.system(command)