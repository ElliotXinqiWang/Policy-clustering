import subprocess
import os
import random
import math

os.system("clear")
gpuid=int(subprocess.run(["python","scripts/select_gpu.py"],stdout=subprocess.PIPE).stdout.decode('utf-8').strip())
random.seed()

algo="vqvae_modify_few_sample"
# algo="vqvae"
# algo="DEC"
# project="vae_v1.24"
if algo=="DEC":
    origin_path="algos/DEC.py"
elif algo=="vqvae_modify_few_sample":
    origin_path="algos/VAE_few_sample.py"
elif algo=="vqvae_modify":
    origin_path="algos/VAE_kmeans.py"
else:
    raise Exception("Unknown algo")
copy_path=f"VAE_kmeans_runtimecopy{random.randint(0,2**30-1)}.py"

os.system(f"cp {origin_path} {copy_path}")

# runtimes=256
# runtimes=16
runtimes=1

vqvae_alpha=1
vqvae_beta=1
codebook=-1
max_updates=800
load_from_rule_based_dataset=True
encoder_attention=True
learning_rate=2e-3
encoder_hidden_dim=8
qk_dim=1
encoder_heads=2
project="vqvae_v1.2a"
batch_size=512
vqvae_modify_use_sigma=False
vqvae_modify_sum_method="sum"
encoder_attention_pre_process="rnn"
encoder_attention_pre_process_layers=1

lr_decay="none"
# lr_decay="warmup-cos"
lr_decay_v1=40
lr_decay_v2=360
lr_decay_v3=0.1

supervise_sample=16

# envs=['MiniGrid-Reacher-MDP','MDPtakeball','MiniGrid-Reacher-extra-good','halfcheetah']
envs=['MDPtakeball']

# env_name="MiniGrid-Reacher-MDP"
# env_name="MDPtakeball"
# env_name="MiniGrid-Reacher-extra-good"
# env_name="halfcheetah"

for env_name in envs:
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
        global seed,codebook,vqvae_beta,vqvae_alpha,encoder_attention,encoder_hidden_dim, algo
        global encoder_heads,learning_rate,vqvae_modify_use_sigma, vqvae_modify_sum_method, encoder_attention_pre_process, encoder_attention_pre_process_layers
        # algo=random.choice(["vqvae","vqvae_modify","vqvae_modify","vqvae_modify"])
        seed = random.randint(0,2**30-1)
        # codebook = 2**random.randint(3,6)
        # vqvae_beta = log_uniform(0.5,2)
        # vqvae_alpha = log_uniform(1e-1,1e3)
        # encoder_attention = random.choice([True,False])
        if encoder_attention:
            # encoder_hidden_dim = random.choice([1,1,1,8,32])
            # encoder_heads = random.choice([1,1,1,4,16])
            # encoder_hidden_dim, encoder_heads = random.choice([(1,1),(16,4)])
            # if encoder_heads > encoder_hidden_dim:
            #     encoder_heads = 1
            # encoder_attention_pre_process_layers=random.randint(0,3)
            pass
        if algo=="vqvae_modify":
            # vqvae_modify_use_sigma = random.choice([True,False])
            # vqvae_modify_sum_method = random.choice(["max","sum"])
            pass
        # learning_rate = log_uniform(1e-4,1e-2)
        # learning_rate = random.choice([5e-3,5e-4,5e-5])

    for _ in range(runtimes):
        rand_hypers()
        command= f"CUDA_VISIBLE_DEVICES={gpuid} "
        command+=f"python {copy_path} "
        command+=f"--env {env_name} "
        command+=f"--seed {seed} "
        command+=f"--project {project} "
        command+=f"--max_updates {max_updates} "
        if algo=="vqvae" or algo=="vqvae_modify" or algo=="vqvae_modify_few_sample":
            command+=f"--vqvae_codebook {codebook} "
            command+=f"--vqvae_alpha {vqvae_alpha} "
        if algo=="vqvae" or algo=="vqvae_modify":
            command+=f"--vqvae_beta {vqvae_beta} "
        command+=f"--algo {algo} "
        command+=f"--learning_rate {learning_rate} "
        if load_from_rule_based_dataset:
            command+=f"--load_from_rule_based_dataset True "
            command+=f"--rule_based_dataset_files {' '.join(rule_based_dataset_files)} "
        if encoder_attention:
            command+=f"--encoder_attention True "
            command+=f"--encoder_heads {encoder_heads} "
            command+=f"--encoder_hidden_dim {encoder_hidden_dim} "
        if algo=="vqvae_modify" or algo=="vqvae_modify_few_sample":
            if vqvae_modify_use_sigma:
                command+=f"--vqvae_modify_use_sigma True "
            command+=f"--encoder_attention_pre_process {encoder_attention_pre_process} "
            if encoder_attention_pre_process=="self_attention":
                command+=f"--encoder_attention_pre_process_layers {encoder_attention_pre_process_layers} "
        if algo=="vqvae_modify":
            command+=f"--vqvae_modify_sum_method {vqvae_modify_sum_method} "
            command+=f"--lr_decay {lr_decay} --lr_decay_v1 {lr_decay_v1} --lr_decay_v2 {lr_decay_v2} --lr_decay_v3 {lr_decay_v3} "
        if algo=="vqvae_modify_few_sample":
            command+=f"--supervise_sample {supervise_sample} "
        command+=f"--batch_size {batch_size} "
        print(command)
        os.system(command)

os.system(f"rm {copy_path}")
print("All done!")