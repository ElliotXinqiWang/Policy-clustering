import subprocess
import os
import sys
import random
import math

os.system("clear")
gpuid=int(subprocess.run(["python","scripts/select_gpu.py"],stdout=subprocess.PIPE).stdout.decode('utf-8').strip())
random.seed()

# print(sys.argv)
# exit(0)
algo=sys.argv[1]
# algo="CAAE"
# algo="CAAE_few_sample"
# algo="CAAE_self_train"
# algo="vqvae"
# algo="vqvae"
# algo="vae"
# algo="DEC"
# algo="SORL"
# project="vae_v1.24"
if algo=="DEC":
    origin_path="algos/DEC.py"
elif algo=="CAAE_few_sample" or algo=="CAAE_self_train":
    origin_path="algos/VAE_few_sample.py"
elif algo=="CAAE" or algo=="vqvae" or algo=='vae':
    origin_path="algos/VAE_kmeans.py"
elif algo=="SORL":
    origin_path="algos/SORL.py"
else:
    raise Exception("Unknown algo")
copy_path=f"VAE_kmeans_runtimecopy{random.randint(0,2**30-1)}.py"

os.system(f"cp {origin_path} {copy_path}")

# runtimes=256
# runtimes=5
runtimes=1

vqvae_alpha=1
vqvae_beta=1
vae_kl_weight=0
# vae_kl_weight=1
codebook=-1
# max_updates=10
# max_updates=40
max_updates=200
# max_updates=500
# max_updates=1000
load_from_rule_based_dataset=True
encoder_attention=True
learning_rate=2e-3
encoder_hidden_dim=8
qk_dim=1
encoder_heads=2
project="test_k"
# project="LORA_v1"
batch_size=512
CAAE_use_sigma=False
CAAE_sum_method="max"
encoder_attention_pre_process="rnn"
encoder_attention_pre_process_layers=1
true_k_available=True

lr_decay="none"
# lr_decay="warmup-cos"
lr_decay_v1=40
lr_decay_v2=360
lr_decay_v3=0.1

supervise_sample=100
# supervise_samples=[221]
# while supervise_samples[-1] < 400:
#     supervise_samples.append(int(supervise_samples[-1]*1.1))
# supervise_samples=supervise_samples[1:]
# print(supervise_samples)
# exit(0)

# envs=['MiniGrid-Reacher-MDP','MDPtakeball','MiniGrid-Reacher-extra-good','halfcheetah','Gridworld-reacher-continous']
# envs=['MiniGrid-Reacher-MDP','halfcheetah','ant','walker2d','hopper','MDPtakeball','MiniGrid-Reacher-extra-good','Gridworld-reacher-continous']
# envs=['halfcheetah']
# envs=['walker2d']
# envs=['MDPtakeball-hard']
# envs=['MDPtakeball']
# envs=['MiniGrid-Reacher-extra-good']
# envs=['MiniGrid-Reacher-MDP']
# envs=['Gridworld-reacher-continous']
envs=[sys.argv[2]]

# env_name="MiniGrid-Reacher-MDP"
# env_name="MDPtakeball"
# env_name="MiniGrid-Reacher-extra-good"
# env_name="halfcheetah"

if algo=="SORL":
    encoder_attention=False
    batch_size=1024
    k_value=6
true_k_available=True

for env_name in envs:
#  k_start=4
#  if env_name == "MiniGrid-Reacher-MDP": k_start=5
#  for k_value in range(k_start, k_start+5):
    load_from_rule_based_dataset=True
    if env_name == "MiniGrid-Reacher-MDP":
        rule_based_dataset_files=[
            "datasets/rule_based/MiniGrid-Reacher-MDP/balanced_20000.pkl",
            "datasets/rule_based/MiniGrid-Reacher-MDP/rightfirst_20000.pkl",
            "datasets/rule_based/MiniGrid-Reacher-MDP/downfirst_20000.pkl",
            "datasets/rule_based/MiniGrid-Reacher-MDP/zigzag1_20000.pkl",
            "datasets/rule_based/MiniGrid-Reacher-MDP/zigzag2_20000.pkl",
            # "datasets/rule_based/MiniGrid-Reacher-MDP/random1_20000.pkl",
            # "datasets/rule_based/MiniGrid-Reacher-MDP/random2_20000.pkl"
        ]
    elif env_name == "MDPtakeball":
        rule_based_dataset_files=[
            "datasets/rule_based/MDPtakeball/fixed_0_20000.pkl",
            "datasets/rule_based/MDPtakeball/fixed_1_20000.pkl",
            "datasets/rule_based/MDPtakeball/fixed_2_20000.pkl",
            "datasets/rule_based/MDPtakeball/fixed_3_20000.pkl",
        ]
    elif env_name == "MDPtakeball-hard":
        rule_based_dataset_files=[
            "datasets/rule_based/MDPtakeball-hard/fixed_0_20000.pkl",
            "datasets/rule_based/MDPtakeball-hard/fixed_1_20000.pkl",
            "datasets/rule_based/MDPtakeball-hard/fixed_2_20000.pkl",
            "datasets/rule_based/MDPtakeball-hard/fixed_3_20000.pkl",
        ]
    elif env_name == "MiniGrid-Reacher-extra-good":
        rule_based_dataset_files=[
            "datasets/rule_based/MiniGrid-Reacher-extra-good/batch_8000.pkl",
            "datasets/rule_based/MiniGrid-Reacher-extra-bad/batch_20000.pkl",
            "datasets/rule_based/MiniGrid-Reacher-extra-med/batch_20000.pkl",
        ]
        # vqvae_alpha=100
    elif env_name == "Gridworld-reacher-continous":
        rule_based_dataset_files=[
            "datasets/Gridworld-reacher-continous/continous/data_20000_0.pkl",
            "datasets/Gridworld-reacher-continous-dr/dr/data_20000_0.pkl",
            "datasets/Gridworld-reacher-continous-lu/lu/data_20000_0.pkl",
        ]
        max_updates=500
    else:
        dataset="medium-expert"
        # dataset="medium-replay"
        if env_name[-2:]!="v2":
            env_name=f"{env_name}-{dataset}-v2"
        load_from_rule_based_dataset=False
        # max_updates=2000

    def log_uniform(mi,mx):
        return mi*math.exp(random.random()*math.log(mx/mi))
    def rand_hypers():
        global seed,codebook,vqvae_beta,vqvae_alpha,encoder_attention,encoder_hidden_dim, algo
        global encoder_heads,learning_rate,CAAE_use_sigma, CAAE_sum_method, encoder_attention_pre_process, encoder_attention_pre_process_layers
        # algo=random.choice(["vqvae","CAAE","CAAE","CAAE"])
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
        if algo=="CAAE":
            # CAAE_use_sigma = random.choice([True,False])
            # CAAE_sum_method = random.choice(["max","sum"])
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
        if algo=="vqvae" or algo=="CAAE" or algo=="CAAE_few_sample" or algo=="CAAE_self_train":
            command+=f"--vqvae_codebook {codebook} "
            command+=f"--vqvae_alpha {vqvae_alpha} "
        if algo=="vqvae" or algo=="CAAE":
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
        if algo=="CAAE" or algo=="CAAE_few_sample" or algo=="CAAE_self_train":
            if CAAE_use_sigma:
                command+=f"--CAAE_use_sigma True "
            command+=f"--encoder_attention_pre_process {encoder_attention_pre_process} "
            if encoder_attention_pre_process=="self_attention":
                command+=f"--encoder_attention_pre_process_layers {encoder_attention_pre_process_layers} "
        if algo=="CAAE":
            command+=f"--CAAE_sum_method {CAAE_sum_method} "
            command+=f"--lr_decay {lr_decay} --lr_decay_v1 {lr_decay_v1} --lr_decay_v2 {lr_decay_v2} --lr_decay_v3 {lr_decay_v3} "
            command+=f"--vae_kl_weight {vae_kl_weight} "
        if algo=="CAAE_few_sample" or algo=="CAAE_self_train":
            command+=f"--supervise_sample {supervise_sample} "
        if true_k_available:
            command+=f"--true_k_available True "
        else:
            command+=f"--k_value {k_value} "
        command+=f"--batch_size {batch_size} "
        print(command)
        os.system(command)

os.system(f"rm {copy_path}")
print("All done!")