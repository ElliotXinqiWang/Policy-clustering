# Policy-clustering

Environments and Datasets:
- Gridworld: Reacher/Takeball/Diagonal
- D4RL medium-expert datasets

Algorithms:
- PG-Kmeans
- DEC-based methods
- VAE+Kmeans
- GMM-based clustering in SORL(Not-implemented)
- Dataset clustering for improved offline policy learning (Not-implemented)
- Diffusion policies as an expressive policy class for offline reinforcement learning (Not-implemented)
- Offline reinforcement learning with closed-formpolicy improvement operators (Not-implemented)

# PG-Kmeans

To collected data for Minigrad environments. Run 
bash scripts/rule_based_collect.sh
you can change the collect settings in rule_based_collect.sh

After collecting the data

To run PG-Kmeans on D4RL datasets, Run
bash scripts/Kmeans_D4RL
To run PG-Kmeans on takeball, Run
bash scripts/Kmeans_gridworld_takeball.sh
To run PG-Kmeans on diagonal, Run
bash scripts/Kmeans_gridworld_diagonal.sh

To run VAE+Kmeans on D4RL dataset, Run
bash scripts/VAE_D4RL.sh 
To run VAE+Kmeans takeball+diagonal, Run
bash scripts/VAE_gridworld_MDP

To run DEC on any environment, change the env_name in the script and Run
bash scripts/DEC_all.sh
