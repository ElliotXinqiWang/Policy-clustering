# Offline Reinforcement Learning Trace Gathering

## Run code

Use `python run.py <algo> <env>` to run our codes.

`<algo>` options: PG-Kmeans, VAE, DEV, CAAE.

PG-Kmeans does not support Best-of-5 by default. You need to manually execute it 5 times(with different seeds, the default implementation mentioned below will help you) and choose the run yielding the minimal $J$."

`<env>` in: halfcheetah, ant, walker2d, hopper, diagonal, takeball, pathfollowing, extra.

`run.py` will first output the shell code, you can refer to it and make changes on seed or k value.

## Dataset collection

Use PG-Kmeans as `<algo>` to run `run.py` will automatic download dataset of D4RL env.

Use `sh scripts/rule_based_collect.sh` to collect dataset of diagonal env.

Use `python scripts/PPO_collect.py` to collect dataset of pathfollowing env.

Other dataset are attached in the support material.