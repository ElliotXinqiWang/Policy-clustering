# Offline Reinforcement Learning Trace Gathering

Use `python run.py <algo> <env>` to run our codes.

`<algo>` in: PG-Kmeans, VAE, DEV, CAAE. The PG-Kmeans don't include Best-of-5, you should run it 5 times and pick the run that has lowerest $J$.

`<env>` in: halfcheetah, ant, walker2d, hopper, diagonal, takeball, pathfollowing, extra.

`run.py` will first output the shell code, you can refer to it and make changes on seed or k value.