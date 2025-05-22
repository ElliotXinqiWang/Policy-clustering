# Offline Reinforcement Learning Trace Gathering

Use `python run.py <algo> <env>` to run our codes.

`<algo>` in: PG-Kmeans, VAE, DEV, CAAE. PG-Kmeans does not support Best-of-5 by default. You need to manually execute it 5 times(with different seeds, the default implementation mentioned below will help you) and choose the run yielding the minimal $J$."

`<env>` in: halfcheetah, ant, walker2d, hopper, diagonal, takeball, pathfollowing, extra.

`run.py` will first output the shell code, you can refer to it and make changes on seed or k value.