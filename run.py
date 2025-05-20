import sys
import os

algo=sys.argv[1]
env=sys.argv[2]
envmap={
	"diagonal":"MiniGrid-Reacher-MDP",
	"takeball":"MDPtakeball",
	"extra":"MiniGrid-Reacher-extra-good",
	"halfcheetah":"halfcheetah",
	"ant":"ant",
	"walker2d":"walker2d",
	"hopper":"hopper",
	"pathfollowing":"Gridworld-reacher-continous",
}
env=env.lower()
print(algo,env)

if algo in ["SORL","CAAE","DEV","VAE"]:
	command=f"python scripts/run.py {algo} {envmap[env]}"
elif algo == "PG-Kmeans":
	if env in ['halfcheetah','ant','walker2d','hopper']:
		command=f"sh scripts/Kmeans_D4RL.sh {env}"
	elif env=="extra":
		command=f"sh scripts/Kmeans_gridworld_extra.sh"
	elif env=="diagonal":
		command=f"sh scripts/Kmeans_gridworld_diagonal.sh"
	elif env=="takeball":
		command=f"sh scripts/Kmeans_gridworld_takeball.sh"
	elif env=="pathfollowing":
		command=f"sh scripts/Kmeans_gridworld_diagonal_continous.sh"
	else:
		raise Exception("Unknown env")
else:
	raise Exception("Unknown algo")
print(command)
os.system(command)