import subprocess

subprocess.run("clear", shell=True)
print("Cleared the screen.")
# env_names = ["Gridworld-reacher-continous-lu"]
env_names = ["Gridworld-reacher-continous-lu","Gridworld-reacher-continous-dr","Gridworld-reacher-continous"]
select_gpu_script = "scripts/select_gpu.py"
selected_gpu = subprocess.check_output(["python", select_gpu_script]).decode("utf-8").strip()
print(f"Selected GPU: {selected_gpu}")
max_updates = "20000"
train=False
collect=True

for env_name in env_names:
	print(f"Running PPO in env {env_name}")
	command = [
		"CUDA_VISIBLE_DEVICES=" + selected_gpu, 
		"python", "algos/PPO/PPO_train.py", 
		"--env", env_name, 
		"--seed", "2", 
		"--max_updates", max_updates,
		# "--gamma", "0.9"
	]
	if train:
		print(" ".join(command))
		subprocess.run(" ".join(command), shell=True)

	model_path = f"behavior_models/PPO_{env_name}/best_params.pkl"
	if collect:
		print(f"Collecting expert data in env {env_name}")
		command = [
			"CUDA_VISIBLE_DEVICES=" + selected_gpu,
			"python", "algos/PPO/Collect.py",
			"--env", env_name,
			"--seed", "2",
			"--n_episodes", "1",
			"--num_steps", "20000",
			"--model_load_path", model_path,
			# "--debug", "True",
		]
		print(" ".join(command))
		subprocess.run(" ".join(command), shell=True)