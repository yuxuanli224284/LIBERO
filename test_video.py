from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils import get_libero_path
import os
import imageio
import numpy as np

# Create output directory for videos
output_dir = "output_videos"
os.makedirs(output_dir, exist_ok=True)

benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_10" # can also choose libero_spatial, libero_object, etc.
task_suite = benchmark_dict[task_suite_name]()

# retrieve a specific task
task_id = 0
task = task_suite.get_task(task_id)
task_name = task.name
task_description = task.language
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
      f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

# step over the environment
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 128,
    "camera_widths": 128
}
env = OffScreenRenderEnv(**env_args)
env.seed(0)
obs = env.reset()
init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
init_state_id = 0
obs = env.set_init_state(init_states[init_state_id])

# Initialize list to store frames
frames = []

# Capture initial frame
if "agentview_image" in obs:
    frames.append(obs["agentview_image"])
elif "image" in obs:
    frames.append(obs["image"])
else:
    frames.append(env.render())

dummy_action = [0.0] * 7
for step in range(10):
    obs, reward, done, info = env.step(dummy_action)
    
    # Capture frame after each step
    if "agentview_image" in obs:
        frames.append(obs["agentview_image"])
    elif "image" in obs:
        frames.append(obs["image"])
    else:
        frames.append(env.render())
    
    if done:
        break

# Save video
video_path = os.path.join(output_dir, f"task_{task_id}_recording.mp4")
print(f"Saving video to: {video_path}")

# Convert frames to uint8 if they're float
if frames[0].dtype == np.float32 or frames[0].dtype == np.float64:
    frames = [(frame * 255).astype(np.uint8) for frame in frames]

# Save video with 10 FPS
imageio.mimsave(video_path, frames, fps=10)

env.close()
print("Video saved successfully!") 