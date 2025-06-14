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
    "camera_heights": 512,
    "camera_widths": 512
}
env = OffScreenRenderEnv(**env_args)
env.seed(0)
env.reset()
init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
init_state_id = 0
env.set_init_state(init_states[init_state_id])


# Initialize list to store frames
frames = []
frames_2 = []


dummy_action = [0.1] * 7
for step in range(50):
    obs, reward, done, info = env.step(dummy_action)
    if "agentview_image" in obs:
        frames.append(obs["agentview_image"])
        print(f"Agentview image shape: {obs['agentview_image'].shape}")
    # if "image" in obs:
    #     frames.append(obs["image"])
    #     print(f"Image shape: {obs['image'].shape}")
    # else:
    #     frames.append(env.render())
    #     print(f"Render shape: {env.render().shape}")
    if 'robot0_eye_in_hand_image' in obs:
        frames_2.append(obs['robot0_eye_in_hand_image'])
        print(f"Robot0 eye in hand image shape: {obs['robot0_eye_in_hand_image'].shape}")
    print(obs)
env.close()

# Save video
video_path = os.path.join(output_dir, f"task_{task_id}_recording.mp4")
print(f"Saving video to: {video_path}")
video_path_2 = os.path.join(output_dir, f"task_{task_id}_recording_2.mp4")
print(f"Saving video to: {video_path_2}")

# Convert frames to uint8 if they're float
if frames[0].dtype == np.float32 or frames[0].dtype == np.float64:
    frames = [(frame * 255).astype(np.uint8) for frame in frames]
if frames_2[0].dtype == np.float32 or frames_2[0].dtype == np.float64:
    frames_2 = [(frame * 255).astype(np.uint8) for frame in frames_2]

# Save video with 10 FPS
imageio.mimsave(video_path, frames, fps=10)
imageio.mimsave(video_path_2, frames_2, fps=10)
# imageio.mimsave(video_path, frames, fps=30, quality=9)