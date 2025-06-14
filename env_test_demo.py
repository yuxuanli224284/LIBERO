from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs import DemoRenderEnv
import os
from libero.libero.utils import get_libero_path
import imageio

benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_spatial" # can also choose libero_spatial, libero_object, etc.
task_suite = benchmark_dict[task_suite_name]()

# retrieve a specific task
task_id = 9
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
print("****************")
print(init_states[init_state_id])
# env.set_init_state(49.)

dummy_action = [0.1] * 7

# Video recording setup
video_filename = "env_render.mp4"
fps = 30  # Frames per second
writer = imageio.get_writer(video_filename, fps=fps)

for step in range(50):
    obs, reward, done, info = env.step(dummy_action)
    print(f"[info] step {step}, reward: {reward}, done: {done}, info: {info}")

    # Render the environment and save the frame
    # frame = env.render(mode="rgb_array")  # Render the environment as an RGB array
    frame = obs["agentview_image"]
    writer.append_data(frame)  # Add the frame to    the video

writer.close()  
env.close()


