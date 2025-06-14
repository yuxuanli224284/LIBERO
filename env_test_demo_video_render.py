from IPython.display import HTML
from base64 import b64encode
import imageio

from libero.libero.envs import OffScreenRenderEnv, DummyVectorEnv
from libero.lifelong.metric import raw_obs_to_tensor_obs

# You can turn on subprocess
env_num = 1
action_dim = 7


# If it's packnet, the weights need to be processed first
task_id = 9
task = benchmark.get_task(task_id)
task_emb = benchmark.get_task_emb(task_id)

if cfg.lifelong.algo == "PackNet":
    algo = algo.get_eval_algo(task_id)

algo.eval()
env_args = {
    "bddl_file_name": os.path.join(
        cfg.bddl_folder, task.problem_folder, task.bddl_file
    ),
    "camera_heights": cfg.data.img_h,
    "camera_widths": cfg.data.img_w,
}

env = DummyVectorEnv(
            [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
)

init_states_path = os.path.join(
    cfg.init_states_folder, task.problem_folder, task.init_states_file
)
init_states = torch.load(init_states_path)

env.reset()

init_state = init_states[0:1]
dones = [False]

algo.reset()

obs = env.set_init_state(init_state)

# Make sure the gripepr is open to make it consistent with the provided demos.
dummy_actions = np.zeros((env_num, action_dim))
for _ in range(5):
    obs, _, _, _ = env.step(dummy_actions)

steps = 0

obs_tensors = [[]] * env_num
while steps < cfg.eval.max_steps:
    steps += 1
    data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
    action = algo.policy.get_action(data)

    obs, reward, done, info = env.step(action)

    for k in range(env_num):
        dones[k] = dones[k] or done[k]
        obs_tensors[k].append(obs[k]["agentview_image"])
    if all(dones):
        break

# visualize video
# obs_tensor: (env_num, T, H, W, C)

images = [img[::-1] for img in obs_tensors[0]]
fps = 30
writer  = imageio.get_writer('tmp_video.mp4', fps=fps)
for image in images:
    writer.append_data(image)
writer.close()

video_data = open("tmp_video.mp4", "rb").read()
video_tag = f'<video controls alt="test" src="data:video/mp4;base64,{b64encode(video_data).decode()}">'
HTML(data=video_tag)
