import d4rl
import gym
import matplotlib.pyplot as plt
import numpy as np

ANTMAZE_BOUNDS = {
    'antmaze-umaze-v2': (-3, 11),
    'antmaze-medium-play-v0': (-3, 23),
    'antmaze-medium-diverse-v0': (-3, 23),
    'antmaze-large-play-v0': (-3, 39),
    'antmaze-large-diverse-v0': (-3, 39),
    'antmaze-ultra-play-v0': (-3, 65),
    'antmaze-ultra-diverse-v0': (-3, 65),
}

def load_expert_trajectories(env_name):
    env = gym.make(env_name)
    dataset = env.get_dataset()
    observations = dataset['observations']
    terminals = dataset['terminals']
    timeouts = dataset['timeouts']
    trajectories = []
    current_trajectory = []

    for i in range(len(observations)):
        current_trajectory.append(observations[i])
        if terminals[i] or timeouts[i]:
            trajectories.append(np.array(current_trajectory))
            current_trajectory = []

    return trajectories

class AntMazeRenderer:

    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name).unwrapped
        self.observation_dim = np.prod(self.env.observation_space.shape)
        self.action_dim = np.prod(self.env.action_space.shape)

    def renders(self, savepath, X, nrow=None):
        plt.clf()

        if X.ndim < 3:
            X = X[None]

        N, path_length, _ = X.shape
        if N > 4:
            fig, axes = plt.subplots(4, int(N / 4))
            axes = axes.flatten()
            fig.set_size_inches(N / 4, 8)
        elif N > 1:
            fig, axes = plt.subplots(1, N)
            fig.set_size_inches(8, 8)
        else:
            fig, axes = plt.subplots(1, 1)
            fig.set_size_inches(8, 8)

        colors = plt.cm.jet(np.linspace(0, 1, path_length))
        for i in range(N):
            ax = axes if N == 1 else axes[i]
            xlim, ylim = self.plot_boundaries(ax=ax)
            x = X[i]
            ax.scatter(x[:, 0], x[:, 1], c=colors)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        plt.savefig(savepath + '.png')
        plt.close()
        print(f'[ attentive/utils/visualization ] Saved to: {savepath}')

    def plot_boundaries(self, N=100, ax=None):
        """
        Plots the maze boundaries in the antmaze environments.
        """
        ax = ax or plt.gca()

        xlim = ANTMAZE_BOUNDS[self.env_name]
        ylim = ANTMAZE_BOUNDS[self.env_name]

        X = np.linspace(*xlim, N)
        Y = np.linspace(*ylim, N)

        Z = np.zeros((N, N))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                collision = self.env.unwrapped._is_in_collision((x, y))
                Z[-j, i] = collision

        ax.imshow(Z, extent=(*xlim, *ylim), aspect='auto', cmap=plt.cm.binary)
        return xlim, ylim

    def render_rollout(self, savepath, states, **video_kwargs):
        if type(states) is list:
            states = np.stack(states, axis=0)[None]
        images = self.renders(savepath, states)

# Initialize the renderer and load expert trajectories
offline_dataset_name = 'antmaze-large-play-v0'
renderer = AntMazeRenderer(offline_dataset_name)
expert_trajectories = load_expert_trajectories(offline_dataset_name)

# print(expert_trajectories[0])

# Render the environment
renderer.env.render('rgb_array')
viewer = renderer.env.viewer
viewer.cam.trackbodyid = -1
viewer.cam.distance = renderer.env.model.stat.extent * 1.0        # how much you "zoom in", model.stat.extent is the max limits of the arena
viewer.cam.lookat[0] += 6       # x,y,z offset from the object (works if trackbodyid=-1)
viewer.cam.lookat[1] += 8
viewer.cam.lookat[2] += 0
viewer.cam.elevation = -90           # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
viewer.cam.azimuth = 0              # camera rotation around the camera's vertical axis

# Plot the expert trajectories
image = np.fliplr(np.rot90((renderer.env.physics.render(1024, 1024)), k=-1))
fig, ax = plt.subplots()
fig.set_size_inches(8, 4)

xlim = ANTMAZE_BOUNDS[offline_dataset_name]
ylim = ANTMAZE_BOUNDS[offline_dataset_name]
ax.imshow(image, extent=(*xlim, *ylim), aspect='auto')

# for n in range(0, min(10, len(expert_trajectories))):
#     print("Trajectory:")
#     # print(expert_trajectories[n])
#     states = np.array([t[:2] for t in expert_trajectories[n]])
#     print("States")
#     print(states)
#     print(states[:, 0])
#     print(states[:, 1])
#     # ax.scatter(states[:, 0], states[:, 1], color=plt.cm.Purples(0), alpha=1, s=1)
#     ax.plot(states[:, 0], states[:, 1], color=plt.cm.Purples(0), alpha=1)

for n in range(0, 50):
    # print(len(expert_trajectories[n]))
    # print(len(expert_trajectories[n][0]))
    # break
    states = np.array([t[:2] for t in expert_trajectories[n]])
    # print("States")
    # print(states)
    # print(states[:, 0])
    # print(states[:, 1])
    # ax.scatter(states[:, 0], states[:, 1], color=plt.cm.Purples(0), alpha=1, s=1)
    ax.plot(states[:, 0], states[:, 1], color=plt.cm.Purples(0), alpha=1)
    # if n==2:
    # break


goal = (20, 20)
ax.scatter(goal[0], goal[1], marker='*', color='white', edgecolor='black', s=500, zorder=100)
plt.tight_layout()
fig.savefig('antmaze_reward_visualization_scatter.png')
