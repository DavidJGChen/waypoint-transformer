import d4rl
import gym
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
# Load model trajectories
# model_trajectories = np.load('wt_waypoints_antmaze-large-diverse-v2_0.7.npy', allow_pickle=True)
model_trajectories = np.load('wt_trajectories_antmaze-large-diverse-v2_0.7.npy', allow_pickle=True)
# model_trajectories = np.load('model_trajectories.npy', allow_pickle=True)

ANTMAZE_BOUNDS = {
    'antmaze-umaze-v2': (-3, 11),
    'antmaze-medium-play-v0': (-3, 23),
    'antmaze-medium-diverse-v0': (-3, 23),
    'antmaze-large-play-v2': (-3, 39),
    'antmaze-large-diverse-v0': (-3, 39),
    'antmaze-ultra-play-v2': (-3, 65),
    'antmaze-ultra-diverse-v0': (-3, 65),
}

def load_expert_trajectories(env_name):
    env = gym.make(env_name)
    dataset = env.get_dataset()
    observations = dataset['observations']
    terminals = dataset['terminals']
    timeouts = dataset['timeouts']
    trajectories = []
    reached_goal = []
    current_trajectory = []
    did_reach_goal = False

    for i in range(len(observations)):
        current_trajectory.append(observations[i])
        if dataset['rewards'][i] > 0:
            did_reach_goal = True
        if timeouts[i]:
            trajectories.append(np.array(current_trajectory))
            reached_goal.append(did_reach_goal)
            current_trajectory = []
            did_reach_goal = False

    return trajectories, reached_goal

def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments

def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
           
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
        
    z = np.asarray(z)
    
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    
    ax = plt.gca()
    ax.add_collection(lc)
    
    return lc

class AntMazeRenderer:

    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name).unwrapped
        self.observation_dim = np.prod(self.env.observation_space.shape)
        self.action_dim = np.prod(self.env.action_space.shape)


# Initialize the renderer and set the environment
offline_dataset_name = 'antmaze-large-play-v2'
renderer = AntMazeRenderer(offline_dataset_name)
# expert_trajectories, reached_goal = load_expert_trajectories(offline_dataset_name)

# Render the environment
renderer.env.render('rgb_array')
viewer = renderer.env.viewer
viewer.cam.trackbodyid = -1
viewer.cam.distance = renderer.env.model.stat.extent * 1.025
viewer.cam.lookat[0] += 6
viewer.cam.lookat[1] += 4
viewer.cam.lookat[2] += 0
viewer.cam.elevation = -90
viewer.cam.azimuth = 0

# Plot the model trajectories
image = np.fliplr(np.rot90((renderer.env.physics.render(1000, 1400)), k=-1))
fig, ax = plt.subplots()
# fig.set_size_inches(4, 4)

# xlim = ANTMAZE_BOUNDS[offline_dataset_name]
# ylim = ANTMAZE_BOUNDS[offline_dataset_name]
ax.imshow(image, extent=(-3, 39, -3, 27), aspect='equal')

# print(expert_trajectories[0])

# for n in range(0, len(expert_trajectories)):
#     states = np.array([t[:2] for t in expert_trajectories[n]])
#     if reached_goal[n]:
#         colorline(states[:, 0], states[:, 1], 'm', alpha=0.95, linewidth=0.5)
#     else:
#         colorline(states[:, 0], states[:, 1], cmap=plt.get_cmap('autumn'), alpha=0.95, linewidth=0.2)

print(len(model_trajectories))

wp_num = 4

for ep in model_trajectories[wp_num: wp_num + 1]:
    states = np.array([t[:2] for t in ep['observations']])
    colorline(states[:, 0], states[:, 1], cmap=plt.get_cmap('autumn'), alpha=0.95, linewidth=0.5)
    # for i in range(states.shape[0]):
    print(ep['observations'][0])
    print(ep['observations'][-1])
    # ax.scatter(states[-1, 0], states[-1, 1], alpha=1)

# for ep in model_trajectories[wp_num: wp_num + 1]:
#     states = ep
#     colorline(states[:, 0], states[:, 1], cmap=plt.get_cmap('cool'), alpha=0.95, linewidth=0.5)
#     ax.scatter(states[-1, 0], states[-1, 1], alpha=1)

goal = (33, 24.5)
ax.scatter(goal[0], goal[1], marker='*', color='white', edgecolor='black', s=500, zorder=100)
plt.tight_layout()
plt.axis('off')
# fig.savefig(f'expert-{offline_dataset_name}-visualization_trajectories.png', bbox_inches='tight', dpi=300)
# fig.savefig(f'{offline_dataset_name}-visualization_endpoints.png', bbox_inches='tight', dpi=300)
# fig.savefig(f'expert-{offline_dataset_name}-visualization_waypoints_{wp_num}.png', bbox_inches='tight', dpi=300)
fig.savefig(f'{offline_dataset_name}-visualization_trajectories_{wp_num}.png', bbox_inches='tight', dpi=300)
