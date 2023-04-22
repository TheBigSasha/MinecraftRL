import numpy as np
import torch
from torch import optim
from tqdm import tqdm
import minedojo
import torch.nn as nn
from helpers.hunt_cow import HuntCowDenseRewardEnv
import csv

# For m1 mac - MPS backend
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
REDUCED_ACTION_SPACE = True

# IMAGE_SIZE = (142,255)
IMAGE_SIZE = (160, 256)
# env = minedojo.make("harvest_wool_with_shears_and_sheep", image_size=(160, 256) )
env = HuntCowDenseRewardEnv(
    step_penalty=0.1,
    nav_reward_scale=2,
    attack_reward=10,
    success_reward=100,
    image_size=IMAGE_SIZE,
)
act_space = env.action_space
obs_space = env.observation_space


def load_item_mapping_from_csv(file_path):
    item_mapping = {}
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row if it exists
        for row in reader:
            item_name, _, item_id, item_id_2 = row  # Change this line to read the first and third columns
            if item_id is None or item_id == '':
                item_id = 10000 + len(item_mapping)
            item_id = int(item_id)
            if item_id_2 is not None:
                item_id = item_id + 10000 * int(item_id_2)

            item_mapping[item_name] = item_id
            item_mapping[item_name.replace(" ", "_")] = item_id
            item_mapping[item_name.replace(" ", "_").lower()] = item_id
            item_mapping[item_name.replace(" ", "_").upper()] = item_id
            item_mapping[item_name.replace(" ", "_").capitalize()] = item_id
            item_mapping[item_name.replace(" ", "_").lower().capitalize()] = item_id
            item_mapping[item_name.replace(" ", "_").upper().capitalize()] = item_id
            item_mapping[item_id] = item_id
    return item_mapping


#
# def minimizeIds(item_mapping):
#     # minimizing IDs by replacing each items id with its index in the list
#     # this is done to reduce the size of the observation space
#     minimized = {}
#     for key, value in item_mapping.items():
#         if key not in minimized:
#             minimized[key] = len(minimized)
#     return minimized

# ensure all numeric ids are unique
# def check_item_mapping(item_mapping):
#     numeric_ids = [item_id for item_id in item_mapping.values() if isinstance(item_id, int)]
#     assert len(numeric_ids) == len(set(numeric_ids))

string_to_index_mapping = load_item_mapping_from_csv("minecraft_items.csv")


def preprocess_observation(obs_dict, space_dict=obs_space, string_to_index_mapping=string_to_index_mapping):
    rgb = obs_dict['rgb']
    equipment = obs_dict['equipment']["name"]
    equipment_count = obs_dict['equipment']["quantity"]
    inventory = obs_dict['inventory']['name']
    inventory_count = obs_dict['inventory']['quantity']
    block_names = obs_dict["voxels"]["block_name"]
    block_collidable = obs_dict["voxels"]["is_collidable"]
    tool_req = obs_dict["voxels"]["is_tool_not_required"]
    is_liquid = obs_dict["voxels"]["is_liquid"]
    is_solid = obs_dict["voxels"]["is_solid"]
    look_vector = obs_dict["voxels"]["cos_look_vec_angle"]
    cur_durability = obs_dict["equipment"]["cur_durability"]
    max_durability = obs_dict["equipment"]["max_durability"]
    inc_name_by_craft = obs_dict["delta_inv"]["inc_name_by_craft"]
    inc_quantity_by_craft = obs_dict["delta_inv"]["inc_quantity_by_craft"]
    inc_name_by_other = obs_dict["delta_inv"]["inc_name_by_other"]
    inc_quantity_by_other = obs_dict["delta_inv"]["inc_quantity_by_other"]
    dec_name_by_craft = obs_dict["delta_inv"]["dec_name_by_craft"]
    dec_quantity_by_craft = obs_dict["delta_inv"]["dec_quantity_by_craft"]
    dec_name_by_other = obs_dict["delta_inv"]["dec_name_by_other"]
    dec_quantity_by_other = obs_dict["delta_inv"]["dec_quantity_by_other"]
    life = obs_dict["life_stats"]["life"]
    oxygen = obs_dict["life_stats"]["oxygen"]
    armor = obs_dict["life_stats"]["armor"]
    food = obs_dict["life_stats"]["food"]
    saturation = obs_dict["life_stats"]["saturation"]
    is_sleeping = obs_dict["life_stats"]["is_sleeping"]
    xp = obs_dict["life_stats"]["xp"]
    pos = obs_dict["location_stats"]["pos"]
    damage_amount = obs_dict["damage_source"]["damage_amount"]

    # give us a 1d array of all this stuff smooshed into ints
    obs_list = []

    rgb_flattened = np.zeros((len(rgb[0]), len(rgb[0][0])), dtype=np.uint32)
    for channel in rgb:
        for idx, row in enumerate(channel):
            for idx2, item in enumerate(row):
                rgb_flattened[idx][idx2] = rgb_flattened[idx][idx2] << 8
                rgb_flattened[idx][idx2] = rgb_flattened[idx][idx2] | item

    for row in rgb_flattened:
        for item in row:
            obs_list.append(item)
    for item in equipment:
        obs_list.append(string_to_index_mapping[item])
    for item in equipment_count:
        obs_list.append(item)
    for item in inventory:
        obs_list.append(string_to_index_mapping[item])
    for item in inventory_count:
        obs_list.append(item)
    for list in block_names:
        for sub_list in list:
            for item in sub_list:
                obs_list.append(string_to_index_mapping[item])
    block_collidable_int = 0
    for item in block_collidable:
        # 2 nested lists inside this, but we want to flatten them each to ints using bit shifting
        for sub_list in item:
            for sub_item in sub_list:
                block_collidable_int = block_collidable_int << 1
                block_collidable_int = block_collidable_int | sub_item
    obs_list.append(block_collidable_int)
    tool_req_int = 0
    for item in tool_req:
        # 2 nested lists inside this, but we want to flatten them each to ints using bit shifting
        for sub_list in item:
            for sub_item in sub_list:
                tool_req_int = tool_req_int << 1
                tool_req_int = tool_req_int | sub_item
    obs_list.append(tool_req_int)
    is_liquid_int = 0
    for item in is_liquid:
        # 2 nested lists inside this, but we want to flatten them each to ints using bit shifting
        for sub_list in item:
            for sub_item in sub_list:
                is_liquid_int = is_liquid_int << 1
                is_liquid_int = is_liquid_int | sub_item
    obs_list.append(is_liquid_int)
    is_solid_int = 0
    for item in is_solid:
        # 2 nested lists inside this, but we want to flatten them each to ints using bit shifting
        for sub_list in item:
            for sub_item in sub_list:
                is_solid_int = is_solid_int << 1
                is_solid_int = is_solid_int | sub_item
    obs_list.append(is_solid_int)
    for item in look_vector:
        for sub_item in item:
            for sub_sub_item in sub_item:
                obs_list.append(sub_sub_item)

    # Data type: numpy.float32
    # Shape: (6,)
    for item in cur_durability:
        obs_list.append(item)

    for item in max_durability:
        obs_list.append(item)

    for item in inc_name_by_craft:
        obs_list.append(string_to_index_mapping[item])
    for item in inc_quantity_by_craft:
        obs_list.append(item)

    for item in inc_name_by_other:
        obs_list.append(string_to_index_mapping[item])
    for item in inc_quantity_by_other:
        obs_list.append(item)

    for item in dec_name_by_craft:
        obs_list.append(string_to_index_mapping[item])
    for item in dec_quantity_by_craft:
        obs_list.append(item)

    for item in dec_name_by_other:
        obs_list.append(string_to_index_mapping[item])
    for item in dec_quantity_by_other:
        obs_list.append(item)

    for item in life:
        obs_list.append(item)

    for item in oxygen:
        obs_list.append(item)

    for item in armor:
        obs_list.append(item)

    for item in food:
        obs_list.append(item)

    for item in saturation:
        obs_list.append(item)

    obs_list.append(is_sleeping)

    # for item in xp:
    #     obs_list.append(item)
    #
    # #
    # for item in pos:
    #     obs_list.append(item)
    #
    # obs_list.append(damage_amount)

    return np.array(obs_list, dtype=np.float32)


input_shape = preprocess_observation(env.reset()).shape[0]
num_actions = env.action_space.nvec.tolist()
#
# if REDUCED_ACTION_SPACE:
#     # Keep actions 0 -> 5
#     num_actions = env.action_space.nvec.tolist()[0:6]

import logging


def get_args_mask_for_func_action(fun_act_idx) -> str:
    if (fun_act_idx == 0):
        return 'no-op'
    elif (fun_act_idx == 1):
        return 'use'
    elif (fun_act_idx == 2):
        return 'drop'
    elif (fun_act_idx == 3):
        return 'attack'
    elif (fun_act_idx == 4):
        return 'craft_smelt'
    elif (fun_act_idx == 5):
        return 'equip'
    elif (fun_act_idx == 6):
        return 'place'
    elif (fun_act_idx == 7):
        return 'destroy'
    else:
        return 'no-op'


def any_valid_arg(mask):
    # return true if any bool in mask is true, mask is 1d
    return np.any(mask)


example_probs = [np.random.rand(3), np.random.rand(3), np.random.rand(4), np.random.rand(25), np.random.rand(25), np.random.rand(8), np.random.rand(244), np.random.rand(36)]

example_probs = [torch.from_numpy(p) for p in example_probs]


def mask_apply(masks, multidiscrete_tensor, isRandom=False, returnProbs=False):
    """
             apply masks as follows:
              1. Determine functional action (index 5, mask 'action_type') (size 8 bools)
                 - action for subsequent steps is argmax of non-false masked values
              2. Determine if action takes args (index 5, mask 'action_arg') (size 8 bools)
              3. Based on action type, pick the corresponding mask from:
                 - 'equip'      (size 36 bools, corresponds to inventory slots)
                 - 'place'      (size 36 bools, corresponds to inventory slots)
                 - 'destroy'    (size 36 bools, corresponds to inventory slots)
                 - 'craft_smelt'    (size 244 bools, corresponds to all craftable items)
                 - no mask for other options
    :param masks: a dict of np array of bools with keys corresponding to what is being masked, as described above
    :param multidiscrete_tensor: list of tensors of probabilities for every possible action
    :return: the choice made based on argmax applied to each tensor in multidiscrete_tensor after masking
    """
    # detach all tensors from the graph
    multidiscrete_tensor = [t.detach().cpu() for t in multidiscrete_tensor]
    # Create a list of zero-filled tensors with the same shapes as multidiscrete_tensor
    result = [torch.zeros_like(t) for t in example_probs]
    for (i, t) in enumerate(multidiscrete_tensor):
        for (j, p) in enumerate(t):
            result[i][j] = p

    # Apply func mask to index 5
    func_mask = masks['action_type']
    print(func_mask)
    for i, is_enabled in enumerate(func_mask):
        if is_enabled and i < len(multidiscrete_tensor[5]):
            if i > 3 and any_valid_arg(masks[get_args_mask_for_func_action(i)]):
                result[5][i] = multidiscrete_tensor[5][i] if not isRandom else np.random.uniform() + 0.5
            elif i <= 3:
                result[5][i] = multidiscrete_tensor[5][i] if not isRandom else np.random.uniform() + 0.5

    for (i, t) in enumerate(result):
        for (j, p) in enumerate(t):
            if REDUCED_ACTION_SPACE:
                if i == 5:
                    if j == 0:
                        result[i][j] = p
                    elif j == 3:
                        result[i][j] = p
                    else:
                        result[i][j] = 0
                else:
                    result[i][j] = p
            else:
                result[i][j] = p
    print(len(result[5]))
    print(result[5])

    func_act = torch.argmax(result[5])
    print(f"func_act: {func_act}")
    # check arg mask with the choice
    args_mask = masks['action_arg']
    print(args_mask)
    if args_mask[func_act.item()]:
        msk = masks[get_args_mask_for_func_action(func_act.item())]
        idx_of_result_to_use = 6 if get_args_mask_for_func_action(func_act.item()) == 'craft_smelt' else 7
        for i, is_enabled in enumerate(msk):
            if idx_of_result_to_use < len(multidiscrete_tensor):
                if i < len(multidiscrete_tensor[idx_of_result_to_use]):
                    if is_enabled:
                        result[idx_of_result_to_use][i] = multidiscrete_tensor[idx_of_result_to_use][i] if not isRandom else np.random.uniform() + 0.5
                    else:
                        result[idx_of_result_to_use][i] = 0

    # argmax all the things
    if returnProbs:
        return result
    for i, t in enumerate(result):
        result[i] = torch.argmax(t).item()
    logging.debug(f"Masked result: {result}")
    if (env.action_space.contains(result)):
        return result
    else:
        logging.debug(f"Masked result is not valid: {result}")
        res = env.action_space.no_op()
        logging.debug(f"Returning no-op instead: {res}")
        return env.action_space.no_op()
    return result


import torch.nn.functional as F

def list_to_tensor(list):
    if type(list) == list:
        return torch.from_numpy(np.array(list)).float().to(device)
    else:
        return list

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dims):
        super(ActorCritic, self).__init__()
        self.device = device
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_branches = nn.ModuleList().to(self.device)
        # Set the desired dimensions for reshaping

        self.C, self.H, self.W = 1, 187, 220

        self.actor_branches = nn.ModuleList().to(self.device)
        for action_dim in action_dims:
            self.actor_branches.append(nn.Sequential(
                nn.Conv2d(self.C, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1),
                nn.Flatten(),
                nn.Flatten(),
                nn.Linear(621, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim),
                # dimension is now (128, action_dim). We want to have (action_dim).print
            ).to(self.device))

        self.critic = nn.Sequential(
            nn.Conv2d(self.C, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Flatten(),
            nn.Linear(621, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.1, 0.1)
                nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, state):
        if state.shape != (1, 1, 41140):
            print(f"Wrong state shape: {state.shape}")
            state = torch.zeros(41140).to(self.device)
        state = state.view(self.C, self.H, self.W)  # Reshape the input to (batch_size, C, H, W)
        print(f"Input state shape: {state.shape}")
        action_probs = [branch(state)[0] for branch in self.actor_branches]
        for i, probs in enumerate(action_probs):
            print(f"Action probs shape at index {i}: {probs.shape}")
        value = self.critic(state)
        return action_probs, value

    def choose_action(self, state, mask, epsilon=0.5):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        state = state.unsqueeze(0)  # Add an extra dimension for the batch size
        dist, _ = self.forward(state)
        action_probs = dist
        choosing_randomly = np.random.uniform() < epsilon
        # Call mask_multidiscrete_probs to mask the action_probs
        for i, probs in enumerate(action_probs):
            print(f"probs: {probs.shape} at index {i}")
        actions = mask_apply(mask, action_probs, isRandom=choosing_randomly)
        # Choose actions based on the masked action probabilities
        # actions = [np.argmax(masked_action_probs[i]) for i in range(len(masked_action_probs))]
        return actions

    def train(self, transitions, gamma=0.99):
        batch_size = len(transitions)
        states, actions, rewards, next_states, masks, done_masks = zip(*transitions)

        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        done_masks = torch.tensor(done_masks, dtype=torch.float).to(self.device)

        # Initialize lists for action_probs and state_values
        action_probs_list = [[] for _ in range(len(self.actor_branches))]
        state_values_list = []

        # Loop through the batch to compute action probabilities and state values
        for i in range(batch_size):
            action_probs, state_value = self.forward(states[i].unsqueeze(0))
            for j in range(len(self.actor_branches)):
                action_probs_list[j].append(action_probs[j])
            state_values_list.append(state_value)

        # Concatenate lists into tensors
        action_probs = [torch.stack(action_probs_list[j], dim=1) for j in range(len(self.actor_branches))]
        state_values = torch.cat(state_values_list, dim=0)

        # Calculate the next state values using the critic
        next_state_values = []
        for i in range(batch_size):
            _, value = self.forward(next_states[i].unsqueeze(0))
            next_state_values.append(value)
        next_state_values = torch.cat(next_state_values, dim=0)

        next_state_values = next_state_values * (1 - done_masks)

        # Calculate the expected state values
        expected_state_values = rewards + (gamma * next_state_values)

        # Calculate the value loss (MSE)
        value_loss = F.mse_loss(state_values, expected_state_values.detach())

        # Calculate the advantages
        advantages = expected_state_values - state_values

        # Calculate the action loss (negative log likelihood)
        # Calculate the action loss (negative log likelihood)
        action_loss = 0
        for j in range(len(self.actor_branches)):
            action_loss += -action_probs[j].gather(0, actions[:, j].view(1, -1)).squeeze() * advantages.detach()

        # Calculate the total loss
        total_loss = value_loss + action_loss.sum()

        # Optimize the model
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()


# log to file
LOG_FILENAME = 'training.log'
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
logging.info('Starting training')

print(minedojo.tasks.ALL_PROGRAMMATIC_TASK_IDS)
print(env.action_space.no_op())
print(f"Input shape: {input_shape} | Number of actions: {num_actions}")
# print(f"CUDA: {torch.cuda.is_available()}")
# print(f"Device: {torch.cuda.get_device_name(0)}")
# print(f"Device count: {torch.cuda.device_count()}")
example_act_tnsr = torch.from_numpy(env.action_space.no_op())
example_act_tnsr = torch.zeros_like(example_act_tnsr)

agent = ActorCritic(input_shape, num_actions)
print(num_actions)

batch_size = 1
num_epochs = 1000 * 15 * 15
max_episode_steps = 400

with tqdm(total=num_epochs * batch_size * max_episode_steps) as pbar:
    for epoch in range(num_epochs):
        episode_rewards = []
        print(f"Epoch: {epoch}")

        transitions = []

        for episode in range(batch_size):
            print(f"Episode: {episode}")
            obs = env.reset()
            state = preprocess_observation(obs)
            done = False
            episode_reward = 0

            for step in range(max_episode_steps):
                pbar.update(1)
                masks = obs["masks"]
                print("chooosing action, stoooooooooge!")
                action = agent.choose_action(state, masks, epsilon=0.3)
                print(f"action: {[i for i in action]}")
                logging.info(f"action: {[i for i in action]}")
                logging.info(f"masks: {masks}")
                obs, reward, done, _ = env.step(action)
                base_reward = reward
                print(f"reward: {reward}")

                next_state = preprocess_observation(obs)

                episode_reward += base_reward
                transitions.append((state, action, reward, next_state, masks, done))

                if done:
                    pbar.update(max_episode_steps - step)
                    break

                state = next_state

            episode_rewards.append(episode_reward)

        # Train the agent using collected transitions
        agent.train(transitions)

        mean_episode_reward = np.mean(episode_rewards)
        torch.save(agent.state_dict(), f'model_{epoch}.pt')
        logging.info(f"Epoch: {epoch + 1}/{num_epochs}, Mean Reward: {mean_episode_reward:.2f}, Epsilon: {0.5 / (epoch + 1)}")
        # print(f"Epoch: {epoch+1}/{num_epochs}, Mean Reward: {mean_episode_reward:.2f}")

    # save the model
    torch.save(agent.state_dict(), 'model.pt')
