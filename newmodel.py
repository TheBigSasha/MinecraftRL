from torch import nn


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value

    def choose_action(self, state, masks):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = self.forward(state)
        action_probs = action_probs * masks["action_type"] + 1e-8

        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item()

    def train(self, experience_buffer, ppo_epochs=10, mini_batch_size=64, gamma=0.99, epsilon=0.2):
        states, actions, rewards, masks, next_states = map(torch.FloatTensor, zip(*experience_buffer))

        for _ in range(ppo_epochs):
            for idx in range(0, len(experience_buffer), mini_batch_size):
                mini_batch_indices = torch.randperm(len(experience_buffer))[idx:idx+mini_batch_size]

                mb_states = states[mini_batch_indices]
                mb_actions = actions[mini_batch_indices].unsqueeze(-1)
                mb_rewards = rewards[mini_batch_indices].unsqueeze(-1)
                mb_masks = masks["action_type"][mini_batch_indices].unsqueeze(-1)
                mb_next_states = next_states[mini_batch_indices]

                _, old_state_values = self.forward(mb_states)
                old_state_values = old_state_values.detach()
                old_action_probs, _ = self.forward(mb_states)
                old_action_probs = old_action_probs.gather(1, mb_actions).detach()

                # Calculate advantages and target values
                _, next_state_values = self.forward(mb_next_states)
                target_values = mb_rewards + gamma * next_state_values
                advantages = target_values - old_state_values

                # Update the networks
                self.optimizer.zero_grad()

                # Policy loss (actor)
                new_action_probs, _ = self.forward(mb_states)
                new_action_probs = new_action_probs.gather(1, mb_actions)
                ratio = (new_action_probs / old_action_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (critic)
                _, new_state_values = self.forward(mb_states)
                value_loss = nn.MSELoss()(new_state_values, target_values.detach())

                # Total loss
                loss = policy_loss + value_loss
                loss.backward()
                self.optimizer.step()



batch_size = 256
num_epochs = 100
max_episode_steps = 500
experience_buffer = []

for epoch in range(num_epochs):
    episode_rewards = []

    for episode in range(batch_size):
        state = preprocess_observation(env.reset())
        done = False
        episode_reward = 0

        for step in range(max_episode_steps):
            obs = env.reset()
            masks = obs["masks"]
            action = agent.choose_action(state, masks)
            obs, reward, done, _ = env.step(action)
            next_state = preprocess_observation(obs)
            experience_buffer.append((state, action, reward, masks, next_state))
            state = next_state
            episode_reward += reward

            if done:
                break

        episode_rewards.append(episode_reward)

        # Train the model using the experience buffer
        if len(experience_buffer) >= batch_size:
            agent.train(experience_buffer)
            experience_buffer = []

    print(f"Epoch {epoch + 1}, Average Reward: {sum(episode_rewards) / len(episode_rewards)}")
