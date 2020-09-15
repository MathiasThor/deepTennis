# Tennis control using a Proximal Policy Optimization (PPO) Agents

## Introduction
This project is about training two agents to control rackets to bounce a ball over a net. The project uses the Unity ML-Agents Tennis Environment. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it gets a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

IMAGE HERE!

The agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,
- After each episode, we add up the rewards that each agent received (without discounting) to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these two scores.
- This yields a single score for each episode.

The task is considered solved when the average (over 100 episodes) of those scores is at least +0.5.

## Learning algorithm
The learning algorithm is split into to main functions: the `rollout_manger()` and the `PPO_algorithm()`. 

### Rollout manager
As the name indicates, `rollout_manger()` is responsible for collecting rollouts or trajectories from the two agents, which can be used for training the PPO agent. Specifically, it collects and stores: actions, log probability of the actions, values, rewards, episode done status, and states. It also calculates the future discounted returns as well as estimating the advantages.

### PPO agent
The `PPO_algorithm()` implements the Proximal Policy Optimization algorithm and is in charge of optimizing the agents’ selection of actions in order to achieve the highest score. The PPO algorithm is adapted to MARL (Multi-Agent reinforcement learning) by using a meta agent like approach in which a single policy is learned for both agents. However, each agent uses its own observations to achieve the best action, resulting in their own personal reward. The advantage is that by using two agents to train the same policy, the trajectories can be sampled faster, and thus it may learn more quickly.

The `PPO_algorithm()` first instantiates the actor and critic networks, which convert 24 input states into a predicted best action and estimated value, respectively. **Both networks have the 24 states as input and two fully connected layers with 32 neurons in each. Both networks also use the ReLu activation function between all layers.**

**The actor-network has 2 outputs**, representing the mean of the predicted distribution of the best actions for the current input. The standard deviation can be calculated from these means, and the best action can then be sampled from the resulting normal distribution.

**The critic-network has 1 output**, which represents the estimated value of the input state. The estimated value is subsequentially converted into the advantage estimate as follows (remember this function is placed in the `rollout_manger()` class):
```python
advantages = torch.tensor(np.zeros((num_agents, 1)), device=DEVICE, dtype=torch.float32)
for i in reversed(range(HORIZON)):
  td = self.rewards[i] + (GAE_DISCOUNT * self.episode_not_dones[i] * self.values[i + 1]) - self.values[i]
  advantages = advantages * GAE_LAMBDA * GAE_DISCOUNT * self.episode_not_dones[i] + td
  self.advantages[i] = advantages.detach() 
```
The advantage estimate basically tells us, “How much better was the action that I took based on the expectation of what would normally happen in the state that I was in”.

Finally, the policy loss (for the actor) and value loss (for the critic) is calculated as follows:
```python
  # Policy Loss
  ratio = (log_prob_action - sampled_log_probs_old).exp() 
  obj = ratio * sampled_advantages
  obj_clipped = ratio.clamp(1.0 - PPO_CLIP_RANGE, 1.0 + PPO_CLIP_RANGE) * sampled_advantages
  policy_loss = -torch.min(obj, obj_clipped).mean() 

  # Value Loss
  value_loss = 0.5 * (sampled_returns - value).pow(2).mean()
```
Note that the policy loss is clipped according to the PPO algorithm. This is done to ensure that the updated policy does not move too far away from the current policy. When moving too far away, the advantage estimate becomes inaccurate and may destroy the policy. The ratio is used to reweigh the advantages which enable us to use sampled tractories generated under a previous policy.

After computing the policy and value loss, the weights if the actor and critic networks can be updated as follows:
```python
  # Optimize network weights
  loss = policy_loss + value_loss
  optimizer.zero_grad()
  loss.backward()
  nn.utils.clip_grad_norm_(self.network.parameters(), 0.75) 
  optimizer.step()
```

### Training, plotting, and testing
For training, the agent with the PPO algorithm `train_agent()` is used. The training function will continue to iterate through a training loop until either a maximum number of episodes is reached or a specified mean score over 100 episodes is achieved. Afterward, the actor and critic network’s weights are saved, and the scores for each episode are written to a CSV file. This CSV file can later be used for plotting the scores for the entire or multiple training sessions (as we show later in this report). Finally, the learned network weights can be loaded, and the agent can be tested to verify the results visually. 

## Hyperparameters
The hyperparameters are set as follows:

| Hyperparametser | Value | Description |
|--|--|--|
| HORIZON | 275 | PPO gathers trajectories as far out as the horizon limits, then performs a stochastic gradient descent (SGD) update |
| DISCOUNT_FACTOR | 0.99 | Discount factor used for calculating future returns |
| GAE_DISCOUNT | 0.99 | GAE Discount factor performs a bias-variance trade-off of the trajectories and can be viewed as a form of reward shaping |
| GAE_LAMBDA | 0.95 | GAE Lambda perform a bias-variance trade-off of the trajectories and can be viewed as a form of reward shaping  |
| EPOCH_RANGE | 12 | Number of updates per optimize step |
| MINIBATCH_SIZE | 64 | Minibatch size used for the optimize step |
| PPO_CLIP_RANGE | 0.05, 0.1, 0.15 | PPO uses a surrogate loss function to keep the step from the old policy to the new policy within a safe range. This hyperparameter sets the clipping of this surrogate loss function |
| LEARNING_RATE | 0.0003 | The learning rate used by the optimizer |
| | | |

Due to this project’s scope, only the `PPO_CLIP_RANGE` hyperparameter will be tested for different settings, while the remaining hyperparameters will remain fixed. The results of varying `PPO_CLIP_RANGE` can be seen in the following section.

## Experiments and results
The following plot shows the results of using three different values of `PPO_CLIP_RANGE` (0.05, 0.10, and 0.15). For each `PPO_CLIP_RANGE` value, the learning algorithm was executed for 3000 episodes and repeated five times. This makes it possible to calculate and plot the mean return and standard error. The thin and light blue, orange, and green lines show the mean scores, while the standard error is shown with transparent areas around these lines. The thicker and darker blue, orange, and green lines show the mean score averaged over 100 episodes.

IMAGE HERE!
*Plot of the mean score for all 20 robots per episode when using three different PPO clip range parameters*

As can be seen, all three PPO clipping ranges can achieve an average score of at least 0.5 (more than 1 in fact) for more than 100 episodes, thus meeting this project’s goal. A PPO_CLIP_RANGE = 0.15 converge the fastest as it only needs around 1000 episodes on average to meet the goal score of 0.5. However, both `PPO_CLIP_RANGE` = 0.05 and `PPO_CLIP_RANGE` = 0.1 are able to achieve larger average scores (around 1.3). In the `/saved_weights` directory a weight set using `PPO_CLIP_RANGE` = 0.05 is placed. This can be loaded and tested with the “Load and test trained agent” cell. 

## Future work
In the future, it would be beneficial to do more parameter tuning (e.g., higher PPO clipping ranges). It would also be interesting to look at different network architectures for the actor and critic networks to see how this affects the learning performance. Finally, it would be interesting to investigate if the `PPO_CLIP_RANGE` could be adapted during learning (e.g., start high and then decrease).

