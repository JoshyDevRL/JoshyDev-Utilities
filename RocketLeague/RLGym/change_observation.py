from stable_baselines3 import PPO
from gym import spaces
import torch
import math

custom_objects = {
    "lr_schedule": 0.00001,
    "clip_range": .02,
    "n_envs": 1,
    "device": "cpu"
}

model = PPO.load("model.zip", custom_objects=custom_objects)
policy = model.policy

NEW_INPUT_SIZE = 265
EXPANDED_WEIGHT = 0.001

space = spaces.Box(low=-math.inf, high=math.inf, shape=(NEW_INPUT_SIZE,))
policy.observation_space = space
model.observation_space = space
old_layer = policy.mlp_extractor.policy_net[0]
new_layer = torch.nn.Linear(NEW_INPUT_SIZE, policy.mlp_extractor.policy_net[0].out_features)
for j in range(len(new_layer.weight)):
    for z in range(len(new_layer.weight[j])):
        with torch.no_grad():
            new_layer.weight[j][z] = EXPANDED_WEIGHT

for i in range(len(old_layer.weight)):
    for x in range(len(old_layer.weight[i])):
        with torch.no_grad():
            new_layer.weight[i][x] = old_layer.weight[i][x].clone()

new_layer.bias = old_layer.bias
policy.mlp_extractor.policy_net[0] = new_layer

model.save("new_model.zip")