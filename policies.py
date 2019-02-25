# Code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
import tensorflow as tf

from ray.rllib.evaluation.sampler import _unbatch_tuple_actions
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.filter import get_filter
import ray,utils


def rollout(policy, env, timestep_limit=None, add_noise=False):
    total_reward = 0.0
    state = env.reset()
    state = utils.to_tensor(state).unsqueeze(0)
    if self.args.is_cuda:
        state = state.cuda()
    done = False

    while not done:
        if store_transition: self.num_frames += 1; self.gen_frames += 1
        if render and is_render: self.env.render()
        # print(state)
        # exit(0)
        action = self.pop.forward(state)
        action.clamp(-1, 1)
        action = utils.to_numpy(action.cpu())
        if is_action_noise: action += self.ounoise.noise()
        # print("come there in evaluate")
        next_state, reward, done, info = self.env.step(action.flatten())  # Simulate one step in environment
        # print("come there in evaluate")
        next_state = utils.to_tensor(next_state).unsqueeze(0)
        if self.args.is_cuda:
            next_state = next_state.cuda()
        total_reward += reward

        if store_transition: self.add_experience(state, action, next_state, reward, done)
        state = next_state
    if store_transition: self.num_games += 1
    # print("come here,total_reward:",total_reward)
    return total_reward



class ActorPolicy(object):
    def __init__(self, sess, action_space, obs_space, preprocessor,
                 observation_filter, model_options, action_noise_std):
        self.sess = sess
        self.action_space = action_space
        self.action_noise_std = action_noise_std
        self.preprocessor = preprocessor
        self.observation_filter = get_filter(observation_filter,
                                             self.preprocessor.shape)
        self.inputs = tf.placeholder(tf.float32,
                                     [None] + list(self.preprocessor.shape))

        # Policy network.
        dist_class, dist_dim = ModelCatalog.get_action_dist(
            self.action_space, model_options, dist_type="deterministic")
        model = ModelCatalog.get_model({
            "obs": self.inputs
        }, obs_space, dist_dim, model_options)
        dist = dist_class(model.outputs)
        self.sampler = dist.sample()

        self.variables = ray.experimental.TensorFlowVariables(
            model.outputs, self.sess)

        self.num_params = sum(
            np.prod(variable.shape.as_list())
            for _, variable in self.variables.variables.items())
        self.sess.run(tf.global_variables_initializer())

    def compute(self, observation, add_noise=False, update=True):
        observation = self.preprocessor.transform(observation)
        observation = self.observation_filter(observation[None], update=update)

        action = self.sess.run(
            self.sampler, feed_dict={self.inputs: observation})
        action = _unbatch_tuple_actions(action)
        if add_noise and isinstance(self.action_space, gym.spaces.Box):
            action += np.random.randn(*action.shape) * self.action_noise_std
        return action

    def set_weights(self, x):
        self.variables.set_flat(x)

    def get_weights(self):
        return self.variables.get_flat()

    def get_filter(self):
        return self.observation_filter

    def set_filter(self, observation_filter):
        self.observation_filter = observation_filter
