import numpy as np, os, time, sys, random
import gym
import argparse
import time
import logging
import copy
import ray,utils
from utils import Actor
import threading,queue
from ray.rllib.utils.timer import TimerStat
from pg_policy_graph import PGPolicyGraph
from ray.rllib.agents.agent import Agent
from ray.rllib.utils.annotations import override
from ray.rllib.optimizers import SyncSamplesOptimizer
from ray.rllib.agents.es import policies
import tensorflow as tf


render = False
parser = argparse.ArgumentParser()
parser.add_argument('-env', help='Environment Choices: (HalfCheetah-v2) (Ant-v2) (Reacher-v2) (Walker2d-v2) (Swimmer-v2) (Hopper-v2)', required=True)
env_tag = vars(parser.parse_args())['env']


logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.setLevel(level=logging.DEBUG)


# yapf: disable
# __sphinx_doc_begin__
DEFAULT_CONFIG = {
    # No remote workers by default
    "num_workers": 0,
    # Learning rate
    "lr": 0.0004,
    # Use PyTorch as backend
    "use_pytorch": False,
}
# __sphinx_doc_end__
# yapf: enable


class Parameters:
    def __init__(self):
        #Number of Frames to Run
        if env_tag == 'Hopper-v2': self.num_frames = 4000000
        elif env_tag == 'Ant-v2': self.num_frames = 6000000
        elif env_tag == 'Walker2d-v2': self.num_frames = 8000000
        else: self.num_frames = 2000000

        #USE CUDA
        self.is_cuda = True; self.is_memory_cuda = True

        #Sunchronization Period
        if env_tag == 'Hopper-v2' or env_tag == 'Ant-v2': self.synch_period = 1
        else: self.synch_period = 10

        #DDPG params
        self.use_ln = True  # True
        self.gamma = 0.99; self.tau = 0.001
        self.seed = 7
        self.batch_size = 128
        self.buffer_size = 1000000
        self.frac_frames_train = 1.0
        self.use_done_mask = True

        ###### NeuroEvolution Params ########
        #Num of trials
        if env_tag == 'Hopper-v2' or env_tag == 'Reacher-v2': self.num_evals = 5
        elif env_tag == 'Walker2d-v2': self.num_evals = 3
        else: self.num_evals = 1

        #Elitism Rate
        if env_tag == 'Hopper-v2' or env_tag == 'Ant-v2': self.elite_fraction = 0.3
        elif env_tag == 'Reacher-v2' or env_tag == 'Walker2d-v2': self.elite_fraction = 0.2
        else: self.elite_fraction = 0.1

        self.pop_size = 10
        self.crossover_prob = 0.0
        self.mutation_prob = 0.9

        #Save Results
        self.state_dim = None; self.action_dim = None #Simply instantiate them here, will be initialized later
        self.save_foldername = 'test3-debug/%s/' % env_tag
        if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)


class OUNoise:
    def __init__(self, action_dimension, scale=0.3, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale


class ActorPolicy(object):
    def __init__(self, hparams, sess):
        # initialization
        self._s = sess

        # build the graph
        self._input = tf.placeholder(tf.float32,
                                     shape=[None, hparams['input_size']])

        hidden1 = tf.contrib.layers.fully_connected(
            inputs=self._input,
            num_outputs=hparams['hidden_size'],
            activation_fn=tf.nn.relu,
            weights_initializer=tf.random_normal)

        logits = tf.contrib.layers.fully_connected(
            inputs=hidden1,
            num_outputs=hparams['num_actions'],
            activation_fn=None)

        # op to sample an action
        self._sample = tf.reshape(tf.multinomial(logits, 1), [])

        # get log probabilities
        log_prob = tf.log(tf.nn.softmax(logits))

        # training part of graph
        self._acts = tf.placeholder(tf.int32)
        self._advantages = tf.placeholder(tf.float32)

        # get log probs of actions from episode
        indices = tf.range(0, tf.shape(log_prob)[0]) * tf.shape(log_prob)[1] + self._acts
        act_prob = tf.gather(tf.reshape(log_prob, [-1]), indices)

        # surrogate loss
        loss = -tf.reduce_sum(tf.mul(act_prob, self._advantages))

        # update
        optimizer = tf.train.RMSPropOptimizer(hparams['learning_rate'])
        self._train = optimizer.minimize(loss)

    def act(self, observation):
        # get one action, by sampling
        return self._s.run(self._sample, feed_dict={self._input: [observation]})

    def train_step(self, obs, acts, advantages):
        batch_feed = {self._input: obs, \
                      self._acts: acts, \
                      self._advantages: advantages}
        self._s.run(self._train, feed_dict=batch_feed)


@ray.remote(num_gpus=0.2)
class Worker(object):
    def __init__(self, args):
        self.env = utils.NormalizedActions(gym.make(env_tag))
        self.args = args
        self.ounoise = OUNoise(args.action_dim)
        self.sess = utils.make_session(single_threaded=True)
        self.policy = ActorPolicy(self.sess, self.args)

    def do_rollout(self, is_action_noise=False, store_transition=True):
        total_reward = 0.0
        state = self.env.reset()
        state = utils.to_tensor(state).unsqueeze(0)
        if self.args.is_cuda:
            state = state.cuda()
        done = False

        while not done:
            action = self.policy.act(state)
            action.clamp(-1, 1)
            action = utils.to_numpy(action.cpu())
            if is_action_noise: action += self.ounoise.noise()
            next_state, reward, done, info = self.env.step(action.flatten())  # Simulate one step in environment
            next_state = utils.to_tensor(next_state).unsqueeze(0)
            if self.args.is_cuda:
                next_state = next_state.cuda()
            total_reward += reward

            if store_transition:
                self.add_experience(state, action, next_state, reward, done)
            state = next_state
        if store_transition: self.num_games += 1
        return total_reward


if __name__ == "__main__":
    num_workers = 10
    parameters = Parameters()

    # Create Env
    env = utils.NormalizedActions(gym.make(env_tag))
    parameters.action_dim = env.action_space.shape[0]
    parameters.state_dim = env.observation_space.shape[0]

    env.seed(parameters.seed)
    tf.random.set_random_seed(parameters.seed)
    np.random.seed(parameters.seed)
    random.seed(parameters.seed)

    ray.init(include_webui=False, ignore_reinit_error=True)
    workers = [Worker.remote(parameters)
               for _ in range(num_workers)]

    rollout_ids = [worker.do_rollouts.remote() for worker in workers]
    results = ray.get(rollout_ids)
    print(results)







# if __name__ == "__main__":
#     parameters = Parameters()  # Create the Parameters class
#     tracker = utils.Tracker(parameters, ['erl'], '_score.csv')  # Initiate tracker
#     frame_tracker = utils.Tracker(parameters, ['frame_erl'], '_score.csv')  # Initiate tracker
#     time_tracker = utils.Tracker(parameters, ['time_erl'], '_score.csv')
#
#     # Create Env
#     env = utils.NormalizedActions(gym.make(env_tag))
#     parameters.action_dim = env.action_space.shape[0]
#     parameters.state_dim = env.observation_space.shape[0]
#
#     logger.debug("action_dim:{0},parameters.state_dim:{1}".format(parameters.action_dim, parameters.state_dim))
#
#     # Seed
#     env.seed(parameters.seed);
#     torch.manual_seed(parameters.seed);
#     np.random.seed(parameters.seed);
#     random.seed(parameters.seed)
#
#     # Create Agent
#     ray.init(include_webui=False, ignore_reinit_error=True)
#     # print(torch.cuda.device_count())
#
#     agent = Agent(parameters, env)
#     print('Running', env_tag, ' State_dim:', parameters.state_dim, ' Action_dim:', parameters.action_dim)
#
#     next_save = 100;
#     time_start = time.time()
#     while True:  # agent.num_frames <= parameters.num_frames:
#         best_train_fitness, erl_score, elite_index = agent.train()
#         print('#Games:', agent.num_games, '#Frames:', agent.num_frames, ' Epoch_Max:',
#               '%.2f' % best_train_fitness if best_train_fitness != None else None, ' Test_Score:',
#               '%.2f' % erl_score if erl_score != None else None, ' Avg:', '%.2f' % tracker.all_tracker[0][1],
#               'ENV ' + env_tag)
#         print('RL Selection Rate: Elite/Selected/Discarded',
#               '%.2f' % (agent.evolver.selection_stats['elite'] / agent.evolver.selection_stats['total']),
#               '%.2f' % (agent.evolver.selection_stats['selected'] / agent.evolver.selection_stats['total']),
#               '%.2f' % (agent.evolver.selection_stats['discarded'] / agent.evolver.selection_stats['total']))
#
#         # log experiment result
#         tracker.update([erl_score], agent.num_games)
#         frame_tracker.update([erl_score], agent.num_frames)
#         time_tracker.update([erl_score], time.time() - time_start)
#
#         # Save Policy
#         if agent.num_games > next_save:
#             next_save += 100
#             if elite_index != None: torch.save(agent.pop[elite_index].state_dict(),
#                                                parameters.save_foldername + 'evo_net')
#             print("Progress Saved")
#
#         exit(0)







