import numpy as np
import gym,os, time, sys, random
import argparse
import logging
import ray, utils
import tensorflow as tf
from tensorflow.python.ops import random_ops
import ray.experimental.tf_utils
from policy import ActorPolicy
import mod_neuro_evo as utils_ne
import copy




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


def make_session(single_threaded):
    if not single_threaded:
        return tf.Session()
    config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


class Parameters:
    def __init__(self):
        self.input_size = None
        self.hidden_size = 36
        self.num_actions = None
        self.learning_rate = 0.1

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


@ray.remote(num_gpus=0.1)
class Worker(object):
    def __init__(self, args):
        self.env = utils.NormalizedActions(gym.make(env_tag))
        self.args = args
        self.ounoise = OUNoise(args.action_dim)
        self.sess = make_session(single_threaded=True)
        self.policy = ActorPolicy(self.args.action_dim, self.args.state_dim, self.sess)

    def do_rollout(self, params,is_action_noise=False, store_transition=True):
        total_reward = 0.0
        if params:
            self.policy.set_weights(params)
        state = self.env.reset()
        # state = utils.to_tensor(state).unsqueeze(0)
        # if self.args.is_cuda:
        #     state = state.cuda()
        done = False
        while not done:
            action = self.policy.choose_action(state)
            # action.clamp(-1, 1)
            # action = utils.to_numpy(action.cpu())
            if is_action_noise: action += self.ounoise.noise()
            next_state, reward, done, info = self.env.step(action)  # Simulate one step in environment
            # next_state = utils.to_tensor(next_state).unsqueeze(0)
            # if self.args.is_cuda:
            #     next_state = next_state.cuda()
            total_reward += reward

            if store_transition:
                self.policy.store_transition(state, action, reward)
            state = next_state
        # if store_transition: self.num_games += 1
        self.policy.learn()

        return total_reward, self.policy.get_weights()


def process_results(results):
    pops = []
    fitness = []
    for result in results:
        pops.append(result[1])
        fitness.append(result[0])
    return fitness, pops


if __name__ == "__main__":
    # time_start = time.time()
    num_workers = 10
    parameters = Parameters()
    # tf.enable_eager_execution()

    # Create Env
    env = utils.NormalizedActions(gym.make(env_tag))
    parameters.action_dim = env.action_space.shape[0]
    parameters.num_actions = env.action_space.shape[0]
    parameters.state_dim = env.observation_space.shape[0]
    parameters.input_size = env.observation_space.shape[0]

    env.seed(parameters.seed)
    tf.set_random_seed(parameters.seed)
    np.random.seed(parameters.seed)
    random.seed(parameters.seed)

    ray.init(include_webui=False, ignore_reinit_error=True)
    workers = [Worker.remote(parameters)
               for _ in range(num_workers)]
    pops_new = [None for _ in range(num_workers)]
    print(pops_new)

    while True:
        # parallel pg process
        time_start = time.time()
        rollout_ids = [worker.do_rollout.remote(pop_params) for worker, pop_params in zip(workers,pops_new)]
        results = ray.get(rollout_ids)
        all_fitness, pops = process_results(results)
        print("maximum score,", max(all_fitness))
        time_evaluate = time.time()-time_start
        time_middle = time.time()
        print("time for evalutation,",time_evaluate)
        pops_new = copy.deepcopy(pops)

        # evolver process
        evolver = utils_ne.SSNE(parameters)
        elite_index = evolver.epoch(pops_new, all_fitness)
        print("elite_index,", elite_index)
        time_evolve = time.time()-time_middle
        print("time for evolve,", time_evolve)











