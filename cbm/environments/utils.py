import gym
import warnings

env_name_to_gym_registry_dict = {
    "half_cheetah": "HalfCheetah-v2",
    "cheetah": "HalfCheetah-v2",
    "swimmer": "Swimmer-v2",
    "ant": "Ant-v2",
    "mb_ant": "AntTruncatedObs-v2",
    "hopper": "Hopper-v2",
    "walker2d": "Walker2d-v2",
    "humanoid": "Humanoid-v2",
    "mb_humanoid": "HumanoidTruncatedObs-v2",
    "2dpointer": "MultiGoal2D-v0",
    "inverted_pendulum": "InvertedPendulum-v2"
}
def env_name_to_gym_registry(env_name):
    if env_name in env_name_to_gym_registry_dict:
        return env_name_to_gym_registry_dict[env_name]
    return env_name

def make_gym_env(env_name, seed):
    env = gym.make(env_name_to_gym_registry(env_name)).env
    env.seed(seed)
    return env

def get_make_fn(env_name, seed):
    def make():
        env = gym.make(env_name_to_gym_registry(env_name)).env
        env.seed(seed)
        return env
    return make

def get_make_fns(env_name, seeds, n_env=1):
    if seeds is None:
        seeds = [None] * n_env
    elif len(seeds) != n_env:
        warnings.warn('the length of the seeds is different from n_env')

    make_fns = [get_make_fn(env_name, seed) for seed in seeds]
    return make_fns
