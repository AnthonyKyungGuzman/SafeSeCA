import numpy as np
import multiprocessing as mp
from multiprocessing import sharedctypes
from gym import Env
from gym.spaces import Box, Dict
import ctypes

import gym
import wandb
from gym_panda_frite.envs.environment import Environment
from gym_panda_frite.envs.panda_frite_env_wrapper_reward_max import PandaFriteEnvWrapperRewardMax
from database_frite import Database_Frite
import os
from utils import MovingAverageReward
import contextlib


# Utility to handle different numpy dtypes
_NP_TO_CT = {
    np.float32: ctypes.c_float,
    np.float64: ctypes.c_double,
    np.int32: ctypes.c_int32,
    np.int64: ctypes.c_int64,
    np.uint8: ctypes.c_uint8,
    np.bool_: ctypes.c_bool
}

def obs_space_info(obs_space, shape, dtype):
    if isinstance(obs_space, Dict):
        keys = list(obs_space.spaces.keys())
        shapes = {k: obs_space.spaces[k].shape for k in keys}
        dtypes = {k: obs_space.spaces[k].dtype for k in keys}
    else:
        keys = ['obs']
        shapes = {'obs': shape}
        dtypes = {'obs': dtype}
    return keys, shapes, dtypes

def obs_to_dict(obs):
    if isinstance(obs, dict):
        return obs
    else:
        return {'obs': obs}

class CloudpickleWrapper:
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


@contextlib.contextmanager
def clear_mpi_env_vars():
    '''
    from mpi4py import MPI will call MPI_Init by default.  If the child process has MPI environment variables, MPI will think that the child process is an MPI process just like the parent and do bad things such as hang.
    This context manager is a hacky way to clear those environment variables temporarily such as when we are starting multiprocessing Processes.
    '''
    removed_environment = {}
    for k, v in list(os.environ.items()):
        for prefix in ['OMPI_', 'PMI_']:
            if k.startswith(prefix):
                removed_environment[k] = v
                del os.environ[k]
    try:
        yield
    finally:
        os.environ.update(removed_environment)



def subproc_worker(pipe, parent_pipe, worker_id, params, obs_bufs, obs_shapes, obs_dtypes, keys):
    def _write_obs(maybe_dict_obs):
        flatdict = obs_to_dict(maybe_dict_obs)
        for k in keys:
            dst = obs_bufs[k].get_obj()
            dst_np = np.frombuffer(dst, dtype=obs_dtypes[k]).reshape(obs_shapes[k])
            np.copyto(dst_np, flatdict[k])


    env_pybullet = Environment(json_decoder=params.json_decoder, gui=params.args.gui)
    env_pybullet.startThread()
    db = Database_Frite(json_decoder=params.json_decoder)
    env = PandaFriteEnvWrapperRewardMax(gym.make(params.env_name, database=db, json_decoder = params.json_decoder, env_pybullet=env_pybullet, gui=params.args.gui, env_rank=worker_id))
    env.seed(params.env_random_seed + worker_id)
    panda_num = 0 


    parent_pipe.close()
    print("Env {} has been created, ready to start".format(worker_id))
    
    step = 0 
    episode = 0
    print_in_done = False

    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == 'reset':
                step = 0
                print_in_done = False
                episode += 1
                env.reset_env_bullet()
                state, initial_error = env.reset_bullet(panda_num)
                pos, orien = env.get_gripper_pos_orientation(panda_num)
                pos_orien = [pos, orien]
                pipe.send((_write_obs(state),pos_orien))
            elif cmd == 'step':
                step += 1 
                obs, reward, done, info = env.step_bullet(data,panda_num,worker_id,episode,step)
                pos, orien = env.get_gripper_pos_orientation(panda_num)
                pos_orien = [pos, orien]
                if done:
                    if not print_in_done:
                        print("Achieved final pose worker_id:{}  step: {} ............. DONE!!!!\n".format(worker_id, step))
                        print_in_done = True
                #     env.reset_env_bullet()
                #     obs, initial_error = env.reset_bullet(panda_num)
                pipe.send((_write_obs(obs), reward, done, info, pos_orien))
            elif cmd == 'close':
                pipe.send(None)
                break
            else:
                raise RuntimeError(f'Unrecognized command {cmd}')
    finally:
        env.close()



class VecEnv:
    def __init__(self, num_workers, sim_params, context='spawn'):
        ctx = mp.get_context(context)
        
        observation_space, action_space = sim_params.seca_args.state_shape, sim_params.seca_args.n_actions
        self.spec = sim_params

        self.moving_avg_return = [MovingAverageReward(window_size=100) for i in range(num_workers)]

        self.num_envs = [i for i in range(num_workers)]
        self.obs_keys, self.obs_shapes, self.obs_dtypes = obs_space_info(observation_space, observation_space, np.float32)
        self.obs_bufs = [
            {k: ctx.Array(_NP_TO_CT[self.obs_dtypes[k]], int(np.prod(self.obs_shapes[k]))) for k in self.obs_keys}
            for _ in self.num_envs 
        ]
        self.parent_pipes = []
        self.procs = []
        worker_id=0
        with clear_mpi_env_vars():
            for id, obs_buf in zip(self.num_envs, self.obs_bufs):
                worker_id = id + 1
                parent_pipe, child_pipe = ctx.Pipe()
                proc = ctx.Process(
                    target=subproc_worker,
                    args=(child_pipe, parent_pipe, worker_id, sim_params, obs_buf, self.obs_shapes, self.obs_dtypes, self.obs_keys)
                )
                proc.daemon = True
                self.procs.append(proc)
                self.parent_pipes.append(parent_pipe)
                proc.start()
                child_pipe.close()

        self.waiting_step = False
        self.closed = False

    def _obs_from_buf(self):
        out = []
        for buf in self.obs_bufs:
            ob = {k: np.frombuffer(buf[k].get_obj(), dtype=self.obs_dtypes[k])
                      .reshape(self.obs_shapes[k]) for k in self.obs_keys}
            out.append(ob)
        if len(self.obs_keys) == 1:
            return np.stack([o[self.obs_keys[0]] for o in out])
        else:
            return {k: np.stack([o[k] for o in out]) for k in self.obs_keys}

    def reset(self):
        for pipe in self.parent_pipes:
            pipe.send(("reset", None))
        results = [pipe.recv() for pipe in self.parent_pipes]
        obs, pos_orien = zip(*results)
        obs = self._obs_from_buf()
        return obs, pos_orien
        # for pipe in self.parent_pipes:
        #     pipe.recv()
        # return self._obs_from_buf()

    def step_async(self, actions):
        self._actions = actions
        for pipe, action in zip(self.parent_pipes, actions):
            pipe.send(("step", action))
        self.waiting_step = True

    def step_wait(self):
        results = [pipe.recv() for pipe in self.parent_pipes]
        self.waiting_step = False
        obs, rewards, dones, infos, pos_orien = zip(*results)
        obs = self._obs_from_buf()
        return obs, np.stack(rewards), np.stack(dones), infos, pos_orien

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):
        if self.closed:
            return
        if self.waiting_step:
            for pipe in self.parent_pipes:
                pipe.recv()
        for pipe in self.parent_pipes:
            pipe.send(("close", None))
        for pipe in self.parent_pipes:
            pipe.recv()
            pipe.close()
        for proc in self.procs:
            proc.join()
        self.closed = True
