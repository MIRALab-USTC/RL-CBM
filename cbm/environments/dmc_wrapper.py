# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# from drqv2
from collections import deque
from typing import Any, NamedTuple
import gym
import dm_env
import numpy as np
from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_control.suite import common
from dm_env import StepType, specs
import cbm.external_libs.distracting_control.suite as dcs_suite
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import cv2
import os


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for _ in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)

class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class SimpleFramekWrapper(dm_env.Environment):
    def __init__(self, env, pixels_key='pixels'):
        self._env = env
        self._pixels_key = pixels_key
        inner_obs_spec = env.observation_spec()
        assert pixels_key in inner_obs_spec
        pixels_shape = inner_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._pixel_spec = specs.BoundedArray(shape=np.array(
            [pixels_shape[2], *pixels_shape[:2]]),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name='observation'
        )
        inner_obs_spec[pixels_key] = self._pixel_spec
        self._obs_spec = inner_obs_spec
        
    def _trans_pixels(self, time_step):
        obs = time_step.observation
        pixels = obs[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        obs[self._pixels_key]=pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        self._trans_pixels(time_step)
        return time_step

    def step(self, action):
        time_step = self._env.step(action)
        self._trans_pixels(time_step)
        return time_step

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def rgb_to_hsv(r, g, b):
    """Convert RGB color to HSV color"""
    maxc = max(r, g, b)
    minc = min(r, g, b)
    v = maxc
    if minc == maxc:
        return 0.0, 0.0, v
    s = (maxc-minc) / maxc
    rc = (maxc-r) / (maxc-minc)
    gc = (maxc-g) / (maxc-minc)
    bc = (maxc-b) / (maxc-minc)
    if r == maxc:
        h = bc-gc
    elif g == maxc:
        h = 2.0+rc-bc
    else:
        h = 4.0+gc-rc
    h = (h/6.0) % 1.0
    return h, s, v


def do_green_screen(x, bg):
    """Removes green background from observation and replaces with bg; not optimized for speed"""
    assert isinstance(x, np.ndarray) and isinstance(bg, np.ndarray), 'inputs must be numpy arrays'
    assert x.dtype == np.uint8 and bg.dtype == np.uint8, 'inputs must be uint8 arrays'
    
    # Get image sizes
    x_h, x_w = x.shape[1:]

    # Convert to RGBA images
    im = TF.to_pil_image(torch.ByteTensor(x))
    im = im.convert('RGBA')
    pix = im.load()
    bg = TF.to_pil_image(torch.ByteTensor(bg))
    bg = bg.convert('RGBA')
    bg = bg.load()

    # Replace pixels
    for x in range(x_w):
        for y in range(x_h):
            r, g, b, a = pix[x, y]
            h_ratio, s_ratio, v_ratio = rgb_to_hsv(r / 255., g / 255., b / 255.)
            h, s, v = (h_ratio * 360, s_ratio * 255, v_ratio * 255)

            min_h, min_s, min_v = (100, 80, 70)
            max_h, max_s, max_v = (185, 255, 255)
            if min_h <= h <= max_h and min_s <= s <= max_s and min_v <= v <= max_v:
                pix[x, y] = bg[x, y]

    return np.moveaxis(np.array(im).astype(np.uint8), -1, 0)[:3]

class ColorWrapper(gym.Wrapper):
    """Wrapper for the color experiments"""
    def __init__(self, env, mode, seed=None):
        # assert isinstance(env, FrameStack), 'wrapped env must be a framestack'
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps
        self._mode = mode
        self._random_state = np.random.RandomState(seed)
        self.time_step = 0
        if 'color' in self._mode:
            self._load_colors()

    def reset(self):
        self.time_step = 0
        if 'color' in self._mode:
            self.randomize()
        elif 'video' in self._mode:
            # apply greenscreen
            setting_kwargs = {
                'skybox_rgb': [.2, .8, .2],
                'skybox_rgb2': [.2, .8, .2],
                'skybox_markrgb': [.2, .8, .2]
            }
            if self._mode == 'video_hard':
                setting_kwargs['grid_rgb1'] = [.2, .8, .2]
                setting_kwargs['grid_rgb2'] = [.2, .8, .2]
                setting_kwargs['grid_markrgb'] = [.2, .8, .2]
            self.reload_physics(setting_kwargs)
        return self.env.reset()

    def step(self, action):
        self.time_step += 1
        return self.env.step(action)

    def randomize(self):
        assert 'color' in self._mode, f'can only randomize in color mode, received {self._mode}'
        self.reload_physics(self.get_random_color())

    def _load_colors(self):
        assert self._mode in {'color_easy', 'color_hard'}
        self._colors = torch.load(f'src/env/data/{self._mode}.pt')

    def get_random_color(self):
        assert len(self._colors) >= 100, 'env must include at least 100 colors'
        return self._colors[self._random_state.randint(len(self._colors))]

    def reload_physics(self, setting_kwargs=None, state=None):
        from dm_control.suite import common
        domain_name = self._get_dmc_wrapper()._domain_name
        if setting_kwargs is None:
            setting_kwargs = {}
        if state is None:
            state = self._get_state()
        self._reload_physics(
            *common.settings.get_model_and_assets_from_setting_kwargs(
                domain_name+'.xml', setting_kwargs
            )
        )
        self._set_state(state)
    
    def get_state(self):
        return self._get_state()
    
    def set_state(self, state):
        self._set_state(state)

    def _get_dmc_wrapper(self):
        _env = self.env
        while not isinstance(_env, dmc2gym.wrappers.DMCWrapper) and hasattr(_env, 'env'):
            _env = _env.env
        assert isinstance(_env, dmc2gym.wrappers.DMCWrapper), 'environment is not dmc2gym-wrapped'

        return _env

    def _reload_physics(self, xml_string, assets=None):
        _env = self.env
        while not hasattr(_env, '_physics') and hasattr(_env, 'env'):
            _env = _env.env
        assert hasattr(_env, '_physics'), 'environment does not have physics attribute'
        _env.physics.reload_from_xml_string(xml_string, assets=assets)

    def _get_physics(self):
        _env = self.env
        while not hasattr(_env, '_physics') and hasattr(_env, 'env'):
            _env = _env.env
        assert hasattr(_env, '_physics'), 'environment does not have physics attribute'

        return _env._physics

    def _get_state(self):
        return self._get_physics().get_state()
        
    def _set_state(self, state):
        self._get_physics().set_state(state)

class VideoWrapper(gym.Wrapper):
    """Green screen for video experiments"""
    def __init__(self, env, mode, seed=0):
        gym.Wrapper.__init__(self, env)
        self._mode = mode
        self._seed = seed
        self._random_state = np.random.RandomState(seed)
        self._index = 0
        self._video_paths = []
        if 'video' in mode:
            self._get_video_paths()
        self._num_videos = len(self._video_paths)
        # self._max_episode_steps = env._max_episode_steps

    def _get_video_paths(self):
        video_dir = os.path.join('/home/qyliu/datasets/gb-data', self._mode)
        if 'video_easy' in self._mode:
            self._video_paths = [os.path.join(video_dir, f'video{i}.mp4') for i in range(10)]
        elif 'video_hard' in self._mode:
            self._video_paths = [os.path.join(video_dir, f'video{i}.mp4') for i in range(100)]
        else:
            raise ValueError(f'received unknown mode "{self._mode}"')

    def _load_video(self, video):
        """Load video from provided filepath and return as numpy array"""
        import cv2
        cap = cv2.VideoCapture(video)
        assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) >= 100, 'width must be at least 100 pixels'
        assert cap.get(cv2.CAP_PROP_FRAME_HEIGHT) >= 100, 'height must be at least 100 pixels'
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        buf = np.empty((n, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), np.dtype('uint8'))
        i, ret = 0, True
        while (i < n  and ret):
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buf[i] = frame
            i += 1
        cap.release()
        return np.moveaxis(buf, -1, 1)

    def _reset_video(self):
        self._index = (self._index + 1) % self._num_videos
        self._data = self._load_video(self._video_paths[self._index])
        
    def reset(self):
        if 'video' in self._mode:
            self._reset_video()
        self._current_frame = 0
        time_step = self.env.reset()
        time_step.observation['pixels'] = self._greenscreen(time_step.observation['pixels'].copy())
        return time_step

    def step(self, action):
        self._current_frame += 1
        time_step=self.env.step(action)
        time_step.observation['pixels'] = self._greenscreen(time_step.observation['pixels'].copy())
        return time_step

    def _interpolate_bg(self, bg, size:tuple):
        """Interpolate background to size of observation"""
        bg = torch.from_numpy(bg).float().unsqueeze(0)/255.
        bg = F.interpolate(bg, size=size, mode='bilinear', align_corners=False)
        return (bg*255.).byte().squeeze(0).numpy()

    def _greenscreen(self, obs):
        """Applies greenscreen if video is selected, otherwise does nothing"""
        if 'video' in self._mode:
            bg = self._data[self._current_frame % len(self._data)] # select frame
            obs = np.moveaxis(obs,-1,0)
            bg = self._interpolate_bg(bg, obs.shape[1:]) # scale bg to observation size
            bg_obs = do_green_screen(obs, bg)
            return np.moveaxis(bg_obs,0,-1) # apply greenscreen
        return obs

    def apply_to(self, obs):
        """Applies greenscreen mode of object to observation"""
        obs = obs.copy()
        channels_last = obs.shape[-1] == 3
        if channels_last:
            obs = torch.from_numpy(obs).permute(2,0,1).numpy()
        obs = self._greenscreen(obs)
        if channels_last:
            obs = torch.from_numpy(obs).permute(1,2,0).numpy()
        return obs
    
def _make(
    domain, 
    task,
    gb_mode = '',
    action_repeat=1, 
    height=84,
    width=84,
    pixels_only=True,
    **kwargs
):
    # make sure reward is not visualized
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(
            domain,
            task,
            visualize_reward=False,
            **kwargs
        )
        pixels_key = 'pixels'
        is_manipulation_task = False
    else:
        name = f'{domain}_{task}_vision'
        env = manipulation.load(name, **kwargs)
        pixels_key = 'front_close'
        is_manipulation_task = True
        raise NotImplementedError
    # add wrappers
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    # add renderings for clasical tasks
    if (domain, task) in suite.ALL_TASKS:
        # zoom in camera for quadruped
        camera_id = dict(quadruped=2).get(domain, 0)
        render_kwargs = dict(
            height=height, 
            width=width, 
            camera_id=camera_id
        )
        env = pixels.Wrapper(env,
                             pixels_only=pixels_only,
                             render_kwargs=render_kwargs)
    if 'video' in gb_mode:
        env = VideoWrapper(env, gb_mode)
    env = SimpleFramekWrapper(env)
    env = ExtendedTimeStepWrapper(env)
    return env


DEFAULT_BACKGROUND_PATH = os.path.expanduser('~/datasets/DAVIS/JPEGImages/480p/')
def _make_dcs(
    domain, 
    task,
    dataset_path=DEFAULT_BACKGROUND_PATH,
    dataset_videos='train',
    action_repeat=1, 
    height=84,
    width=84,
    pixels_only=True,
    gb_mode='',
    **kwargs
):
    assert (domain, task) in suite.ALL_TASKS, "%s, %s"%(domain, task)
    render_kwargs = dict(height=height, width=width)
    env = dcs_suite.load(
        domain,
        task,
        background_dataset_path=dataset_path,
        background_dataset_videos=dataset_videos,
        pixels_only=pixels_only,
        render_kwargs=render_kwargs,
        **kwargs
    )
    # add wrappers
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    env = SimpleFramekWrapper(env)
    env = ExtendedTimeStepWrapper(env)
    return env


def make(
    domain, 
    task,
    distracting=True,
    **kwargs
):
    if distracting:
        return _make_dcs(domain, task, **kwargs)
    else:
        return _make(domain, task, **kwargs)
