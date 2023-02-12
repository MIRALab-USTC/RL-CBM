from typing import Optional, Tuple, Union, List
from collections import OrderedDict
import cv2
from gym.spaces import Box
import numpy as np
from collections import deque

from cbm.environments.base_env import Env
import cbm.external_libs.distracting_control.suite_utils as dcs_utils
from cbm.utils.logger import logger
from cbm.environments.dmc_wrapper import make
from cbm.utils.launch_utils import recursively_update_config
import os 
import imageio

Obs = np.ndarray
Traj = Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]

class DMControlEnv(Env):
    def __init__(
        self, 
        domain: str,
        task: str,
        distracting=None, #None, "none", "camera", "color", "background", "easy", "medium", "hard"
        scale = None,
        dynamic_distracting=False,
        gb_mode = '',
        action_repeat: int = 4,
        image_size: int = 64,
        frame_stack: int = 3,
        obs_before_reset: str = "repeat",
        return_state: bool = False,
        return_physics: bool = False,
        return_traj: bool = False,
        record_video: bool = False,
        render_via_env: bool = True,
        env_render_size: int = 480,
        video_fps: int = 10,
        video_dir: Optional[str] = None,
        video_prefix: str = 'dmc_video_',
        **inner_env_kwargs
    ) -> None:
        #obs_befor_reset: "repeat" or "zero"
        #action and reward: frame_stack - 1 
        pixels_only = not return_state
        self.return_state = return_state
        self.return_physics = return_physics
        self.distracting_mode = distracting
        distracting, temp_kwargs = self._get_dcs_kwargs(
            domain, dynamic_distracting, scale, distracting, 
        )
        recursively_update_config(temp_kwargs, inner_env_kwargs)
        env = make(
            domain=domain,
            task=task,
            distracting=distracting,
            gb_mode = gb_mode,
            height=image_size,
            width=image_size,
            action_repeat=action_repeat,
            pixels_only=pixels_only,
            **temp_kwargs
        )
        self.action_repeat = action_repeat
        self._env = env
        # for frame_stack
        self._k = self.frame_stack = frame_stack
        self._frames = deque([], maxlen=frame_stack)
        self.obs_before_reset = obs_before_reset
        if return_traj:
            self._actions = deque([], maxlen=frame_stack-1)
            self._rewards = deque([], maxlen=frame_stack-1)
        self.return_traj = return_traj
        # construct a new obs space and action space
        wrapped_obs_spec = env.observation_spec()
        f_sh = wrapped_obs_spec['pixels'].shape
        # remove batch dim
        if len(f_sh) == 4:
            f_sh = f_sh[1:]
        self.frame_shape = f_sh
        self.action_shape = env.action_spec().shape
        if not return_traj:
            self.observation_space = Box(
                low=0,
                high=255,
                shape=(f_sh[0] * frame_stack,) + f_sh[1:],
                dtype=wrapped_obs_spec['pixels'].dtype
            )
        else:
            self.observation_space = Box(
                low=0,
                high=255,
                shape=f_sh,
                dtype=wrapped_obs_spec['pixels'].dtype
            )
        self.action_space = Box(
            low= -1,
            high= 1,
            shape=self.action_shape,
            dtype=env.action_spec().dtype
        )
        #return state
        self.return_state = return_state
        self.state_dim_dict = OrderedDict()
        self.state_size = 0
        if return_state:
            for k in wrapped_obs_spec:
                if k == 'pixels':
                    continue
                k_shape = wrapped_obs_spec[k].shape
                if len(k_shape) == 1:
                    d = k_shape[0]
                elif len(k_shape) == 0:
                    d = 1
                else:
                    raise NotImplementedError
                self.state_dim_dict[k] = d
                self.state_size += d
        self.state_shape = (self.state_size,)
        # for video recording
        self.record_video = record_video
        if record_video:
            if video_dir == None:
                self.video_dir = os.path.join(logger._snapshot_dir, 'videos')
            else:
                self.video_dir = video_dir
            self.video_prefix = video_prefix
            os.makedirs(self.video_dir, exist_ok=True)
            self.video_frames = []
            self.video_count = 0
            self.env_render_size = env_render_size
            self.render_via_env = render_via_env
            self.video_fps = video_fps
            self.have_saved_video = False

    def _get_dcs_kwargs(self, domain_name, dynamic, scale, distracting):
        # scale 0.1~1 or 1,2,3...
        dcs_kwargs = {}
        if distracting in [None, "none"]:
            distracting = False
        else:
            if distracting == "camera":
                # scale = dcs_utils.DIFFICULTY_SCALE["hard"]
                dcs_kwargs['camera_kwargs'] = dcs_utils.get_camera_kwargs(
                    domain_name, scale, dynamic)
            elif distracting == "color":
                # scale = dcs_utils.DIFFICULTY_SCALE["hard"]
                dcs_kwargs['color_kwargs'] = dcs_utils.get_color_kwargs(
                    scale, dynamic)
            elif distracting == "background":
                # num_videos = dcs_utils.DIFFICULTY_NUM_VIDEOS["hard"]
                dcs_kwargs["background_kwargs"] = {
                    "num_videos": scale,
                    "dynamic": dynamic}
            elif distracting in ["easy", "medium", "hard"]:
                dcs_kwargs["difficulty"] = distracting
            else:
                print(distracting)
                raise NotImplementedError
            distracting = True
        return distracting, dcs_kwargs

    @property
    def n_env(self):
        return 1
            
    def _recorde_frame(self, frame: np.ndarray) -> None:
        if self.render_via_env:
            if hasattr(self._env, 'physics'):
                frame = self._env.physics.render(
                    height=self.env_render_size,
                    width=self.env_render_size,
                    camera_id=0
                )
            else:
                frame = self._env.render()
        self.video_frames.append(frame)

    def _dump_video(self) -> None:
        if not self.render_via_env:
            video_frames = []
            for frame in self.video_frames:
                video_frames.append(frame[-3:].transpose(1, 2, 0))
        else:
            video_frames = self.video_frames
        video_name = '%s%d.mp4'%(self.video_prefix, self.video_count)
        path = os.path.join(self.video_dir, video_name)
        imageio.mimsave(path, video_frames, fps=self.video_fps)
        logger.log('have save the video: %s' % (video_name))
        self.video_frames = []
        self.video_count += 1

    def _get_obs(self) -> Obs:
        assert len(self._frames) == self._k
        if self._k == 1:
            return self._frames[0]
        else:
            return np.concatenate(list(self._frames), axis=1)

    # for slac-type policy
    def _get_traj(self) -> Traj:
        assert len(self._frames) == self._k
        return {
            "frames": list(self._frames), 
            "actions": list(self._actions),
            "rewars": list(self._rewards) 
        }

    # r: [(1)] or [(1,1)]
    def reset(self) -> Union[Obs, Traj]:
        self.cur_step_id = 0
        time_step = self._env.reset()
        o = time_step.observation['pixels']
        if self.return_state:
            self._state = self._get_state(time_step.observation)
        if self.distracting_mode in ["background", "easy", "medium", "hard"]:
            self._noise_index = self._current_img_index
        if self.record_video:
            self.have_saved_video = False
            self._recorde_frame(o)
        # padding before step
        for _ in range(self._k-1):
            if self.obs_before_reset == "repeat": # drq
                self._frames.append(np.array([o]))
            elif self.obs_before_reset == "zero": # slac
                self._frames.append(np.zeros((1,)+self.frame_shape))
            else:
                raise NotImplementedError
            if self.return_traj:
                self._actions.append(np.zeros((1,)+self.action_shape))
                self._rewards.append(np.zeros((1,1)))
        self._frames.append(np.array([o]))
        if self.return_traj:
            return self._get_traj()
        else:
            return self._get_obs()

    def _get_state(self, timestep_o):
        state = []
        if self.return_state:
            for k in self.state_dim_dict: # position, velocity
                _s = timestep_o[k]
                if len(_s.shape) == 0:
                    _s = _s[None]
                state.append(_s)
        state = np.concatenate(state)
        return state[None]

    def step(
        self, 
        action: np.ndarray
    ) -> Tuple[Union[Obs, Traj], np.ndarray, np.ndarray, dict]:
        self.cur_step_id = self.cur_step_id + 1
        # policy.action_np
        if len(action.shape) > len(self.action_shape): 
            if self.return_traj:
                self._actions.append(action)
            action = action[0]
        # action_space.sample()
        else: 
            if self.return_traj:
                self._actions.append(np.array([action]))
        #  step
        time_step= self._env.step(action)
        o = time_step.observation['pixels']
        r = time_step.reward
        done = 1-time_step.discount
        if self.record_video and (not self.have_saved_video):
            self._recorde_frame(o)
        self._frames.append(np.array([o]))
        r, done = np.array([[r]]), np.array([[done]])
        if self.return_traj:
            self._rewards.append(r)
        if time_step.last() and self.record_video and (not self.have_saved_video):
            self._dump_video()
            self.have_saved_video = True
        info = {}
        if self.distracting_mode in ["background", "easy", "medium", "hard"]:
            info["domain"] = self.cur_video_path
            info["noise_label"] = self._noise_index
            self._noise_index = self._current_img_index
            info["next_noise_label"] = self._noise_index
            info["domain_size"] = self.num_images
        if self.return_state:
            info['state'] = self._state
            self._state = self._get_state(time_step.observation)
        if self.return_physics:
            info['physics'] = self._env.physics.get_state().copy()
            info['next_state'] = self._state
        if self.return_traj:
            return self._get_traj(), r, done, info
        else:
            return self._get_obs(), r, done, info

    def __getattr__(self, name):
        return getattr(self._env, name)

    def get_reward_done_function(self, known=None):
        raise NotImplementedError