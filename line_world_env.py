#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created on 9/8/25 12:00 AM
@File:line_world_env.py
@Author:Zhuoli Yin
@Contact: yin195@purdue.edu
'''

# my_envs/line_world.py
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class LineWorldEnv(gym.Env):
    """A toy 1D world: start at 0, goal at +10. Actions: -1 or +1 step."""
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None, max_steps: int = 100):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps

        # Observation: current position (scalar float32)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf], dtype=np.float32),
            high=np.array([np.inf], dtype=np.float32),
            dtype=np.float32,
        )

        # Action: move left (-1) or right (+1)
        self.action_space = spaces.Discrete(2)  # 0 -> -1, 1 -> +1

        # Internal state
        self.pos = 0.0
        self.goal = 10.0
        self._step_count = 0

        # Optional: RNG
        self.np_random = None

    def seed(self, seed: int | None = None):
        # Not required in >=0.26, but kept for compatibility
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # New API: must accept seed/options and return (obs, info)
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        self.pos = 0.0
        self._step_count = 0
        obs = np.array([self.pos], dtype=np.float32)
        info = {}
        if self.render_mode == "human":
            self._render_human(obs)
        return obs, info

    def step(self, action: int):
        # Map action to delta
        delta = -1.0 if action == 0 else 1.0
        self.pos += delta
        self._step_count += 1

        # Reward: +1 when at goal, small penalty per step
        at_goal = abs(self.pos - self.goal) < 1e-6
        reward = 1.0 if at_goal else -0.01

        # Termination & truncation
        terminated = at_goal
        truncated = self._step_count >= self.max_steps

        obs = np.array([self.pos], dtype=np.float32)
        info = {}

        if self.render_mode == "human":
            self._render_human(obs)

        return obs, reward, terminated, truncated, info

    # --- Rendering helpers ---
    def render(self):
        if self.render_mode == "rgb_array":
            # Produce a simple 64x256 image showing position & goal
            H, W = 64, 256
            img = np.zeros((H, W, 3), dtype=np.uint8)
            # Map position and goal to pixels in [0, W-1]
            def to_px(x):
                # visualize x in range [-5, 15] for this toy
                return int(np.clip((x + 5) / 20 * (W - 1), 0, W - 1))

            pos_px = to_px(self.pos)
            goal_px = to_px(self.goal)

            img[:, pos_px : pos_px + 2, :] = [0, 200, 255]   # cyan player
            img[:, goal_px : goal_px + 2, :] = [0, 255, 0]    # green goal
            return img

        elif self.render_mode == "human":
            # Human-mode rendering is handled inline in _render_human
            # (no return needed)
            pass

    def _render_human(self, obs):
        # Very simple text render; replace with pygame/matplotlib if desired
        bar = "".join(
            "G" if i == int(self.goal) else ("P" if i == int(round(self.pos)) else "-")
            for i in range(-5, 16)
        )
        print(f"[{bar}] pos={self.pos:.1f}, goal={self.goal:.1f}")

    def close(self):
        # Clean up any viewer/window handles here if you have them
        pass
