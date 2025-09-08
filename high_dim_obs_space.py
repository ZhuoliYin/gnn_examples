from gymnasium import spaces
import numpy as np

# Example: observation contains three matrices of different shapes
self.observation_space = spaces.Dict({
    "state_matrix": spaces.Box(
        low=-np.inf, high=np.inf,
        shape=(5, 5), dtype=np.float32  # 5×5 matrix
    ),
    "feature_matrix": spaces.Box(
        low=-1.0, high=1.0,
        shape=(3, 4), dtype=np.float32  # 3×4 matrix
    ),
    "adjacency_matrix": spaces.Box(
        low=0, high=1,
        shape=(5, 5), dtype=np.int8      # 5×5 binary matrix
    ),
})
