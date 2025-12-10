# codex_distance_metrics_instructions.md

## 📌 Goal

Implement a standalone Python module `distance_metrics.py` that provides interchangeable **trajectory distance metrics**:

- **Euclidean distance**
- **Dynamic Time Warping (DTW) distance**
- **Discrete Fréchet distance**

The code will be used for clustering aircraft trajectories in a noise-simulation pipeline.

Trajectories are already:

- smoothed  
- converted to UTM  
- resampled to fixed length  
- stored as `numpy.ndarray` with shape **(T, D)** where `D = 2 or 3`

---

# 📁 Requirements for `distance_metrics.py`

## ✔ Allowed Libraries

- **NumPy only**
- No third-party libraries (`tslearn`, `fastdtw`, etc.)

---

## ✔ Trajectory Type Definition

```python
import numpy as np
Trajectory = np.ndarray  # shape (T, D), D=2 or 3
```

---

# 1. Implement **Euclidean Distance**

### Function signature

```python
def euclidean_distance(traj1: Trajectory, traj2: Trajectory) -> float:
    """
    Compute simple Euclidean distance between two trajectories of equal length.

    Distance is defined as:
        || vec(traj1) - vec(traj2) ||_2

    Raises:
        ValueError if shapes do not match or inputs are invalid.
    """
```

### Notes

- Requires **same shape** `(T, D)`
- Flatten both arrays and compute Euclidean norm
- Validate shapes and dimensions

---

# 2. Implement **Dynamic Time Warping (DTW)**

### Function signature

```python
def dtw_distance(traj1: Trajectory, traj2: Trajectory, window_size: int | None = None) -> float:
    """
    Compute DTW distance between two trajectories using DP.

    Local cost: Euclidean distance between points in R^D.

    Args:
        traj1: shape (T1, D)
        traj2: shape (T2, D)
        window_size: optional Sakoe–Chiba band. If provided, only
                     |i - j| <= window_size are allowed.

    Returns:
        DTW distance (float).
    """
```

### Implementation details

- Create DP matrix `dp` shape `(T1+1, T2+1)` initialized with `+inf`
- Set `dp[0, 0] = 0`
- Recurrence:

```
cost = || traj1[i-1] - traj2[j-1] ||
dp[i,j] = cost + min(dp[i-1,j], dp[i,j-1], dp[i-1,j-1])
```

- Skip `(i, j)` if window constraint violated
- Return `dp[T1, T2]`

---

# 3. Implement **Discrete Fréchet Distance**

### Function signature

```python
def discrete_frechet_distance(traj1: Trajectory, traj2: Trajectory) -> float:
    """
    Compute discrete Fréchet distance between two trajectories.

    Uses dp array 'ca' of shape (T1, T2).
    """
```

### Recurrence (0-based indexing)

Let:

```
d(i,j) = Euclidean distance between traj1[i] and traj2[j]
```

Then:

```
ca[0,0] = d(0,0)
ca[i,0] = max(ca[i-1,0], d(i,0))
ca[0,j] = max(ca[0,j-1], d(0,j))
ca[i,j] = max(
                min(ca[i-1,j], ca[i-1,j-1], ca[i,j-1]),
                d(i,j)
            )
```

Return:

```
ca[T1-1, T2-1]
```

---

# 4. Metric Selector

### Function signature

```python
from typing import Callable

DistanceFn = Callable[[Trajectory, Trajectory], float]

def get_trajectory_distance_fn(name: str) -> DistanceFn:
    """
    Supported: 'euclidean', 'dtw', 'frechet'
    Case-insensitive.
    Raises ValueError if unknown.
    """
```

---

# 5. Pairwise Distance Matrix

### Function signature

```python
from collections.abc import Sequence
import numpy as np

def pairwise_distance_matrix(
    trajectories: Sequence[Trajectory],
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Compute symmetric pairwise distance matrix.

    Returns:
        D[i,j] = distance(traj_i, traj_j)

    Requirements:
        - Zero diagonal
        - Symmetric matrix
    """
```

---

# 6. Validation Rules

- All inputs must be 2-D numpy arrays
- D (dimension) must match for both trajectories
- Raise `ValueError` on shape mismatch
- Distance must always be `float`
- Handle NaN detection and error messages

---

# 7. Add a Self-Test Block

```python
if __name__ == "__main__":
    # Create simple test trajectories
    # Example:
    # traj1 = np.array([[0,0],[1,1],[2,2]])
    # traj2 = np.array([[0,0],[1,2],[2,4]])
    # Print Euclidean, DTW, Frechet distances
    pass
```

Purpose:
- Quick manual verification
- Helps ensure correctness when running the module standalone

---

# ✔ Deliverable

Codex should output a complete file:

```
distance_metrics.py
```

Containing:

- Euclidean implementation
- DTW implementation
- Discrete Fréchet implementation
- Metric selector
- Pairwise distance matrix
- Self‑test block

Fully documented and using only NumPy.

