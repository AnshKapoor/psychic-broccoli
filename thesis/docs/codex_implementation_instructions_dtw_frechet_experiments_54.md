# Codex implementation instructions: DTW & Fréchet for flight-trajectory clustering

This file defines **implementation-ready instructions** for adding **DTW** and **discrete Fréchet** distance into your clustering pipeline **without** accidentally triggering an \(\mathcal{O}(N^2)\) full distance matrix.

**Global rule:** never compute all-pairs distances. Use **candidate edges** (kNN / gating) and compute elastic distances only on those edges.

---

## Shared prerequisites (applies to all experiments below)

### Coordinate system (must be metric)
- Convert latitude/longitude to a **metric** coordinate system before any Euclidean-type cost:
  - Preferred: **UTM** or local **ENU** (meters).

### Trajectory representation
- Each trajectory \(T\) is an ordered sequence of points:
  \[
  T = \{(x_1,y_1[,z_1]),\dots,(x_n,y_n[,z_n])\}
  \]
- Store both versions:
  1) **Fixed-length resample**: \(n=40\) points (for embeddings and banded DTW).
  2) **Simplified polyline** (RDP): variable length (for Fréchet speedups).

### Preprocessing
- Remove duplicates and NaNs.
- Optional smoothing: median filter over \(x,y\) indices (small window).
- Ensure monotonic time ordering.

### Candidate generation (mandatory)
Create a function that returns candidate neighbor indices for each trajectory:
- Input: embeddings \(Z\in\mathbb{R}^{N\times d}\).
- Output: neighbor list \(\mathcal{N}(i)\) of size \(k\) (e.g., 30–100).

This guarantees distance evaluations are \(\mathcal{O}(Nk)\) not \(\mathcal{O}(N^2)\).

---

# Experiment 54 — DTW distance (banded + pruning) on kNN edges

## Goal
Implement **2D DTW** (optionally 3D) with:
- **Sakoe–Chiba band** (window \(w\))
- optional **lower-bound pruning** (LB_Keogh)
- optional **early abandoning** using a threshold \(\tau\)
- parallel execution over kNN edges

## Mathematical definition
Given two equal-length sequences \(A=(a_1,\dots,a_n)\), \(B=(b_1,\dots,b_n)\) with \(a_i\in\mathbb{R}^m\):
- Point cost:
  \[
  c(i,j)=\lVert a_i-b_j\rVert_2
  \]
- DTW recurrence:
  \[
  D(i,j)=c(i,j)+\min\{D(i-1,j),D(i,j-1),D(i-1,j-1)\}
  \]
- Banded constraint:
  \[
  |i-j|\le w
  \]
- Return \(\text{DTW}(A,B)=D(n,n)\).

## Complexity target
- Per pair: \(\mathcal{O}(nw)\) time, \(\mathcal{O}(n)\) memory.
- Total: \(\mathcal{O}(N\,k\,n\,w)\).

## Implementation instructions

### 54.1 Function signature
Implement:
- `dtw_banded(A: ndarray[n,m], B: ndarray[n,m], w: int, tau: float | None) -> float`

Requirements:
- Use **rolling arrays** (two rows) to avoid \(n^2\) memory.
- Only iterate \(j\) in \([i-w, i+w]\).
- If `tau` is given, do **early abandoning**: if the minimum value in current row exceeds `tau`, return `inf`.

### 54.2 Distance cost
Use squared Euclidean inside DP (faster):
\[
\lVert a-b\rVert_2^2=(a_x-b_x)^2+(a_y-b_y)^2(+ (a_z-b_z)^2)
\]
Return `sqrt(final_cost)` only at the end (optional).

### 54.3 Lower-bound pruning (optional but recommended)
Implement `lb_keogh(A,B,r)` for equal-length sequences:
- For each index \(i\), build envelope of \(B\):
  \[
  U_i=\max_{j\in[i-r,i+r]} B_j,\quad L_i=\min_{j\in[i-r,i+r]} B_j
  \]
- Lower bound:
  \[
  LB(A,B)=\sum_i \begin{cases}
  (A_i-U_i)^2 & A_i>U_i\\
  (L_i-A_i)^2 & A_i<L_i\\
  0 & \text{else}
  \end{cases}
  \]
Use it as:
- If `LB(A,B) > tau^2` then skip DTW and return `inf`.

### 54.4 Edge list construction
Input trajectories are huge, so compute DTW only on candidate edges.

Implement:
- `build_knn_edges(Z, k) -> list[tuple[i,j]]`
  - Use approximate kNN (FAISS/Annoy) or sklearn NearestNeighbors.
  - Return directed edges `(i,j)`; later symmetrize if needed.

### 54.5 Parallelization
- Use multiprocessing / joblib to evaluate DTW on edges.
- Chunk edges by batches of size ~10k for overhead control.

### 54.6 Output format
Return a sparse graph:
- `edges = [(i,j,dist_ij), ...]` with `dist_ij < inf`.
- Optionally store as CSR sparse matrix.

### 54.7 Clustering consumption
- For **HDBSCAN**: can accept a sparse precomputed distance matrix.
- For **DBSCAN/OPTICS**: prefer embedding-space clustering first, or custom neighbor queries using the edge list.

## Default parameters to start
- \(n=40\)
- \(w\in\{6,8,10\}\)
- \(k\in\{30,50\}\)
- Optional \(\tau\): choose from a quantile of embedding distances or pilot DTW sample.

## Validation checklist
- Symmetry check: \(d(i,j)\approx d(j,i)\) (if implemented symmetrically)
- Sanity: identical trajectories ⇒ distance ≈ 0
- Runtime: DTW evaluations should be roughly \(N\cdot k\), not \(N^2\).

---

# Experiment 55 — Discrete Fréchet distance (polyline) with simplification

## Goal
Implement **discrete Fréchet distance** for 2D polylines and make it usable via:
- **RDP simplification** (reduce points)
- candidate edges (kNN / gating)
- optional bounding-box pruning

## Mathematical definition (discrete)
Given polylines \(P=(p_1,\dots,p_n)\), \(Q=(q_1,\dots,q_m)\):
- Recurrence:
  \[
  F(i,j)=\max\Big(\lVert p_i-q_j\rVert_2,\min\{F(i-1,j),F(i-1,j-1),F(i,j-1)\}\Big)
  \]
- Result: \(\delta_F(P,Q)=F(n,m)\).

## Complexity target
- Per pair: \(\mathcal{O}(nm)\) time.
- Keep \(n,m\) small using RDP (e.g., 15–30 points typical).

## Implementation instructions

### 55.1 Function signature
- `frechet_discrete(P: ndarray[n,m], Q: ndarray[m,m], tau: float | None) -> float`

Implementation notes:
- Use DP with rolling rows (memory \(\mathcal{O}(\min(n,m))\)).
- Use squared distances internally, take sqrt at the end.
- Early abandoning: if partial DP values already exceed `tau`, return `inf`.

### 55.2 Simplification (mandatory)
Implement RDP simplification on each trajectory polyline before Fréchet:
- `rdp(points, epsilon_meters) -> simplified_points`

Defaults to start:
- `epsilon_meters` in {25, 50, 100} depending on noise level.

### 55.3 Candidate edges
Reuse the same kNN candidate list as Experiment 54:
- Compute Fréchet only on `(i,j)` edges.

### 55.4 Cheap pruning (optional)
Before running Fréchet on a pair:
- Compare bounding boxes or start/end distances; if already > `tau`, skip.

### 55.5 Output format
Same as Experiment 54:
- `edges = [(i,j,dist_ij), ...]` sparse distance graph.

## Default parameters to start
- RDP epsilon: 50 m
- kNN k: 30
- tau: estimated from pilot sample

## Validation checklist
- Identical polylines ⇒ distance ≈ 0
- RDP monotonic: simplifying should not *increase* Fréchet too wildly (spot check)

---

## Shared unit tests (run on a tiny sample)

### Test A: identity
- Generate one trajectory \(T\) and compare to itself.
- Expect \(\text{DTW}(T,T)=0\), \(\delta_F(T,T)=0\).

### Test B: translation
- Let \(T' = T + (\Delta x,\Delta y)\).
- Expect distances scale roughly with \(\lVert(\Delta x,\Delta y)\rVert\).

### Test C: permutation break
- Reverse one sequence order.
- Expect large distance.

---

## Integration notes (how Codex should wire it into your pipeline)

### Data API
Assume you already have a per-flow dataset:
- `traj_ids: list[str]`
- `traj_xy_fixed: ndarray[N, n=40, 2]`
- `traj_xy_rdp: list[ndarray[mi,2]]`

### Step order
1) Build embedding \(Z\) from fixed trajectories (flatten or feature-engineered).
2) Build kNN neighbor lists \(\mathcal{N}(i)\).
3) Compute DTW edges (Exp 54) **or** Fréchet edges (Exp 55) on those candidates.
4) Cluster using:
   - HDBSCAN with sparse precomputed distances, or
   - embedding-based clustering + local elastic refinement.

### Logging
For each run, log:
- \(N\), \(k\), \(n\), \(w\), RDP \(\epsilon\)
- number of candidate edges \(|E|\approx Nk\)
- number of evaluated edges (after pruning)
- runtime per 10k edges

---

## Parameter sweep templates (do **not** run yet)

### DTW sweep
- \(w\in\{6,8,10\}\), \(k\in\{30,50\}\)

### Fréchet sweep
- RDP \(\epsilon\in\{25,50,100\}\), \(k\in\{30,50\}\)

---

## Notes on feasibility for your dataset sizes
- If you keep \(|E|\approx Nk\) with \(k\le 50\), the compute is proportional to \(N\), not \(N^2\).
- DTW is typically cheaper than Fréchet for the same point counts when using a tight band \(w\).
- Fréchet becomes reasonable only after strong simplification (RDP) and pruning.

