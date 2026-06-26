# FedLAW v2 — Paper-Faithful Implementation Design

**Date:** 2026-06-26  
**Paper:** Byzantine-Robust Federated Learning with Learnable Aggregation Weights (ICLR 2026)  
**Scope:** MNIST only. CIFAR-10 / CNN deferred to future work.

---

## Goal

Replace the current `src/fedlaw.py` with a complete, paper-faithful implementation
covering all algorithm fixes, correct data partitioning, and all 5 attack types from
the paper. Configurable for n=20 (local CPU) and n=200 (Colab GPU).

---

## 1. Data Partitioning

### Cao et al. (2021) q-parameter method

**File:** `src/data_partition.py`

Each training example with label `l` is assigned to group `l` with probability `q`,
and to each other group with probability `(1−q)/(L−1)`. L=10 for MNIST.

- 200 clients → 10 groups of 20 clients each (or n/L clients per group for general n)
- Within each group: examples distributed uniformly across all clients in the group
- Parameters: `q ∈ {0.6, 0.9}` (paper uses both)

**Malicious client selection: group-oriented**

The paper selects malicious clients in whole groups (hardest case):
- `n_groups_mal = ceil(n_byz / clients_per_group)`
- Fill those groups entirely with malicious clients first
- If remainder: take from an additional group

**API:**
```python
def cao_partition(
    dataset,           # torchvision MNIST train set
    n_clients: int,    # total clients (20 or 200)
    n_labels: int,     # 10
    q: float,          # 0.6 or 0.9
    batch_size: int,   # 64
    seed: int,
) -> list[DataLoader]
```

Returns `n_clients` DataLoaders in client order.

**Malicious index selection:**
```python
def select_malicious_indices(
    n_clients: int,
    n_byz: int,
    clients_per_group: int,
    seed: int,
) -> list[int]
```

Returns list of Byzantine client indices (group-oriented selection).

---

## 2. Attacks

### 2.1 Attack taxonomy

Two fundamentally different types requiring different pipelines:

| Type | How it works | Pipeline |
|---|---|---|
| **Data poisoning** | Byzantine client trains on poisoned data; submits genuine pseudo-gradient | Poisoned DataLoader fed to normal Client |
| **Gradient manipulation** | Byzantine client trains normally; pseudo-gradient replaced post-collection | Post-collection override |

### 2.2 Data poisoning attacks

**File:** `src/attacks.py` (dataset wrapper classes)

**FlipLabel**: label `l → L−l−1` (for L=10: 0→9, 1→8, 2→7, etc.)

```python
class FlipLabelDataset(Dataset):
    def __init__(self, base_dataset, n_labels=10): ...
    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, self.n_labels - 1 - y
```

**Backdoor**: add 8×8 black square to centre of image + random relabel

```python
class BackdoorDataset(Dataset):
    def __init__(self, base_dataset, trigger_size=8, n_labels=10, seed=0): ...
    def __getitem__(self, idx):
        x, y = self.base[idx]
        x = add_trigger(x, trigger_size)   # set 8×8 centre pixels to 0
        y = self.random_labels[idx]        # pre-sampled random label
        return x, y
```

Byzantine clients with poisoned DataLoaders are built as normal ByzFL `Client` objects.
No special handling in the collection loop — their pseudo-gradients are submitted as-is.

### 2.3 Gradient manipulation attacks

**File:** `src/attacks.py` (callable classes)

Interface: `attack_fn(pseudo_grads: list[np.ndarray], theta: np.ndarray, round_k: int) -> list[np.ndarray]`

**InverseGradient**: negate each Byzantine client's own pseudo-gradient
```
g_byz_j = −g_local_j
```

**GlobalParamAttack**: perturb global parameters, return implied pseudo-gradient
```
ε_j ~ N(ν₁·mean(θ), ν₂·var(θ))  element-wise, ν₁=−5, ν₂=1.5
g_byz_j = −ε_j / α
```

**DoubleAttack**: split Byzantine clients 50/50; temporal activation
```
Group M1 (first 50% of byz): InverseGradient, active from round k≥1
Group M2 (second 50% of byz): GlobalParamAttack, active from round k≥4
Before activation: submit honest pseudo-gradient (no replacement)
```

**LIEAttack** (Little Is Enough): wrap ByzFL's `ALittleIsEnough`
```
b_j = μ_k + z·σ_k  where z = stealth bound (Baruch et al. 2019)
```
Note: LIE operates on the full set of honest pseudo-grads (needs all honest grads as input),
so it must receive them as a batch. All LIE Byzantine clients submit the same vector.

---

## 3. FedLAW v2 Algorithm

**File:** `src/fedlaw_v2.py`

### 3.1 All fixes applied

| Fix | Description |
|---|---|
| Gap 1 (CRITICAL) | Pseudo-gradient `g_i = (θ − ψ_i)/α` from E=3 local SGD epochs |
| Gap 2 (REQUIRED) | Server ℓ2-clip all grads: `C = max_{i∈honest} ‖g_i‖` |
| Gap 3 (CORRECTNESS) | Cap `t = 1/(s−10)` per Table 1; verify `s·t ≥ 1` |
| Loss definition (NOT a gap) | `f_i(θ)` at broadcast point — confirmed by eq. (151) p.60 |
| **w-freeze** (NEW) | w updated only for first 20 rounds; θ updated every round |

### 3.2 Unified client collection

All `n_total` clients are ByzFL `Client` objects. Collection loop:

```
_collect(theta, round_k):
    for each client i:
        client.set_parameters(theta)
        f_i = client.compute_gradients()          # loss at theta
        steps = E * len(client.training_dataloader)
        client.compute_model_update(steps)         # train to psi_i
        psi_i = client.get_flat_parameters()
        g_i = (theta - psi_i) / alpha             # pseudo-gradient

    # Post-collection: gradient attack overrides
    for each gradient-attack Byzantine client j:
        g_j = attack_fn_j([g_j], theta, round_k)[0]

    # LIE needs honest grads
    if any LIE Byzantine clients:
        honest_pseudo_grads = [g_i for i in honest_indices]
        for each LIE client j:
            g_j = lie_attack(honest_pseudo_grads)[0]

    # Impute Byzantine losses
    mean_f_honest = mean([f_i for i in honest_indices])
    for i in byz_indices:
        f_i = mean_f_honest

    return G (n×d), f (n,)
```

### 3.3 Round structure

```
initialise: w = (1/n)·1, theta_0 from Server

for k in range(T):
    theta_k = server.get_flat_parameters()

    # Round A: collect at theta_k
    G, f = _collect(theta_k, k)
    G = _clip(G)                              # Gap 2

    if k < w_freeze_rounds:                   # w-freeze: update w only for first 20 rounds
        # Tentative model step
        theta_tilde = theta_k - alpha * G.T @ w

        # Round B: collect at theta_tilde
        G_tilde, f_tilde = _collect(theta_tilde, k)
        G_tilde = _clip(G_tilde)

        # Weight update (Algorithm 2 line 12)
        cross = G @ G_tilde.T                 # (n, n)
        h = w + alpha*beta * (cross @ w) - beta * f_tilde
        w = project_sparse_capped_simplex(h, s=sparsity, t=cap)

    # Model update (Algorithm 2 lines 14-15)
    theta_{k+1} = theta_k - alpha * G.T @ w
    server.set_parameters(theta_{k+1})
```

### 3.4 Hyperparameters (from paper Table 1 / §I.2)

| Param | Value |
|---|---|
| α (model lr) | 0.01 |
| β | grid 10⁻² – 10⁻⁴; default 10⁻² |
| E (local epochs) | 3 |
| Batch size | 64 |
| s (sparsity) | (1 − frac_mal) × n |
| t (cap) | 1/(s − 10); for small n use 1/(s − 2) |
| w_freeze_rounds | 20 |
| T (total rounds) | 200 (MNIST) |

---

## 4. Configuration

**File:** `configs/fedlaw_v2_mnist.yaml` (paper scale)

```yaml
dataset: mnist
n_clients: 200
q: 0.9                     # heterogeneity
frac_malicious: 0.4        # 40% Byzantine (worst case)
attack: flipping_label     # one of: flipping_label, backdoor, inverse_gradient,
                           #         global_parameter, double, lie
alpha: 0.01
beta: 0.01
E: 3
batch_size: 64
T: 200
w_freeze_rounds: 20
seeds: [0, 1, 2, 3, 4]    # 5 runs (paper reports mean ± std over 5)
eval_every: 10
results_dir: results/v2
```

**File:** `configs/fedlaw_v2_small.yaml` (CPU validation)

```yaml
dataset: mnist
n_clients: 20
q: 0.9
frac_malicious: 0.4        # 8 Byzantine / 12 honest
attack: flipping_label
alpha: 0.01
beta: 0.01
E: 3
batch_size: 64
T: 30
w_freeze_rounds: 20
seeds: [0]
eval_every: 5
results_dir: results/v2_small
```

---

## 5. File Structure

```
src/
  data_partition.py   NEW  Cao et al. q-partitioner + group-oriented byz selection
  attacks.py          NEW  FlipLabelDataset, BackdoorDataset, InverseGradient,
                           GlobalParamAttack, DoubleAttack, LIEAttack
  fedlaw_v2.py        NEW  Unified FedLAW loop (all 4 fixes + w-freeze)
  run_fedlaw_v2.py    NEW  CLI entrypoint (reads YAML, writes results/v2/)
  projections.py      unchanged
  models.py           unchanged
  fedlaw.py           kept for reference (old validated code)

configs/
  fedlaw_v2_mnist.yaml   NEW  paper scale (n=200)
  fedlaw_v2_small.yaml   NEW  CPU validation (n=20)

results/v2/
  <attack>/<q>/<frac_mal>/seed_<N>/
    metrics.csv      round, test_acc, test_loss
    weights.npy      (T+1, n) weight trajectories
    config.yaml      copy of run config
```

---

## 6. Out of scope

- CIFAR-10 / CNN — future work
- Baseline comparisons (Krum, TrMean, Bulyan) — ByzFL has these; add later
- Partial participation — separate milestone
- Dormancy attack — separate milestone
- Honest mitigation — separate milestone

---

## 7. Open implementation risks

1. **ByzFL Client with poisoned DataLoader**: `Client` initialised with
   `"LabelFlipping": False` accepts any DataLoader. Poisoned DataLoader wrapper
   should be transparent to ByzFL. Verify by checking Client's training step
   (it calls `next(iter(dataloader))` — standard PyTorch, no label access beyond
   the batch).

2. **LIE needs honest grads first**: collection loop must separate honest
   pseudo-grad collection before applying LIE override to Byzantine clients.

3. **DoubleAttack temporal logic**: round counter must be passed into attack_fn.
   Before activation rounds, DoubleAttack Byzantine clients submit their own
   honest pseudo-gradient (no override). After round 1: M1 overrides.
   After round 4: M2 overrides.

4. **w-freeze with small n**: with n=20 and s=12 (80% honest, 40% byz),
   s−10=2, t=1/2, s·t=6 ≥ 1 ✓. For n=20 with 40% byz: n_byz=8, n_honest=12,
   s=12, t=1/2. Verify feasibility before each run.

5. **Backdoor trigger on MNIST**: MNIST images are 28×28. An 8×8 trigger in the
   centre occupies pixels [10:18, 10:18]. Black = 0 after normalisation requires
   setting to `(0 − 0.1307) / 0.3081 ≈ −0.424` in normalised space, or applying
   trigger before the normalisation transform.
