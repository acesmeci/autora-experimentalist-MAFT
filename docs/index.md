# **Experimentalist Search Method**

## **Conceptual Overview**

**Our approach: Search Method – Stratified Random-Subset Novelty (Stratified-RSN) with fixed ε-greedy exploration**

Let us first describe the concept we tried to develop.

The experimentalist selects new experiment conditions that are *diverse*, *representative*, and *spatially balanced*.  
It combines **stratified coverage weighting**, **ε-greedy exploration**, and **greedy novelty maximization** within random subsets of the design space.

<br>
<br>

### **Search Space**

- **Definition:**  
  The **search space** is defined as the *full grid* of all possible experimental conditions (`reference_conditions`), generated via the Cartesian product of each variable’s discrete values.

- **Role in selection:**  
  - **Candidate pool** = all grid points **minus** those already tested (`conditions`).  
  - This pool forms the set of possible new points to pick from at each discovery cycle.  
  - The grid itself remains **fixed** across all iterations — only the subset of “tested” points grows as experiments are performed.  
  - This ensures consistency: every sampling step occurs in the same global design space, while progress is tracked through which regions have been explored.

<br>
<br>

### **Method**


- **Stratification:**  
  The grid is divided into **bins** (default: 10) for each numeric variable using `pd.cut`.  
  Each bin receives a **weight inversely proportional** to how many tested samples it contains:
  $$
  w_i \propto \frac{1}{n_{\text{tested in bin}_i} + 1}.
  $$
  → This biases sampling toward *under-covered regions*, encouraging uniform exploration.

- **Novelty score:**  
  For each untested candidate \(x\), compute the minimum Euclidean distance to any tested point:
  $$
  \text{novelty}(x) = \min_{s \in S} \|x - s\|_2,
  $$
  where \(S\) is the set of tested conditions.  
  High novelty means the candidate lies far from previously explored areas.

- **Max–min selection (diversity):**  
  1. Pick the candidate with the highest novelty score.  
  2. Add it to the selection set.  
  3. Repeat, each time choosing the candidate that maximizes the *minimum distance* to all tested + newly chosen points.  
  → Ensures that the new batch of experiments is spatially well-distributed.

- **ε-greedy exploration:**  
  With probability **ε** (default = 0.4), the sampler **explores randomly** according to the stratified bin weights.  
  Otherwise, it **exploits** by selecting points via the novelty-based max–min rule.  
  → Maintains controlled stochasticity while avoiding redundant local sampling.

- **Random-subset efficiency:**  
  The novelty search is applied only within a **weighted random subset** of the candidate pool  
  (size ≈ `subset_factor × num_samples`, capped at `subset_cap`) to improve efficiency.

<br>
<br>


## **(2 pts) Inputs: Which inputs does your experiment sampling method consider, and why?**

**Inputs considered:**

- **`conditions`** – the set of experimental conditions already tested in previous cycles.  
- **`reference_conditions`** – the full grid (Cartesian product) of all possible experiment conditions.  
- **`num_samples`** – number of new conditions to select per iteration.  
- **`epsilon`** – probability of random exploration (default = 0.4).  
- **`bins`** – number of bins used for stratified coverage weighting (default = 10).  
- **`subset_factor`** / **`subset_cap`** – control how many candidate points are included in the random subset before novelty selection.  
- **`random_state`** – ensures reproducible sampling.

**Why these inputs?**  
These inputs define the *geometry and diversity structure* of the search space.  
- `conditions` and `reference_conditions` establish what has been explored and what remains.  
- `bins` and weighting promote balanced coverage across regions.  
- `epsilon` and subset parameters manage exploration vs. exploitation and computational cost.  
Together, they allow the experimentalist to make informed, geometry-based selections without relying on model predictions.

<br>
<br>


## **(2 pts) Sampling Method: Which sampling method are you using, and why?**

**Method used:**  
The implemented sampler is a **Stratified Random-Subset Novelty (Stratified-RSN)** method.  
It combines **ε-greedy exploration**, **bin-weighted stratification**, and **greedy max–min novelty selection**.

**How it works:**
1. Divide the search grid into bins and compute coverage weights \(w_i \propto 1/(n_{\text{tested in bin}_i} + 1)\).  
2. With probability ε, **explore** randomly using these weights.  
3. Otherwise, **exploit** by selecting the most novel and spatially diverse points using a greedy max–min criterion:
   $$
   X_{\text{new}} = \arg\max_{x \in \text{subset}} \min_{s \in S} \|x - s\|_2.
   $$
4. To stay efficient, perform this novelty selection within a smaller weighted random subset of the grid.

**Why this method?**  
This approach balances **exploration** (broad coverage and stochastic sampling) and **exploitation** (novel, diverse, information-rich points).  
We chose this method since we wanted to treat it as *"randomness + structure"*, which previously gave the best results in coverage and efficiency.  
- Stratification ensures under-sampled regions aren’t neglected.  
- The novelty metric enforces diversity.  
- The random subset keeps computation efficient.  
Overall, Stratified-RSN achieves consistent, evenly distributed exploration of the experimental space while avoiding redundant or clustered sampling.

<br>

### **Overview over our Configurable Parameters**

| Parameter | Default | Description |
|------------|----------|-------------|
| `epsilon` | 0.4 | Probability of random exploration; fixed for the run. |
| `bins` | 10 | Number of stratification bins per variable. |
| `subset_factor` | 4 | Controls the subset size for novelty selection. |
| `subset_cap` | 200 | Maximum number of candidates per iteration. |
| `random_state` | `None` | Seed for reproducibility. |
| `num_samples` | user-defined | Number of new conditions selected per cycle. |
