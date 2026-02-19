# A transformer-diffusion chess engine without game rules

**The optimal architecture combines a UniZero-style transformer world model, an Lc0 BT4-inspired policy backbone, and DiffuSearch-style discrete diffusion for implicit search—all unified through a shared latent space and trained via self-play reinforcement learning.** This design is superior because it addresses the fundamental TC⁰ expressivity ceiling of single-pass transformers by adding iterative computational depth through diffusion, while maintaining the MuZero paradigm of learning game dynamics from scratch. The mathematical justification rests on three pillars: the value equivalence principle (learned latent dynamics need only preserve planning-relevant information), the TC⁰ limitation theorem (constant-depth transformers cannot perform tree search), and the computational expressivity of iterative diffusion (T denoising steps provide O(L×T) effective depth, breaking the TC⁰ barrier). What follows is a complete component-by-component architectural specification with mathematical foundations, compatibility analysis, and training pipeline design.

---

## Why single-pass transformers cannot play chess optimally

The entire architectural motivation begins with a circuit complexity result. Standard transformers with fixed depth L and O(log n) precision per neuron are contained in **DLOGTIME-uniform TC⁰**—the class of constant-depth, polynomial-size threshold circuits. This was established by Merrill, Sabharwal, and Smith (2022–2023) and strengthened by Chiang et al. (2024): both average-hard and softmax-attention transformers fall within this class.

TC⁰ contains addition, multiplication, sorting, and many pattern-matching tasks. But it is widely conjectured to be strictly contained in NC¹, meaning it *excludes* Boolean formula evaluation, tree traversal, and—critically—**minimax search**. Evaluating a game tree of depth d requires Ω(d) sequential steps to propagate values from leaves to root. This is fundamentally compositional and recursive, not parallelizable into constant depth. Since chess with optimal play requires evaluating lines 10–30+ moves deep through iterated application of a transition function, a single transformer forward pass can learn strong heuristic evaluations (pattern matching, piece interactions, positional features are all within TC⁰) but **cannot perform the depth of reasoning required for optimal play**.

This creates a clear architectural imperative: the system needs a mechanism for iterative computation. Three candidates exist—looped transformers, autoregressive chain-of-thought generation (as in DeepMind's MAV), and diffusion-based iterative refinement. Diffusion is the most promising for reasons detailed below.

---

## The three-component architecture

The system consists of three major components sharing a common latent space of dimension d_s, plus representation and training infrastructure:

### Component 1: Transformer world model (learned dynamics)

The world model follows the **UniZero paradigm**—a GPT-like causal transformer that processes sequences of (latent state, action) tokens to predict next states, rewards, policies, and values. UniZero (NeurIPS 2024) demonstrated that this approach disentangles latent state from implicit history, outperforms MuZero on long-term dependency tasks, and scales to multi-task settings.

The architecture consists of a representation network h_θ that encodes single-frame observations into latent states (unlike MuZero's frame stacking), and a transformer dynamics network that takes sequences of fused (state, action) tokens with causal masking. Following STORM's insight, each timestep is compressed into a **single token** via MLP embedding of the concatenated state-action pair—this is critical for computational efficiency versus IRIS's multi-token approach. Prediction heads (separate MLPs) output policy π, value v, and reward r from transformer hidden states.

The mathematical justification relies on the **value equivalence principle** (Grimm, Barreto, Singh, NeurIPS 2020–2021). The learned latent dynamics need not reconstruct the full board state—they need only satisfy:

$$T_{\tilde{m}}^{\pi} v = T_{m^*}^{\pi} v, \quad \forall \pi \in \Pi, v \in V$$

where T is the Bellman operator. MuZero minimizes an upper bound on the proper value equivalence loss. This dramatically reduces the representational burden: the latent state only needs to preserve information relevant to value estimation and policy improvement, not pixel-perfect or state-perfect reconstruction. He et al. (2023) showed empirically that MuZero's model is accurate primarily for trajectories under the behavior policy, with errors growing for unlikely action sequences—a limitation partially mitigated by the policy prior in search.

**Why UniZero over alternatives:** DIAMOND uses diffusion-based world modeling directly in pixel space with a U-Net backbone—effective for Atari but computationally prohibitive for chess's long-horizon planning and incompatible with latent-space MCTS-style search. IRIS uses VQ-VAE discrete tokenization plus a GPT transformer, which is elegant but requires multiple tokens per frame and lacks search compatibility. STORM is the closest competitor, but its DreamerV3-style actor-critic training (without search) is less sample-efficient than MCTS-based policy improvement. **UniZero is the only transformer world model that maintains full MCTS compatibility while gaining the expressivity benefits of attention-based dynamics.**

### Component 2: Transformer policy backbone (evaluation and move generation)

The policy and value networks use an **Lc0 BT4-inspired encoder-only transformer** with square-as-token encoding. BT4 represents the state of the art in chess neural network architecture: 15 layers, 1024 embedding dimension, **32 attention heads**, and smolgen dynamic attention biases, totaling 191.3M parameters at 7.6 GFLOPs per position.

**Square-as-token encoding** maps each of the 64 board squares to a separate token. Each token carries piece information (12-dimensional one-hot for piece type per ply, plus castling, en passant, rule-50 features). This encoding has five mathematical advantages over alternatives:

- **Natural factorization of the action space.** Chess moves are (source, destination) pairs. The policy head computes attention between source-token query vectors and destination-token key vectors, producing a structured 64×64 logit matrix rather than a monolithic 4672-dimensional output vector. This bilinear decomposition is both more parameter-efficient and more interpretable.
- **Fixed positional relationships.** Unlike FEN-string tokenization (where a1 and h8 are arbitrarily distant), square tokens have invariant 2D spatial relationships, enabling chess-meaningful positional encodings. The ChessFormer paper (arXiv:2409.12272) demonstrates that Shaw relative position encoding—which learns arbitrary biases per relative position pair—massively outperforms RoPE, which enforces Euclidean decay inappropriate for chess topology (a bishop on a1 relates more to h8 than to a2).
- **Uniform information density.** Each token carries ~log₂(13) ≈ 3.7 bits for piece identity per ply, versus FEN tokenization where different characters carry vastly different amounts of information.
- **O(64²) = O(4096) attention cost per layer is trivially small**, making the quadratic cost of self-attention irrelevant for chess.
- **Interpretability.** Attention maps directly show which squares attend to which, enabling visualization of learned chess concepts.

**Smolgen dynamic attention biases** are the single most impactful architectural innovation for chess transformers. Standard self-attention computes α_{ij} = softmax(q_i·k_j/√d). Smolgen adds a position- and content-dependent bias: α_{ij} = softmax(q_i·k_j/√d + b_{ij} + s_{ij}(X)), where s(X) is generated by compressing the full 64-token board representation into a 256-dimensional vector, then projecting to h×64×64 attention bias matrices through a shared 256×4096 linear layer. This enables **dynamic chess topology**: in closed positions, distant squares have suppressed attention; in open positions, long-range connections are amplified. Visualization of BT4's learned attention maps reveals heads specializing in rook movement patterns, bishop diagonals, knight L-shapes, and king safety zones.

**Mathematical justification for attention over convolution:** Cordonnier et al. (ICLR 2020) proved that multi-head self-attention can express any convolutional layer—attention strictly subsumes convolution. Moreover, a NeurIPS 2022 result shows approximating the self-attention function class with permutation-invariant fully-connected networks requires exponential width: W*(ξ, d, F) = Ω(exp(d)). This exponential lower bound demonstrates that the relational reasoning embedded in self-attention is fundamentally complex and irreplaceable. BT4's empirical results confirm the theory: **270 Elo stronger** than the best convolutional model (T78) with 40% fewer FLOPs.

**AlphaVile's representation lesson** provides an important caveat: Czech et al. (ECAI 2024) showed that improved input features (material counts, pieces giving check, opposite-colored bishop detection) yielded **180 Elo improvement**—far more than any architecture change alone. The design should incorporate these extended features as additional per-token channels.

### Component 3: Discrete diffusion for implicit search

This is the most novel component. Following DiffuSearch (ICLR 2025), the system replaces explicit MCTS with a **discrete diffusion model that imagines future board state trajectories**, using the iterative denoising process as implicit search.

DiffuSearch (Ye et al., 2025, HKUNLP) demonstrated that discrete diffusion can outperform MCTS on chess: **+14% action accuracy over MCTS-enhanced policy, +540 Elo over searchless baseline**, and 30% improvement on puzzle solving. The mechanism works as follows:

1. The model is trained on sequences containing the current board state followed by future board states and actions (extracted from actual games).
2. At inference, future positions are initialized as noise tokens.
3. Through T iterative denoising steps using the D3PM framework (structured transition matrices for categorical data), the model progressively refines random tokens into plausible future game trajectories.
4. The denoised future trajectory conditions the final action prediction—the model "sees" where the game is heading and chooses accordingly.

The mathematical foundation uses **D3PM (Discrete Denoising Diffusion Probabilistic Models)**, which extends continuous diffusion to categorical data through transition matrices Q_t ∈ ℝ^{K×K}:

$$q(x_t | x_{t-1}) = \text{Cat}(x_t; Q_t \cdot x_{t-1})$$

The absorbing-state variant (where tokens transition to a [MASK] token) performed best, creating a direct connection to masked language modeling. The reverse process uses x_0-parameterization: the model predicts clean data p_θ(x_0|x_t), then computes the posterior p_θ(x_{t-1}|x_t) via Bayes' rule.

**Why diffusion over alternatives for search replacement:**

- **MAV (DeepMind, 2024)** achieves 2923 Elo by training a decoder-only transformer to generate linearized minimax trees as text sequences. This is powerful but computationally expensive at inference (generating thousands of tokens for each search tree) and fundamentally autoregressive—left-to-right generation cannot correct earlier decisions. MAV's internal search also requires training on pre-generated Stockfish-annotated search trees, creating a dependency on an existing strong engine.
- **Looped transformers** are theoretically Turing complete (Giannou et al., ICML 2023, proved a 13-layer looped transformer can simulate SUBLEQ programs) and can simulate graph algorithms including BFS and DFS. However, they require careful engineering of the loop mechanism, and practical implementations (LoopFormer, Ouro) have not yet been demonstrated for game-playing. The loop count must be determined before inference, limiting adaptive computation.
- **Diffusion's unique advantages**: (1) Each denoising step applies a full neural network forward pass, providing effective depth O(L×T)—this definitively breaks the TC⁰ barrier. (2) Unlike autoregressive generation, diffusion refines the **entire trajectory simultaneously**, enabling global coherence and bidirectional information flow. (3) The number of denoising steps is adjustable at inference time, providing natural **anytime search**—more steps for critical positions, fewer for obvious moves. (4) DiffuSearch has already been empirically validated on chess specifically.

---

## How the three components interconnect

The system operates in a shared latent space ℝ^{d_s} where d_s is the token embedding dimension (1024 following BT4). The dimensional constraints are:

- **Encoder** h_θ: ℝ^{64×C_obs} → ℝ^{64×d_s} (per-square features to per-square latent tokens)
- **World model** g_θ: ℝ^{64×d_s} × ℝ^{d_a} → ℝ^{64×d_s} × ℝ¹ (latent state + action → next latent state + reward)
- **Policy head**: ℝ^{64×d_s} → ℝ^{64×64} attention logits (source-destination policy)
- **Value head**: ℝ^{64×d_s} → mean pool → ℝ³ (win/draw/loss)
- **Diffusion module**: Operates on ℝ^{H×64×d_s} where H is the imagination horizon, conditioned on current latent state

The **inference pipeline** proceeds: (1) Encode current position into 64 latent tokens. (2) The diffusion module generates T-step denoised future trajectories in latent space, conditioned on the current state. (3) The policy head takes the current latent state enriched with diffusion-derived future context and outputs move probabilities. (4) The value head estimates win/draw/loss probability.

During **self-play training**, the world model enables MCTS-style planning in latent space (following UniZero's approach) to generate high-quality training targets. The diffusion module is trained to replicate and eventually surpass MCTS's search capabilities. This creates a two-phase training strategy where MCTS bootstraps the diffusion component.

**Gradient flow considerations** are critical at the interfaces. When training end-to-end, gradients must flow through K unrolled world-model steps plus T diffusion denoising steps, creating a backpropagation depth of O(K×T×L). Three mitigations are essential: (1) Stop-gradient boundaries between the world model and diffusion module during early training phases. (2) Separate learning rates—slower for the world model, faster for policy and diffusion. (3) Curriculum over diffusion steps, starting with T=1 and gradually increasing.

---

## The self-play training pipeline in detail

Training follows a modified MuZero Reanalyse loop with five concurrent processes:

**Process 1: Self-play actors.** Multiple actors run games using the latest network checkpoint. For each position, MCTS is executed in the world model's latent space (UniZero-style) to generate improved policy targets π_MCTS and value estimates v_MCTS. Actions are sampled from the visit count distribution with temperature: p(a) ∝ N(s,a)^{1/τ}. Completed games (observations, actions, MCTS policies, game outcomes) are stored in the replay buffer. Temperature scheduling provides exploration→exploitation transition.

**Process 2: Reanalyse actors.** Sample old trajectories from the replay buffer and re-run MCTS using the latest network, generating fresh policy and value targets without new environment interaction. This is MuZero Reanalyse—the most impactful technique for sample efficiency. Reanalysed data is fed to the learner indistinguishably from fresh data.

**Process 3: Diffusion data generation.** Extract trajectory segments from the replay buffer to create diffusion training pairs: (current state, future state sequence). The future states serve as clean targets; the D3PM forward process corrupts them into training inputs at various noise levels.

**Process 4: Network training (learner).** The total loss function combines six terms:

$$\mathcal{L}_{\text{total}} = \lambda_p \mathcal{L}_{\text{policy}} + \lambda_v \mathcal{L}_{\text{value}} + \lambda_c \mathcal{L}_{\text{consistency}} + \lambda_d \mathcal{L}_{\text{diffusion}} + \lambda_r \mathcal{L}_{\text{reward}} + \lambda_{\text{reg}} \|\theta\|^2$$

The **policy loss** is cross-entropy between the MCTS visit count distribution and the network's policy logits: $\mathcal{L}_p = -\sum_a \pi_{\text{MCTS}}(a) \log p_\theta(a)$. The **value loss** uses categorical cross-entropy with distributional representation (scalar values mapped to distributions over discrete support, following MuZero's approach, which provides stronger gradients than MSE). The target is the game outcome for board games (γ=1, Monte Carlo return). The **consistency loss** follows EfficientZero's SimSiam-style formulation: $\mathcal{L}_c = -\text{cos\_sim}(\text{proj}(g_\theta(s_t, a_t)), \text{sg}(\text{proj}(h_\theta(o_{t+1}))))$—this was empirically EfficientZero's most impactful contribution, preventing latent state collapse. The **diffusion loss** is the D3PM variational lower bound plus auxiliary cross-entropy: $\mathcal{L}_d = \mathcal{L}_{\text{VLB}} + 0.001 \cdot \mathcal{L}_{\text{aux}}$. The **reward loss** is categorical cross-entropy on predicted rewards (minimal for chess where rewards are sparse).

Loss balancing uses **HarmonyDream** (Ma et al., ICML 2024), which dynamically adjusts coefficients to maintain equilibrium between losses of different dimensionalities—this is critical because the diffusion loss (high-dimensional trajectory reconstruction) would otherwise dominate the 1D reward and 3D value losses. HarmonyDream achieved **10–69% performance improvements** on visual RL tasks through automatic rebalancing.

**Process 5: Replay buffer management.** The buffer stores complete game trajectories with uniform sampling for board games (MuZero's design). EfficientZero V2's **priority precalculation** warms up priorities for new trajectories using the current model's Bellman error. Staleness is managed through reanalyse (refreshing targets) and explicit staleness thresholds beyond which data is down-weighted.

### Two-phase training schedule

**Phase 1 (Bootstrap):** Train the world model and policy using standard MCTS-based self-play, following UniZero's approach exactly. The diffusion module trains in parallel on trajectory data but does not influence self-play decisions. This phase establishes a strong world model and policy foundation. The world model's transformer backbone processes trajectory sequences with causal masking, jointly predicting latent dynamics and decision quantities.

**Phase 2 (Diffusion integration):** Gradually replace MCTS with the diffusion module for action selection during self-play. The diffusion module's denoising process now generates future trajectory imaginations that condition the policy head. MCTS targets are replaced with diffusion-refined targets. The transition is gradual: a mixing parameter α interpolates between MCTS and diffusion action selection, annealing from 0 (pure MCTS) to 1 (pure diffusion) over training.

This phased approach avoids the instability of jointly training all components from scratch—a known challenge documented in DiWA (2025) and DAWM (2025), where backpropagation through denoising chains is "often unstable and computationally expensive."

---

## The diffusion transformer as both world model and search mechanism

A compelling design variant uses a single **Diffusion Transformer (DiT)** architecture to serve dual roles. The DiT architecture (Peebles & Xie, ICCV 2023) replaces U-Net backbones with vision transformers, using **AdaLN-Zero conditioning**: timestep and condition information generate six modulation parameters (γ₁, β₁, α₁, γ₂, β₂, α₂) per block that scale, shift, and gate the layer normalization, attention, and FFN outputs. Zero-initialization of projection layers ensures each block initially acts as identity, providing stable training.

For chess, the DiT adaptation works as follows. Input tokens are the 64 square tokens of the current position (or a noisy version of a future position). The conditioning signal is the current latent state plus the diffusion timestep. The model learns two tasks simultaneously: (1) predicting the next board state given current state and action (world model role), and (2) denoising corrupted future trajectories back to plausible game continuations (search role). The AdaLN-Zero mechanism naturally separates these tasks through different conditioning—the timestep signal tells the network whether it is doing one-step dynamics prediction (t=0) or multi-step trajectory denoising (t>0).

This unification is mathematically coherent because both tasks share the same underlying computation: predicting chess positions given partial or noisy information about game trajectories. The world model is simply the T=1 special case of the diffusion process. DIAMOND (NeurIPS 2024 Spotlight) demonstrated that diffusion world models achieve state-of-the-art results on Atari (mean HNS 1.46, outperforming human on 11/26 games) using only **4.4M parameters**, validating the approach for game environments.

---

## Representation engineering decisions that cascade through the system

AlphaVile's finding that input representation yields **180 Elo** improvement—far exceeding any architecture change—demands careful representation engineering. The input feature set should include:

- **Standard features**: 12-dimensional piece one-hot per square for current and past 7 positions (history), castling rights, en passant, rule-50 counter, repetition flags
- **Extended features (AlphaVile FX)**: Material count and difference features, pieces giving check, opposite-colored bishop indicator
- **Global embedding (BT3/BT4)**: The full 12×64 board representation is flattened and projected per-square to C additional channels, followed by an FFN—this "global embedding" allows encoding entire board context from layer 0, providing +15% effective size at only +5% latency
- **Win-Draw-Loss-Ply (WDLP) value head**: Predict not just WDL probability but also expected game length, which provides auxiliary gradient signal

The **positional encoding** choice is critical. Shaw et al.'s relative position encoding—which adds learned bias vectors a^Q_{ij} and a^K_{ij} to attention logits based on the relative position (Δrank, Δfile) between squares—outperforms all alternatives. RoPE enforces monotonic decay with 1D distance, which is "especially deleterious" for chess (the ChessFormer paper's words) since a bishop on a1 should maximally attend to h8 despite maximal Euclidean distance. The newer **Geometric Attention Bias (GAB)** from the ICLR 2026 Chessformer submission offers an even better option: it compresses the board state into a global vector via learned projections, then generates per-head h×64×64 bias matrices—essentially a generalization of smolgen with stronger empirical results and claimed adoption in "a leading open-source engine."

---

## Complete module specifications

The following table summarizes the architectural specification for each module:

| Module | Architecture | Input | Output | Parameters |
|--------|-------------|-------|--------|------------|
| Encoder h_θ | Linear + Mish + scale/shift per token | 64 × C_obs features | 64 × 1024 latent tokens | ~5M |
| Policy backbone | 15-layer encoder transformer, 32 heads, smolgen/GAB | 64 × 1024 tokens | 64 × 1024 encoded tokens | ~150M |
| Policy head | Source-destination attention | 64 × 1024 tokens | 64 × 64 move logits | ~2M |
| Value head | Mean pool + LayerNorm + Linear | 64 × 1024 → 1024 | 3 (WDL) | ~3K |
| World model dynamics | 12-layer causal transformer | Sequence of (state, action) tokens | Next state, reward | ~100M |
| Diffusion module | 8-layer DiT with AdaLN-Zero | 64 × 1024 noisy future tokens + condition | 64 × 1024 denoised tokens | ~80M |
| Consistency projector | 2-layer MLP (SimSiam) | 64 × 1024 | 256 | ~1M |

Total system: approximately **340M parameters**, comparable to BT4 (191M) plus world model and diffusion overhead.

---

## Conclusion: why this architecture is the right bet

Three converging research trajectories make this design timely. First, the TC⁰ limitation is now rigorously established—no amount of scaling a standard transformer will overcome the fundamental depth-of-reasoning ceiling, making iterative computation mechanisms mandatory for approaching optimal play. Second, DiffuSearch has demonstrated that discrete diffusion can outperform MCTS on chess while being more parallelizable on modern hardware (all denoising steps share the same network, unlike MCTS which requires serial tree traversal). Third, UniZero and Lc0 BT4 independently validated that transformer-based world models and transformer-based position evaluation are both superior to their CNN predecessors, with the gap widening as model scale increases.

The key architectural insight is **separation of timescales**: the policy backbone captures fast, pattern-matching evaluation (within TC⁰) while the diffusion module provides slow, deliberate search (breaking TC⁰). The world model connects these by providing a learned physics engine for chess dynamics in latent space. The phased training pipeline—bootstrapping with MCTS, then transitioning to diffusion—provides a smooth path from established techniques (MuZero self-play) to the frontier (implicit search via diffusion). No game rules are hardcoded anywhere: the encoder learns to represent positions, the world model learns to predict transitions, the policy learns to select moves, and the diffusion module learns to search—all from self-play data alone.