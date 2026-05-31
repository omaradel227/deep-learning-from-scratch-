# Deep Learning — Hands-On Projects
> Built from scratch in PyTorch. One project per architecture, understand WHY not just HOW.

---

## Projects Summary

| # | Architecture | Dataset | Task | Result |
|---|---|---|---|---|
| 1 | DNN Classification | Mushroom (3.1M rows) | Binary Classification | 99%, AUC 0.997 |
| 2 | DNN Regression | Used Car Prices (188k) | Regression | R²=0.64 (log scale) |
| 3 | CNN | CIFAR-10 (60k images) | Image Classification | 85.76% |
| 4 | Transfer Learning (ResNet18) | CIFAR-10 | Image Classification | 92.11% |
| 5 | RNN / LSTM / BiLSTM | IMDB (50k reviews) | Sentiment Analysis | BiLSTM 88.48% |
| 6 | Transformer (from scratch) | IMDB | Sentiment Analysis | 86.59% |
| 7 | BERT fine-tuning | IMDB | Sentiment Analysis | 92.58% |
| 8 | VAE | MNIST (70k images) | Image Generation | Smooth latent space |
| 9 | GAN (DCGAN) | MNIST | Image Generation | Nash equilibrium ep 20 |
| 10 | DQN | CartPole-v1 | Reinforcement Learning | Solved ep 1267 |
| 11 | PPO | CartPole-v1 | Reinforcement Learning | Solved ep ~750 |

---

## PyTorch Fundamentals

```python
# Always:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.train()   # before training — enables Dropout, BatchNorm batch stats
model.eval()    # before eval — disables Dropout, uses running stats

# Training loop:
optimizer.zero_grad()   # clear gradients (PyTorch accumulates by default)
loss.backward()         # backpropagation — calculate gradients
optimizer.step()        # update weights

# Evaluation:
with torch.no_grad():   # no gradients needed → saves memory
    predictions = model(X)
```

**autograd** — tracks all operations, calculates gradients automatically
**zero_grad** — must call before each batch or gradients accumulate
**no_grad** — disables tracking during evaluation, saves memory

---

## Key Components

### Activations
```
ReLU     : max(0, x) — hidden layers, introduces non-linearity
LeakyReLU: max(0.2x, x) — GANs, prevents dead neurons
Sigmoid  : 0-1 — binary output (or use BCEWithLogitsLoss without it)
Tanh     : -1 to 1 — GAN generator output
Softmax  : multiclass probabilities (CrossEntropyLoss applies internally)
```

### Loss Functions
```
BCEWithLogitsLoss : binary classification, expects raw logits
CrossEntropyLoss  : multiclass, applies softmax internally
MSELoss           : regression, sensitive to outliers
HuberLoss         : regression with outliers, MSE+MAE hybrid
BCELoss           : binary, expects probabilities (after sigmoid)
```

### Normalization
```
BatchNorm1d/2d : normalize across batch — faster training, acts as regularizer
LayerNorm      : normalize within each sample — preferred for NLP/Transformers
```

### Regularization
```
Dropout   : zeros random neurons — use during train, disable during eval
Dropout2d : zeros entire feature maps — better for CNNs
Weight decay (AdamW): L2 penalty on weights
```

### Optimizers
```
Adam  : adaptive lr, fast convergence, good default (lr=0.001)
AdamW : Adam + correct weight decay — use for BERT fine-tuning (lr=2e-5)
betas=(0.5, 0.999) instead of default (0.9, 0.999) for GANs
```

### Schedulers
```
ReduceLROnPlateau : reduce lr when metric stops improving
LinearWarmup      : gradually increase lr then decay — BERT fine-tuning
```

---

## Project Lessons

### DNN (Projects 1-2)
- input_size always = X_train.shape[1] — never hardcode
- 3-way split: train/val/test — val monitors training, test for final eval
- BatchNorm val loss can be lower than train (running stats effect) — not a bug
- Target transform: log1p when skewness > 1.0, reverse with expm1
- Target encoding: encode AFTER split — leakage otherwise
- Double log transform bug: verify y range after transform
- HuberLoss for regression with outliers, MSELoss for clean targets
- squeeze() — removes extra dim: [batch, 1] → [batch]

### CNN (Projects 3-4)
- Conv2d output: input_size - kernel_size + 2×padding + 1
- padding=1 with kernel=3 preserves spatial dimensions
- Filters are learned automatically — you only choose HOW MANY
- MaxPool(2,2): halves spatial dimensions, adds translation invariance
- BatchNorm2d for conv layers, Dropout2d zeros entire feature maps
- permute(1,2,0): PyTorch [C,H,W] → matplotlib [H,W,C]
- Data augmentation: train only, never test — adds 2%+ accuracy
- Transfer Learning: fine-tuning >> feature extraction for similar domains
- Differential learning rates: pretrained layers lr×0.1, new layers full lr
- ResNet skip connections: solve vanishing gradient → enables deep training
- Modify ResNet for small images: conv1 kernel 7→3, remove maxpool

### RNN/LSTM (Project 5)
- Pad at START not end — RNN reads content last, remembers better
- Gradient clipping: clip_grad_norm_(params, 1.0) — always with RNNs
- LSTM gates solve vanishing gradient: 76.92% → 88.13% (+11%)
- BiLSTM marginal gain (+0.35%) — reads both directions
- nn.LSTM returns (output, (hidden, cell)) vs nn.RNN returns (output, hidden)
- batch_first=True: [batch, seq, features] — more intuitive

### Transformer (Project 6)
- Attention: Q=query, K=key, V=value — search engine analogy
- Temperature = √d_k — scales dot products before softmax
- Multi-head: 8 parallel attention heads, each learns different patterns
- register_buffer: saves tensor with model, not a trainable parameter
- Positional encoding: sine/cosine waves give each position unique pattern
- LayerNorm preferred over BatchNorm for NLP

### BERT (Project 7)
- Encoder-only: bidirectional, understanding tasks
- Decoder-only (GPT): left-to-right, generation tasks
- Encoder-Decoder (T5): translation, summarization
- [CLS]=101 always first, [SEP]=102 always last
- attention_mask: 1=real token, 0=padding
- AdamW + linear warmup + lr=2e-5 + 3 epochs = standard fine-tuning recipe
- 86.59% (Transformer scratch) → 92.58% (BERT) with same data

### VAE (Project 8)
- Reparameterization trick: z = μ + σ×ε — makes sampling differentiable
- Loss = Reconstruction (BCE) + KL Divergence
- KL forces smooth latent space → can interpolate between digits
- reduction='sum' → loss in hundreds, reduction='mean' → loss ~0.1 (same model)
- Use same reduction for reconstruction and KL — mixing causes imbalance
- VAE=stable+blurry, GAN=sharp+unstable

### GAN (Project 9)
- Two separate optimizers, one per network
- .detach() when training D — stops gradients reaching G
- No .detach() when training G — gradients must flow through D to G
- D Loss ≈ 1.386 at equilibrium (-log(0.5)-log(0.5)) — D guessing 50/50
- betas=(0.5, 0.999) for GANs — lower momentum, reacts faster
- Normalize to [-1,1] not [0,1] — matches Tanh output
- Fixed noise for visualization — tracks Generator progress across epochs
- Mode collapse: Generator produces same image — add noise, use minibatch discrimination

### DQN (Project 10)
- Q(s,a) = expected future reward taking action a in state s
- Experience replay: breaks correlation between consecutive states
- Target network: stable Q-value targets — update every N episodes
- Epsilon-greedy: start ε=1.0 (explore), decay to 0.01 (exploit)
- .gather(1, actions): picks Q-value for the action actually taken
- Plateau then sudden jump — typical DQN learning curve
- Solved CartPole at episode 1267

### PPO (Project 11)
- Policy-based: learns π(action|state) directly vs DQN's Q-values
- Works for continuous AND discrete actions (DQN discrete only)
- Clipped ratio: clip(π_new/π_old, 1-ε, 1+ε) — policy can't change too much
- GAE (λ=0.95): better advantage estimation, reduces variance
- Entropy bonus: encourages exploration without epsilon-greedy
- Rollout-based >> episode-based: collect 2048 steps before updating
- Actor-Critic: Actor=what to do, Critic=how good is this state
- Solved CartPole at episode ~750 (faster than DQN)

---

## Architecture Comparison

### NLP
```
RNN      : vanishing gradient, forgets long sequences
LSTM     : gates solve vanishing gradient, standard for sequences
BiLSTM   : reads both directions, marginal gain for sentiment
Transformer: parallel, attention, needs large data to shine
BERT     : pretrained Transformer, fine-tune in 3 epochs → best results
```

### Computer Vision
```
CNN from scratch    : good baseline, needs augmentation
Transfer Learning   : pretrained features → 6%+ improvement
Feature Extraction  : freeze all → only works if domain very similar
Fine-tuning         : unfreeze all → almost always better
```

### Generative
```
VAE : stable training, blurry outputs, smooth latent space
GAN : unstable training, sharp outputs, Nash equilibrium
```

### Reinforcement Learning
```
DQN : value-based, replay buffer, epsilon-greedy, discrete actions
PPO : policy-based, rollout buffer, entropy bonus, discrete+continuous
```

---

## Common Bugs Fixed
```
1. Forgetting model.train()/model.eval() → wrong Dropout/BatchNorm behavior
2. Not calling zero_grad() → accumulated gradients → wrong updates
3. Hardcoding input_size → breaks when features change
4. Double log transform → verify y range after transform
5. Target encoding before split → data leakage
6. .detach() missing in GAN D training → G updated incorrectly
7. squeeze() missing → shape mismatch in loss calculation
8. padding at end not start → RNN forgets content
9. Episode-based PPO → use rollout-based for proper sample efficiency
10. Feature extraction with modified frozen layers → random weights frozen
```
