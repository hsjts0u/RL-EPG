# Expected Policy Gradient

This repo aims to replicate the result of [Expected Policy Gradient]([https://arxiv.org/abs/1706.05374](https://arxiv.org/abs/1706.05374))   with pytorch.

# Dependencies

- Python
- PyTorch
- OpenAI Gym
- MuJoCo (Warning: MuJoCo is not supported by Apple Silicon)

Install the dependencies:

```bash
pip install -r requirements.txt
```

# Learning Environment

### MuJoCo

- InvertedPendulum-v2
- HalfCheetah-v2
- Reacher-v2
- Walker2d-v2

# How to run

Train model

```bash
python [spg.py|ddpg.py|epg_*.py]
```

Generate a graph with the existing data

```bash
python figure.py
```

Generate a comparison graph of the variation4 and ddpg in HalfCheetah-v2

```bash
python variation4/test_curve.py
```