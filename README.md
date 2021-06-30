# Expected Policy Gradient

This repo aims to replicate the result of [Expected Policy Gradient](https://arxiv.org/abs/1706.05374) with pytorch.

# Dependencies

- Python
- PyTorch (tested on 1.8.1+cpu and 1.9.0+cpu)
- OpenAI Gym
- MuJoCo (Warning: MuJoCo is not supported by Apple Silicon)
- numpy
- numdifftools (only for epg\_rb\_target\_numdifftools.py and epg\_vanilla.py)
- matplotlib and pandas (for graphing)

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
