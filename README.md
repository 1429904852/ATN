# ATN
ATN is the proposed model in 《[Attention Transfer Network for Aspect-level Sentiment Classification](https://arxiv.org/pdf/2010.12156.pdf)》, which is accepted by Coling'2020.

# Dependencies

```bash
python==3.5
numpy==1.14.2
tensorflow==1.9
```
# Quick Start

### Step1: pretrained (skip this step)
- train
```bash
python pre_train.py
```
- eval (get attention scores)
```bash
python pre_train_eval.py
```
### step2: transfer
- Attention Guide

```bash
sh run_atn_guide.sh
```

- Attention Fusion
```bash
sh run_atn_fusion.sh
```
# Cite
```bash
@article{zhao2020attention,
  title={Attention Transfer Network for Aspect-level Sentiment Classification},
  author={Zhao, Fei and Wu, Zhen and Dai, Xinyu},
  journal={arXiv preprint arXiv:2010.12156},
  year={2020}
}
```
if you have any questions, please contact me zhaof@smail.nju.edu.cn.
