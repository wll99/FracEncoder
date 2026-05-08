# FracEncoder: Towards Adaptive Cognitive Trajectories via Fractional-Order Context Encoding

This repository contains the code for our NeurIPS 2026 submission, FracEncoder: Towards Adaptive Cognitive Trajectories via Fractional-Order Context Encoding.

## Reproducing Results

To run our code, go to the project root.

```
python -m judge_sac.trainer
--env                            PendulumP, PendulumV, CartPoleP, CartPoleV
--steps                          total environment steps
--seed                           random seed
--seq-len                        history length T fed to the encoder
--actor-context-mode             fade enables the actor-side FracEncoder
--critic-context-mode            fade enables the critic-side FracEncoder
--actor-context-frac-alpha       initial alpha_a in (0, 1)
--critic-context-frac-alpha      initial alpha_c in (0, 1)
--freeze-actor-context-frac-alpha    fix alpha_a (used in the alpha ablation)
--freeze-critic-context-frac-alpha   fix alpha_c (used in the alpha ablation)
--actor-context-kl-weight        weight of the temporal KL on the actor side
--critic-context-kl-weight       weight of the temporal KL on the critic side
--reward-mode env                train on the raw environment reward
```

FOR EXAMPLE:

```
python -m judge_sac.trainer --env PendulumP --steps 100000 --seed 0 --reward-mode env --disable-judge --actor-context-mode fade --critic-context-mode fade --actor-context-frac-alpha 0.6 --critic-context-frac-alpha 0.6 --seq-len 64 --lr 3e-4
```

## Citation

If you find our work useful, please cite us as follows:

```
@inproceedings{fracencoder2026,
  title     = {FracEncoder: Towards Adaptive Cognitive Trajectories via Fractional-Order Context Encoding},
  author    = {Anonymous},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year      = {2026}
}
```
