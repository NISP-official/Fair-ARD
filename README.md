## Revisiting Adversarial Robustness Distillation from the Perspective of Robust Fairness

Code for NeurIPS 2023 "Revisiting Adversarial Robustness Distillation from the Perspective of Robust Fairness".

## Environment

- Python (3.9.12)
- Pytorch (1.11.0)
- torchvision (0.12.0)
- CUDA
- AutoAttack

## Content

- ```./models```: Models used for distillation.
- ```Fair-ARD.py```: Fair Adversarial Robustness Distillation.
- ```Fair-IAD.py```: Fair Introspective Adversarial Distillation.
- ```Fair-RSLAD.py```: Fair Robust Soft Label Adversarial Distillation.
- ```Fair-MTARD.py```: Fair Multi-Teacher Adversarial Robustness Distillation.
- ```eval.py```:  Evaluate the average robustness and worst-class robustness of the model.

## Run

- Fair-ARD
```bash
CUDA_VISIBLE_DEVICES='0' python Fair-ARD.py --teacher_path INSERT-YOUR-TEACHER-PATH --beta 2.0
```

- Fair-IAD
```bash
CUDA_VISIBLE_DEVICES='0' python Fair-IAD.py --teacher_path INSERT-YOUR-TEACHER-PATH --beta 2.0
```

- Fair-RSLAD

```bash
CUDA_VISIBLE_DEVICES='0' python Fair-RSLAD.py --teacher_path INSERT-YOUR-TEACHER-PATH --beta 2.0
```

- Fair-MTARD

```bash
CUDA_VISIBLE_DEVICES='0' python Fair-MTARD.py --adv_teacher_path INSERT-YOUR-ADV-TEACHER-PATH --nat_teacher_path INSERT-YOUR-NAT-TEACHER-PATH --beta 2.0
```

- Evaluation

```bash
CUDA_VISIBLE_DEVICES='0' python eval.py --model_path INSERT-YOUR-MODEL-PATH
```

## Pre-trained Models

- The teacher models and pre-trained models can be downloaded from the [Google Drive](https://drive.google.com/drive/folders/1LQ24_b6-cw8V4H6u48Tb-HfhTg4XEJzK?usp=drive_link)

## Reference Code

[1] ARD: https://github.com/goldblum/AdversariallyRobustDistillation

[2] IAD: https://github.com/ZFancy/IAD

[3] RSLAD: https://github.com/zibojia/RSLAD

[4] MTARD: https://github.com/zhaoshiji123/MTARD

[5] GAIRAT: https://github.com/zjfheart/Geometry-aware-Instance-reweighted-Adversarial-Training