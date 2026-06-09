# TML Assignment 3 Robustness - Updated Submission-Compatible Code

This repo matches the task template:

```python
images = torch.from_numpy(data["images"]).float() / 255.0
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 9)
```

Important decisions:

- No ImageNet normalization.
- No dataset mean/std normalization.
- Default torchvision ResNet stem is kept unchanged.
- Saved checkpoints are raw `model.state_dict()` only.
- PGD is generated directly in raw `[0, 1]` image space.

## 1. Inspect dataset

```powershell
python inspect_npz.py --npz_path data/train.npz
```

## 2. Train clean sanity baseline first

```powershell
python train_validate.py `
  --npz_path data/train.npz `
  --output_dir runs/clean_resnet18_no_norm `
  --model_name resnet18 `
  --epochs 80 `
  --batch_size 128 `
  --val_size 0.1 `
  --lr 0.05 `
  --weight_decay 5e-4 `
  --patience 15 `
  --min_epochs 20 `
  --clean_acc_floor 0.55
```

Verify:

```powershell
python verify_submission.py `
  --model_name resnet18 `
  --checkpoint runs/clean_resnet18_no_norm/best_resnet18_clean_state_dict.pt
```

Submit this clean model once to confirm server clean accuracy passes 50%.

## 3. Train first mild PGD baseline

This is intentionally mild to keep server clean accuracy above the 50% gate.

```powershell
python train_pgd.py `
  --npz_path data/train.npz `
  --output_dir runs/pgd_resnet18_eps4_adv04 `
  --model_name resnet18 `
  --epochs 80 `
  --batch_size 128 `
  --val_size 0.1 `
  --lr 0.05 `
  --weight_decay 5e-4 `
  --train_eps 0.015686275 `
  --train_alpha 0.003921568 `
  --train_steps 5 `
  --adv_weight 0.4 `
  --eval_eps 0.031372549 `
  --eval_alpha 0.007843137 `
  --eval_steps 10 `
  --patience 15 `
  --min_epochs 20 `
  --clean_acc_floor 0.60
```

Verify:

```powershell
python verify_submission.py `
  --model_name resnet18 `
  --checkpoint runs/pgd_resnet18_eps4_adv04/best_resnet18_pgd_state_dict.pt
```

## 4. Stronger PGD variants after the mild version passes

Balanced:

```powershell
python train_pgd.py `
  --npz_path data/train.npz `
  --output_dir runs/pgd_resnet18_eps6_adv05 `
  --model_name resnet18 `
  --epochs 100 `
  --batch_size 128 `
  --val_size 0.1 `
  --lr 0.05 `
  --weight_decay 5e-4 `
  --train_eps 0.023529412 `
  --train_alpha 0.005882353 `
  --train_steps 5 `
  --adv_weight 0.5 `
  --eval_eps 0.031372549 `
  --eval_alpha 0.007843137 `
  --eval_steps 10 `
  --patience 15 `
  --min_epochs 20 `
  --clean_acc_floor 0.60
```

Stronger:

```powershell
python train_pgd.py `
  --npz_path data/train.npz `
  --output_dir runs/pgd_resnet18_eps8_adv05 `
  --model_name resnet18 `
  --epochs 100 `
  --batch_size 128 `
  --val_size 0.1 `
  --lr 0.05 `
  --weight_decay 5e-4 `
  --train_eps 0.031372549 `
  --train_alpha 0.007843137 `
  --train_steps 5 `
  --adv_weight 0.5 `
  --eval_eps 0.031372549 `
  --eval_alpha 0.007843137 `
  --eval_steps 10 `
  --patience 15 `
  --min_epochs 20 `
  --clean_acc_floor 0.60
```

## 5. Reproduce the current best model

The current best saved run is `runs/pgd_resnet34_eps6_steps7_adv05`. Re-train it with:

```powershell
python train_pgd.py `
  --npz_path ./data/train.npz `
  --output_dir runs/pgd_resnet50_eps6_steps7_adv06_long `
  --model_name resnet50 `
  --epochs 120 `
  --batch_size 128 `
  --val_size 0.1 `
  --lr 0.05 `
  --weight_decay 5e-4 `
  --train_eps 0.023529412 `
  --train_alpha 0.003921568 `
  --train_steps 7 `
  --adv_weight 0.6 `
  --eval_eps 0.031372549 `
  --eval_alpha 0.007843137 `
  --eval_steps 20 `
  --patience 25 `
  --min_epochs 40 `
  --clean_acc_floor 0.56
```

After training, the checkpoint to verify is `runs/pgd_resnet34_eps6_steps7_adv05/best_resnet34_pgd_state_dict.pt`.
