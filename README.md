# TML Assignment 3 - Robustness

Train an image classifier that is robust against adversarial attacks

## Overview
This assignment invloves training a robust classifier that maintains high accuracy on both clean and adversarially perturbed inputs.

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


## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
# or
source .venv/bin/activate  # On macOS/Linux
```

2. Install dependencies:

```bash
pip install torch torchvision pandas numpy scikit-learn requests python-dotenv safetensors
```

3. Create a `.env` file in the project root and add your API key:

```env
API_KEY=your_api_key_here
```

You can copy `env.example` and replace the placeholder value.

## Project Structure
- **`data/`**: The given training dataset of 50,000 labeled images of shape 3x32x32 across 9 classes
- **`runs/`**: Saves the history for each epoch, final model state dict, summary and config 
- **`task_template.py`**: Example code showing how to train models
- **`submission.py`**: File to submit the robust model.pt to leaderboard
- **`env.example`**: Template for environment variables


## Reproducing Our Best Leaderboard Result
Follow these exact steps to recreate the submission file that produced our best leaderboard score.

1. Create and activate the virtual environment (Windows example):

```powershell
python -m venv .venv
.venv\Scripts\activate
```

2. Install required packages:

```bash
pip install torch torchvision pandas numpy scikit-learn requests python-dotenv safetensors
```

3. (Optional) Copy the example environment file and set your API key (required for online submission):

```bash
copy env.example .env
# edit .env and set API_KEY to your key
```

4. Train the robust model:
The best saved run is `runs/pgd_resnet50_eps6_steps7_adv06_epochs200_seed20` with `0.587303` score on leaderboard. Re-train it with:

```powershell
python train_pgd.py `
  --npz_path ./data/train.npz `
  --output_dir runs/pgd_resnet50_eps6_steps7_adv06_epochs200_seed20 `
  --model_name resnet50 `
  --epochs 200 `
  --batch_size 128 `
  --val_size 0.1 `
  --lr 0.05 `
  --weight_decay 5e-4 `
  --seed 20 `
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

After training, the checkpoint to verify is `runs/pgd_resnet50_eps6_steps7_adv06_epochs200_seed20/best_resnet50_pgd_state_dict.pt`.

5. (Optional) Submit the model to the server using the provided submit helper:

```bash
# Ensure .env contains API_KEY
python submission.py
```

`submission.py` is pre-configured to upload the file at  `runs/pgd_resnet50_eps6_steps7_adv06_epochs200_seed20/best_resnet50_pgd_state_dict.pt`.
