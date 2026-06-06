# Assignment 3 - Robustness

Detect whether given suspect models are stolen versions of a target model.

## Overview

This assignment involves analyzing 360 suspect models to determine if they are stolen versions of the target model. Each suspect model needs to be evaluated and assigned a confidence score indicating the likelihood that it's a stolen model.

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

- **`target_model/`**: The target model to compare against
- **`suspect_models/`**: 360 suspect model files (suspect_000.safetensors to suspect_359.safetensors)
- **`task_template.py`**: Example code showing how to load models and format submissions
- **`submission.py`**: Your implementation for the model stealing detection algorithm
- **`env.example`**: Template for environment variables

## Model Loading

Use the provided example in `task_template.py` to load models from safetensors format:

```python
from safetensors.torch import load_file
state_dict = load_file("path/to/model.safetensors", device="cpu")
model = make_model()
model.load_state_dict(state_dict, strict=True)
model.eval()
```

## Resources

- See `task_template.py` for model loading examples
- See `submission.py` for submission format requirements and validation rules
- CIFAR-100 dataset can be automatically downloaded during model evaluation

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

4. Generate the scored functional submission (this reproduces the exact CSV we submitted):

```bash
python score_functional_features.py \
	--features outputs_functional/features_train_target_target_aug_n-1_temp2.0_seed42.csv \
	--variant simple_top5_samewrong \
	--out_dir outputs_functional_scored
```

This will write two files into `outputs_functional_scored/`:

- `scored_features_train_target_target_aug_n-1_temp2.0_seed42_simple_top5_samewrong.csv` (scored features with diagnostics)
- `submission_features_train_target_target_aug_n-1_temp2.0_seed42_simple_top5_samewrong.csv` (the final submission file)

5. (Optional) Submit the produced CSV to the server using the provided submit helper:

```bash
# Ensure .env contains API_KEY
python submission.py
```

`submission.py` is pre-configured to upload the file at
`./outputs_functional_scored/submission_features_train_target_target_aug_n-1_temp2.0_seed42_simple_top5_samewrong.csv`.

Notes and reproducibility details:

- The `--variant simple_top5_samewrong` scoring rule (in `score_functional_features.py`) produced the best leaderboard score for our run.
- The input features file used is `outputs_functional/features_train_target_target_aug_n-1_temp2.0_seed42.csv` (already included in the repository outputs).
- If you change `--variant` or the features CSV, leaderboard results will differ.
- Random seeds are not required for the scoring step (it is deterministic given the same features CSV). If you regenerate the features CSV from raw model runs, use seed `42` where applicable to match our preprocessing pipeline.

