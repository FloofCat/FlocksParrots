# Flocks of Stochastic Parrots: Differentially Private Prompt Learning for Large Language Models

## About
This repository is an unofficial implementation of the paper "Flocks of Stochastic Parrots: Differentially Private Prompt Learning for Large Language Models" by Duan et al. [arxiv/2305.15594](https://arxiv.org/pdf/2305.15594).

✨ Submission for CS F425 (Deep Learning) Assignment.

## Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Configurations](#config-descriptions)
    - [PromptDPSGD Configurations](#promptdpsgd-configuration)
    - [PromptPATE Configurations](#promptpate-configuration)
4. [Citation](#citation)
5. [License](#license)

---
## Installation
- We recommend running all experiments on 1 A100 GPU with at least 40GB memory. These experiments are specifically designed for the SLURM environment.
- As a result, you can run your experiments using the following Docker image, publicly available on Docker Hub:
```bash
docker pull floofcat/parrots:latest
```
- You will need to install RoBERTa or a model of your choice in `./model-cache` and download the following datasets (as per the paper) in the `./dataset-cache` directory.

---

## Usage
The scripts for these experiments can be run using `main.py`. However, you may need to modify the configuration file (`config.json`) to match your experiment requirements. 

> By default, the configuration file (`config.json`) is set up for a single GPU training. There are three experiments that are supported by this repository:
> - Experiment 1: Deploying a MIA on a model
> - Experiment 2: Training a model via PromptDPSGD
> - Experiment 3: Training a student model via PromptPATE

> You will also need an OPENAI API key to run Experiment 3. GPT-Babbage would be used.

---

## Config Descriptions
`current_run` in the `config.json` file is used to specify the type of experiment that should be run. It accepts "mia", "promptdpsgd", or "promptpate". Other details can be found below:

### PromptDPSGD Configuration

| Parameter          | Description                                                                 | Example Value |
|--------------------|-----------------------------------------------------------------------------|---------------|
| `prompt_length`     | Length of the prompt used during training.                                  | `100`         |
| `learning_rate`     | Learning rate for optimization.                                              | `0.001`       |
| `max_grad_norm`     | Maximum L2 norm for gradients; used in gradient clipping.                   | `1.0`         |
| `noise_multiplier`  | Noise multiplier for Gaussian noise added for DP.                           | `1.0`         |
| `device`            | Device to run training on (`cuda:0` for GPU).                               | `cuda:0`      |
| `target_epsilon`    | Target epsilon for differential privacy.                                     | `"inf"`       |
| `delta`             | Probability of privacy guarantee not holding.                              | `1e-5`        |
| `dataset_path`      | Dataset name used for training.                                     | `"sst2"`      |


### PromptPATE Configuration

| Parameter             | Description                                                                 | Example Value |
|-----------------------|-----------------------------------------------------------------------------|---------------|
| `num_teachers`         | Number of teacher models in PATE aggregation.                              | `10`          |
| `examples_per_teacher`| Number of examples each teacher is trained on.                             | `3`           |
| `sigma1`              | Gaussian noise scale for votes in the first aggregation round.              | `1`           |
| `sigma2`              | Gaussian noise scale for final vote aggregation.                            | `20`          |
| `threshold`           | Threshold for accepting a teacher vote.                                    | `180`         |
| `target_eps`          | Target epsilon for the final privacy guarantee.                            | `0`           |
| `delta`               | Probability of the privacy guarantee not holding.                          | `1e-6`        |
| `candid_prompts`      | Number of candidate prompts to consider.                                   | `3`           |
| `dataset_path`        | Training dataset name.                                           | `"sst2"`      |
| `infer_dataset`       | Dataset used for inference/testing.                                        | `"imdb"`      |

---
### Citation
If you found this work useful, please consider citing the original paper:
```
@misc{duan2023flocksstochasticparrotsdifferentially,
      title={Flocks of Stochastic Parrots: Differentially Private Prompt Learning for Large Language Models}, 
      author={Haonan Duan and Adam Dziedzic and Nicolas Papernot and Franziska Boenisch},
      year={2023},
      eprint={2305.15594},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2305.15594}, 
}
```

---

### Licence
Under the MIT License.
