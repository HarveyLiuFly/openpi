# Model Architecture

This document describes the network architecture of the π₀ and π₀.₅ models in the openpi repository.

## Overview

The openpi repository contains three types of Vision-Language-Action (VLA) models:
- **π₀ (pi0)**: Flow-based VLA model
- **π₀-FAST (pi0_fast)**: Autoregressive VLA with FAST action tokenizer
- **π₀.₅ (pi05)**: Upgraded version of π₀ with better open-world generalization

This document focuses on the π₀ and π₀.₅ architectures.

## π₀ and π₀.₅ Base Architecture

Both π₀ and π₀.₅ models share a common base architecture with key differences in how they process state information and timesteps.

### High-Level Components

The models consist of the following main components:

1. **Vision Encoder (SigLIP)**: Processes visual observations from multiple camera views
2. **Language Model (Gemma)**: Dual Gemma transformers for vision-language understanding and action prediction
3. **Action Expert**: Specialized transformer for action sequence generation
4. **Projection Layers**: Linear layers for embedding and output projection

### Detailed Architecture

#### 1. Vision Encoder (SigLIP)

- **Model**: SigLIP So400m/14 variant
- **Purpose**: Encodes RGB images from multiple camera views into visual tokens
- **Configuration**:
  - Pool type: None (uses all patch tokens)
  - Scan: Enabled for efficient processing
  - Precision: bfloat16 for inference, configurable for training

The vision encoder processes three standard camera views:
- `base_0_rgb`: Base/static camera view
- `left_wrist_0_rgb`: Left wrist-mounted camera
- `right_wrist_0_rgb`: Right wrist-mounted camera

Each image is resized to 224×224 resolution and encoded into a sequence of visual tokens.

#### 2. Language Model (PaliGemma)

The model uses two Gemma transformer variants:

**Primary Gemma (PaliGemma backbone)**:
- **Variant**: Gemma 2B by default
- **Purpose**: Processes vision and language inputs jointly
- **Input**: Concatenated visual tokens + tokenized language prompt

**Action Expert Gemma**:
- **Variant**: Gemma 300M by default
- **Purpose**: Generates action sequences conditioned on vision-language understanding
- **Input**: Action tokens + temporal information

#### 3. Embedding and Projection Layers

**Input Projections**:
- `action_in_proj`: Projects action dimensions to action expert width
  - Input: `action_dim` (default: 32)
  - Output: Gemma width (typically 768 for 300M variant)

**π₀-specific layers**:
- `state_proj`: Projects robot state to action expert width
- `action_time_mlp_in` and `action_time_mlp_out`: MLPs for mixing timestep and action information
  - Input: 2 × action_expert_width
  - Output: action_expert_width

**π₀.₅-specific layers**:
- `time_mlp_in` and `time_mlp_out`: MLPs for processing flow matching timestep (for AdaRMS)
  - Input: action_expert_width
  - Output: action_expert_width

**Output Projection**:
- `action_out_proj`: Projects from action expert width back to action space
  - Input: action_expert_width
  - Output: `action_dim` (default: 32)

### Forward Pass

#### Prefix Embedding (Vision-Language Processing)

1. **Image Embedding**: Each camera view is passed through SigLIP encoder
   - Output: Visual tokens per image (typically 256 tokens for 224×224 images with 14×14 patches)

2. **Language Embedding**: Tokenized prompt is embedded using Gemma's embedding layer
   - Maximum token length: 48 for π₀, 200 for π₀.₅

3. **Concatenation**: Visual tokens from all views + language tokens are concatenated
   - All tokens can attend to each other (full attention mask)

#### Suffix Embedding (Action Processing)

**π₀ approach**:
1. Robot state is projected to a single state token via `state_proj`
2. Noisy actions are projected via `action_in_proj`
3. Flow matching timestep is embedded using sine-cosine positional encoding
4. Timestep and action embeddings are concatenated and processed through MLPs
5. Result forms the action expert tokens

**π₀.₅ approach**:
1. Robot state is tokenized and included in the discrete language tokens (in prefix)
2. Noisy actions are projected via `action_in_proj`
3. Flow matching timestep is processed through time MLPs to generate AdaRMS conditioning
4. Action tokens use AdaRMSNorm with timestep conditioning in the action expert

#### Transformer Processing

1. **Prefix Processing**: Vision-language tokens are processed through the primary Gemma transformer
   - Self-attention across all image and language tokens
   - Outputs rich multimodal representations

2. **Action Generation**: Action expert tokens are processed through the action expert Gemma
   - Prefix-LM attention: action tokens attend to all prefix tokens + previous action tokens
   - For π₀.₅: AdaRMSNorm injects flow matching timestep information
   - For π₀: Timestep is mixed directly with action embeddings via MLPs

3. **Output Projection**: Action expert outputs are projected to action space via `action_out_proj`
   - Output shape: `[batch, action_horizon, action_dim]`
   - Default: [batch, 50, 32] (50 timesteps, 32-dimensional actions)

## Key Differences Between π₀ and π₀.₅

| Aspect | π₀ | π₀.₅ |
|--------|-----|------|
| **State Input** | Continuous state token in suffix | Discrete tokens in prefix (part of language) |
| **Timestep Conditioning** | Mixed with actions via MLPs | Injected via AdaRMSNorm in action expert |
| **Max Token Length** | 48 | 200 |
| **Attention Mechanism** | Standard transformer attention | AdaRMSNorm-conditioned attention |
| **Architecture Changes** | `state_proj`, `action_time_mlp_*` | `time_mlp_*` for AdaRMS conditioning |

## π₀.₅ ALOHA Sim Configuration

The `pi05_aloha_sim` configuration is specifically designed for training π₀.₅ on the ALOHA simulator:

```python
TrainConfig(
    name="pi05_aloha_sim",
    model=pi0_config.Pi0Config(pi05=True),  # Enable π₀.₅ features
    data=LeRobotAlohaDataConfig(
        repo_id="lerobot/aloha_sim_transfer_cube_human",
        default_prompt="Transfer cube",
        use_delta_joint_actions=False,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    num_train_steps=20_000,
)
```

### Key Configuration Details

- **Model**: π₀.₅ architecture (Pi0Config with `pi05=True`)
- **Dataset**: ALOHA simulator cube transfer task from LeRobot
- **Task**: Transfer cube manipulation
- **Action Space**: Absolute joint positions (not delta actions)
- **Base Weights**: Pre-trained π₀.₅ base model with 10k+ hours of robot data
- **Training Steps**: 20,000 fine-tuning steps

### Network Parameters

Default π₀.₅ configuration:
- **Action Dimension**: 32 (joint positions + gripper states for bimanual robot)
- **Action Horizon**: 50 timesteps (predicts 50-step action sequences)
- **Vision Encoder**: SigLIP So400m/14
- **Primary LLM**: Gemma 2B
- **Action Expert**: Gemma 300M with AdaRMSNorm
- **Precision**: bfloat16 for efficiency

## Flow Matching Training

Both π₀ and π₀.₅ use flow matching for training:

1. **Noise Injection**: Clean actions are corrupted with Gaussian noise
   - `x_t = t * noise + (1 - t) * actions`
   - Time `t` is sampled from Beta(1.5, 1) distribution

2. **Velocity Prediction**: Model predicts the velocity field `u_t = noise - actions`

3. **Loss**: Mean squared error between predicted and true velocity

4. **Inference**: Actions are denoised through multiple diffusion steps

## Usage Example

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

# Load π₀.₅ ALOHA Sim configuration
config = _config.get_config("pi05_aloha_sim")

# Download checkpoint
checkpoint_dir = download.maybe_download(
    "gs://openpi-assets/checkpoints/pi05_aloha_sim"
)

# Create policy
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference
observation = {
    "observation/exterior_image_1_left": ...,  # [h, w, 3] RGB image
    "observation/wrist_image_left": ...,        # [h, w, 3] RGB image
    "observation/wrist_image_right": ...,       # [h, w, 3] RGB image
    "observation/state": ...,                    # [state_dim] robot state
    "prompt": "Transfer cube"
}
action_chunk = policy.infer(observation)["actions"]  # [50, 32] action sequence
```

## References

- [π₀ Blog Post](https://www.physicalintelligence.company/blog/pi0)
- [π₀.₅ Blog Post](https://www.physicalintelligence.company/blog/pi05)
- [π₀-FAST Research Paper](https://www.physicalintelligence.company/research/fast)
- [Knowledge Insulation](https://www.physicalintelligence.company/research/knowledge_insulation)
