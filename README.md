# Sentrl AI ML Platform Tools

**Enterprise-grade machine learning platform for building personalized AI agents through user action demonstration.**

Developed by **Sentrl AI Inc** - Advanced AI solutions for intelligent automation and personalized agent development.

The platform captures, processes, and learns from user interactions to create agents that understand and predict user workflows with unprecedented accuracy.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup Ollama with vision model
brew install ollama
ollama serve
ollama pull llama3.2-vision

# 3. Configure the platform
cp config/.env.example config/.env
# Edit config/config.yaml for your preferences

# 4. Capture user actions
python -m core.logger.action_logger

# 5. Process captured data
python -m core.transforms.run_pipeline --data ~/sentrl_data/session_*/raw/

# 6. Annotate with vision model
python -m core.annotator.vision_annotator

# 7. (Optional) Review and correct annotations
python -m tools.ground_truth.reviewer --data ~/sentrl_data/session_*/

# 8. Prepare training dataset
python -m training.prepare_dataset --data ~/sentrl_data/session_*/ --format alpaca

# 9. Train personalized model
python -m training.train --data-dir ~/sentrl_data/session_*/training/ --config training/config.yaml
```

## Features

- **Action Logging**: Capture detailed user interactions (keyboard, mouse, window events) on macOS
- **Vision-Based Annotation**: Automatically label screenshots using Llama 3.2 Vision (via Ollama)
- **Data Processing**: Standardize and transform captured actions
- **Ground Truth Review**: Optional manual review/correction of LLM annotations
- **Training Pipeline**: Fine-tune models with LoRA on user action data
- **Config-Driven**: Single configuration file for all settings

## Project Structure

```
sentrl_platform/
├── config/
│   ├── config.yaml         # Main configuration file
│   └── .env.example        # Environment variables template
│
├── core/
│   ├── config.py           # Configuration loader
│   ├── logger/             # User action logger
│   ├── annotator/          # Vision-based annotation
│   └── transforms/         # Data standardization pipeline
│
├── training/
│   ├── prepare_dataset.py  # Dataset formatting (to be created)
│   ├── train.py            # Model training (to be created)
│   └── models/             # Trained models (gitignored)
│
└── tools/
    └── ground_truth/       # Manual review tool
        └── reviewer.py
```

## Data Pipeline

```
1. [Action Logger] → Capture user actions
   Output: raw/actions.json + raw/screenshots/*.png

2. [Data Transforms] → Standardize format
   Output: processed/actions_standardized.json

3. [Vision Annotator] → Auto-label with LLM
   Output: annotations/llm_responses/*.json
          annotations/actions_annotated.json

4. [Ground Truth Reviewer] → OPTIONAL manual review/correction
   Output: ground_truth/actions_labeled.json

5. [Dataset Formatter] → Prepare for training
   Output: training/train/ and training/test/ (HuggingFace format)

6. [Model Trainer] → Fine-tune with LoRA
   Output: models/adapters/ (LoRA weights)
```

## Installation

### Prerequisites

- Python >=3.10
- macOS (for action logging)
- Ollama with Llama 3.2 Vision model

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/afrozmohammad22/sentrl_ml_platform_tools.git
cd sentrl_ml_platform_tools
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Install Ollama and Llama 3.2 Vision**
```bash
# Install Ollama (if not already installed)
brew install ollama

# Start Ollama service
ollama serve

# Pull Llama 3.2 Vision model
ollama pull llama3.2-vision
```

4. **Configure the platform**
```bash
# Copy environment template
cp config/.env.example config/.env

# Edit config.yaml for your preferences
vim config/config.yaml
```

## Usage

### 1. Capture User Actions

```bash
python -m core.logger.action_logger
```

- Logs all keyboard, mouse, and window events
- Captures screenshots on clicks, Enter/Tab keys, window changes
- Press `CTRL+Z` to stop and save
- Data saved to `~/sentrl_data/session_YYYYMMDD_HHMMSS/raw/`

### 2. Standardize Data

```bash
python -m core.transforms.run_pipeline --data ~/sentrl_data/session_*/raw/
```

- Matches screenshots to events
- Converts datetime formats
- Standardizes keystroke representations
- Output: `processed/actions_standardized.json`

### 3. Annotate with Vision Model

```bash
python -m core.annotator.vision_annotator
```

- Launches PyQt6 GUI
- Select the `processed/` directory
- Analyzes each screenshot with Llama 3.2 Vision
- Saves annotations to `annotations/llm_responses/`

### 4. Optional: Review Annotations

```bash
python -m tools.ground_truth.reviewer --data ~/sentrl_data/session_*/
```

- Loads LLM annotations (pre-filled in form)
- Review and correct labels
- Saves corrections to `ground_truth/actions_labeled.json`

### 5. Prepare Training Dataset

```bash
python -m training.prepare_dataset \
  --data ~/sentrl_data/session_*/ \
  --format alpaca \
  --test-size 0.2
```

- Formats data for instruction tuning (Alpaca or OAI-chat format)
- Creates train/val/test splits
- Generates HuggingFace dataset structure
- Prioritizes ground truth labels over LLM annotations
- Output: `training/train/`, `training/val/`, `training/test/`

### 6. Train Model

```bash
python -m training.train \
  --data-dir ~/sentrl_data/session_*/training/ \
  --config training/config.yaml
```

- Fine-tunes DeepSeek-1.5B with LoRA
- Uses MPS (Apple Silicon) or CPU automatically
- Saves checkpoints to `training/models/run_*/`
- Saves LoRA adapters to `training/models/adapters/`
- Tracks perplexity and loss metrics

## Configuration

Edit `config/config.yaml` to customize:

- **Data paths**: Where to store sessions and models
- **Logger settings**: Screenshot format, scroll timeout
- **Vision model**: Model name, endpoint, temperature
- **Training**: Base model, LoRA parameters, hyperparameters

## Data Directory Structure

Each capture session creates this structure:

```
~/sentrl_data/session_YYYYMMDD_HHMMSS/
├── raw/
│   ├── actions.json
│   └── screenshots/
│       └── screen_*.png
├── processed/
│   └── actions_standardized.json
├── annotations/
│   ├── llm_responses/
│   │   └── screen_*_analysis.json
│   └── actions_annotated.json
├── ground_truth/              # Optional
│   └── actions_labeled.json
└── training/
    ├── train/
    │   ├── metadata.jsonl
    │   └── *.png
    └── test/
        ├── metadata.jsonl
        └── *.png
```

## Model Details

### Base Model
- **DeepSeek-R1-Distill-Qwen-1.5B**: Efficient 1.5B parameter model
- Optimized for Apple Silicon (MPS) and CPU

### Fine-Tuning
- **LoRA** (Low-Rank Adaptation): Parameter-efficient training
- ~37M trainable parameters (2-3% of total)
- Trains on consumer hardware (16GB+ RAM recommended)

### Vision Model
- **Llama 3.2 Vision** (3B): Multimodal model for screenshot analysis
- Runs via Ollama locally
- Analyzes UI context, user actions, and workflow patterns

## Development

### Adding New Data Transforms

Add your transform script to `core/transforms/` and import in `run_pipeline.py`:

```python
from core.transforms.your_transform import process_function
```

### Customizing Prompts

Edit prompts in `config/config.yaml` under the `prompts` section.

## Troubleshooting

### Action Logger Issues
- **Permission errors**: Grant Terminal accessibility permissions in System Preferences
- **Screenshots not capturing**: Check screen recording permissions

### Vision Annotator Issues
- **Connection refused**: Ensure Ollama is running (`ollama serve`)
- **Model not found**: Pull the model (`ollama pull llama3.2-vision`)
- **Slow performance**: Reduce batch size or use smaller model

### Training Issues
- **Out of memory**: Reduce batch size, increase gradient accumulation
- **MPS errors**: Fall back to CPU by setting `device: cpu` in config

## License

**Proprietary Software - All Rights Reserved**

Copyright (c) 2024-2025 Sentrl AI Inc. All rights reserved.

This software and associated documentation are proprietary and confidential to Sentrl AI Inc.
Unauthorized copying, modification, distribution, or use is strictly prohibited.

See the [LICENSE](LICENSE) file for complete terms and conditions.

## Privacy Note

**Important**: The action logger captures ALL user interactions including potentially sensitive data:
- Keystrokes (including passwords if typed)
- Screenshots of all windows
- Application names and window titles

Only use on your own machine for your own actions. Review and anonymize data before sharing.

## Roadmap

- [x] Complete `training/prepare_dataset.py` (merge formatting scripts) ✅
- [x] Complete `training/train.py` (adapt from proof-of-concept) ✅
- [ ] Add inference module for agent deployment
- [ ] Build unified CLI interface
- [ ] Add automated pipeline orchestration
- [ ] Update ground truth reviewer for review/correction mode
- [ ] Create web UI alternative to CLI
- [ ] Support for additional platforms (Linux, Windows)
- [ ] Add RAG for action retrieval (FAISS/Qdrant)

## Contact & Support

**Sentrl AI Inc**
- Website: https://www.sentrlai.com
- Email: support@sentrlai.com
- GitHub: https://github.com/afrozmohammad22/sentrl_ml_platform_tools

For licensing inquiries: legal@sentrlai.com

---

© 2024-2025 Sentrl AI Inc. All rights reserved.
