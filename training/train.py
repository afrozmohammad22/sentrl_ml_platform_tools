# Copyright (c) 2024-2025 Sentrl AI Inc. All rights reserved.
# This software is proprietary and confidential.
# Unauthorized copying, modification, or distribution is strictly prohibited.

"""
Model Training for Sentrl AI Agent Platform
Fine-tunes models with LoRA on user action data
Adapted for generic action-based learning (not conversation-specific)
"""

import os
import sys
import yaml
import json
import torch
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import gc

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel
)
from datasets import load_dataset, Dataset
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger(__name__)


def load_training_config(config_path: str = 'training/config.yaml') -> Dict:
    """Load training configuration from YAML"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded training config from: {config_path}")
    return config


def get_available_device():
    """Get the best available device for training"""
    if torch.backends.mps.is_available():
        return {
            "device": "mps",
            "device_name": "Apple Silicon (MPS)",
            "supports_fp16": False,
            "supports_bf16": False,
            "device_map": None
        }
    elif torch.cuda.is_available():
        return {
            "device": "cuda",
            "device_name": f"CUDA ({torch.cuda.get_device_name()})",
            "supports_fp16": True,
            "supports_bf16": torch.cuda.is_bf16_supported(),
            "device_map": "auto"
        }
    else:
        return {
            "device": "cpu",
            "device_name": "CPU",
            "supports_fp16": False,
            "supports_bf16": False,
            "device_map": None
        }


def cleanup_memory():
    """Clean up memory"""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Log memory usage
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        logger.info(f"Memory Usage: {memory_info.rss / (1024 * 1024 * 1024):.2f}GB")
    except ImportError:
        pass


def create_lora_model(config: Dict, tokenizer):
    """Create a LoRA model for fine-tuning"""
    logger.info("Initializing LoRA model...")

    device_info = get_available_device()
    logger.info(f"Using device: {device_info['device_name']}")

    # Configure LoRA
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['dropout'],
        bias=config['lora']['bias'],
        task_type=config['lora']['task_type']
    )

    # Load base model
    model_name = config['model']['base_model']
    logger.info(f"Loading base model: {model_name}")

    # Determine dtype based on device
    if device_info['device'] == 'mps':
        torch_dtype = torch.float32  # MPS requires float32
    elif device_info['device'] == 'cuda' and device_info['supports_bf16']:
        torch_dtype = torch.bfloat16
    elif device_info['device'] == 'cuda' and device_info['supports_fp16']:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_info["device_map"],
        trust_remote_code=config['model'].get('trust_remote_code', True),
        use_flash_attention_2=False,  # Disable for compatibility
        use_cache=False  # Disable KV cache for training
    )

    # Add LoRA adapters
    model = get_peft_model(model, lora_config)

    # Enable gradient checkpointing for memory efficiency
    if config['training']['gradient_checkpointing']:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # Print trainable parameters
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    logger.info(
        f"Trainable params: {trainable_params:,} || "
        f"All params: {all_param:,} || "
        f"Trainable%: {100 * trainable_params / all_param:.2f}"
    )

    return model, device_info


def prepare_dataset(data_dir: Path, tokenizer, config: Dict):
    """
    Load and prepare dataset from HuggingFace format.

    Expected structure:
    data_dir/
      train/
        metadata.jsonl
        *.png
      val/
        metadata.jsonl
        *.png
      test/
        metadata.jsonl
        *.png
    """
    logger.info(f"Loading dataset from: {data_dir}")

    # Load from directory
    # Note: This is a simplified version. For multimodal (image+text),
    # you'll need custom dataset loading logic
    try:
        dataset = load_dataset(
            'json',
            data_files={
                'train': str(data_dir / 'train' / 'metadata.jsonl'),
                'validation': str(data_dir / 'val' / 'metadata.jsonl'),
                'test': str(data_dir / 'test' / 'metadata.jsonl')
            }
        )
        logger.info(f"Dataset loaded: {dataset}")
        return dataset

    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def compute_metrics(eval_preds):
    """Compute metrics for model evaluation"""
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Calculate perplexity
    loss = np.mean(
        np.where(
            labels != -100,
            -np.log(np.take_along_axis(logits, labels[..., None], axis=-1).squeeze(-1)),
            0
        )
    )
    perplexity = np.exp(loss)

    return {"perplexity": perplexity}


def get_training_args(config: Dict, device_info: Dict, output_dir: Path) -> TrainingArguments:
    """Create TrainingArguments from config"""
    training_config = config['training']

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_config['num_train_epochs'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        eval_steps=training_config['eval_steps'],
        save_steps=training_config['save_steps'],
        logging_steps=training_config['logging_steps'],
        learning_rate=training_config['learning_rate'],
        evaluation_strategy=training_config['evaluation_strategy'],
        save_strategy=training_config['save_strategy'],
        save_total_limit=training_config['save_total_limit'],
        load_best_model_at_end=training_config['load_best_model_at_end'],
        metric_for_best_model=training_config.get('metric_for_best_model', 'eval_loss'),
        report_to=training_config['report_to'],
        remove_unused_columns=False,
        optim=training_config['optim'],
        ddp_find_unused_parameters=False,
        group_by_length=training_config['group_by_length'],
        warmup_ratio=training_config['warmup_ratio'],
        lr_scheduler_type=training_config['lr_scheduler_type'],
        weight_decay=training_config['weight_decay'],
        gradient_checkpointing=training_config['gradient_checkpointing'],
        max_grad_norm=training_config['max_grad_norm'],
        dataloader_pin_memory=training_config['dataloader_pin_memory'],
        torch_compile=False,
        seed=42
    )

    # Device-specific adjustments
    if device_info["device"] == "mps":
        args.fp16 = False
        args.bf16 = False
        args.half_precision_backend = None
    elif device_info["device"] == "cuda":
        args.fp16 = device_info['supports_fp16'] and not device_info['supports_bf16']
        args.bf16 = device_info['supports_bf16']

    return args


def train_model(
    config_path: str = 'training/config.yaml',
    data_dir: Optional[str] = None
):
    """Main training function"""
    try:
        # Load config
        config = load_training_config(config_path)

        # Determine data directory
        if data_dir is None:
            # Default: look for most recent session
            logger.error("Please specify --data-dir with the path to your training data")
            return

        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Create output directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = config['output'].get('run_name') or f"run_{timestamp}"
        output_dir = Path(config['output']['base_dir']) / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Output directory: {output_dir}")

        # Load tokenizer
        model_name = config['model']['base_model']
        logger.info(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=config['model'].get('trust_remote_code', True)
        )

        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Create model
        model, device_info = create_lora_model(config, tokenizer)
        model.resize_token_embeddings(len(tokenizer))

        # Load dataset
        dataset = prepare_dataset(data_dir, tokenizer, config)

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # Get training arguments
        training_args = get_training_args(config, device_info, output_dir)

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=config['training']['early_stopping_patience'],
                    early_stopping_threshold=config['training']['early_stopping_threshold']
                )
            ]
        )

        # Train
        logger.info("Starting training...")
        train_result = trainer.train()
        logger.info("Training completed successfully")

        # Evaluate
        logger.info("Running final evaluation...")
        eval_results = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_results}")

        # Test
        logger.info("Running test evaluation...")
        test_results = trainer.evaluate(dataset['test'], metric_key_prefix="test")
        logger.info(f"Test results: {test_results}")

        # Save final model
        logger.info(f"Saving final model to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Save LoRA adapters separately
        adapter_dir = Path(config['output']['adapter_dir'])
        adapter_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving LoRA adapters to {adapter_dir}")
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        # Save results
        results = {
            "train_loss": train_result.metrics["train_loss"],
            "eval_loss": eval_results["eval_loss"],
            "eval_perplexity": eval_results.get("eval_perplexity"),
            "test_loss": test_results["test_loss"],
            "test_perplexity": test_results.get("test_perplexity"),
            "best_checkpoint": trainer.state.best_model_checkpoint
        }

        results_path = output_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info("Training completed successfully!")
        print("\n✅ Training Complete!")
        print(f"  Output: {output_dir}")
        print(f"  Adapters: {adapter_dir}")
        print(f"  Train loss: {results['train_loss']:.4f}")
        print(f"  Eval loss: {results['eval_loss']:.4f}")
        print(f"  Test loss: {results['test_loss']:.4f}")

    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        cleanup_memory()
        sys.exit(1)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        cleanup_memory()
        raise


def main():
    parser = argparse.ArgumentParser(description='Train Sentrl AI Agent Model')
    parser.add_argument('--config', type=str, default='training/config.yaml',
                        help='Path to training config file')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing train/val/test data')

    args = parser.parse_args()

    train_model(config_path=args.config, data_dir=args.data_dir)


if __name__ == '__main__':
    main()
