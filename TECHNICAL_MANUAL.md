# Technical Manual: MLX Fine-tuning Pipeline

A comprehensive technical guide for fine-tuning the `gemma-3-1b-it-4bit` model using MLX for function calling tasks.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

```bash
# Step 1: Load data from HuggingFace
python src/load_data.py --dataset Salesforce/xlam-function-calling-60k --split train --num_examples 1000

# Step 2: Prepare data for Gemma model and split into train/validation/test
python src/prep_data.py --input_file data/training/xlam-function-calling-60k_train_1000examples
```

### 3. Start Fine-tuning

```bash
# Basic fine-tuning with MLX-LM native command (4-bit quantized model)
python -m mlx_lm lora \
  --model mlx-community/gemma-3-1b-it-4bit \
  --train \
  --data data/training \
  --iters 500 \
  --val-batches 20 \
  --learning-rate 1e-4 \
  --batch-size 2 \
  --num-layers 8 \
  --save-every 100 \
  --adapter-path gemma-3-1b-function-calling-4bit \
  --grad-checkpoint \
  --seed 42

# Resume from checkpoint
python -m mlx_lm lora \
  --model mlx-community/gemma-3-1b-it-4bit \
  --train \
  --data data/training \
  --adapter-path gemma-3-1b-function-calling-4bit \
  --resume-adapter-file gemma-3-1b-function-calling-4bit/adapters.safetensors
```

### 4. Test Your Model

```bash
# Test base model performance
python src/inference.py

# Test fine-tuned model (easy-to-use script)
python src/test_model.py

# Test with more examples and verbose output
python src/test_model.py --num_examples 10 --verbose

# Analyze and compare results
python src/analyze_results.py --verbose
```

## 📁 Project Structure

```
mlx_finetuning/
├── src/
│   ├── load_data.py         # Load data from HuggingFace
│   ├── prep_data.py         # Format data for Gemma model
│   ├── inference.py         # Inference script
│   ├── test_model.py        # Easy-to-use test script for fine-tuned model
│   └── analyze_results.py   # Compare and analyze test results
├── data/
│   ├── training/            # Training data
│   │   ├── train.jsonl      # Training data (80%)
│   │   ├── valid.jsonl      # Validation data (10%)
│   │   └── test.jsonl       # Test data (10%)
│   └── results/             # Test results
│       ├── base_model_test_results.json
│       └── finetuned_model_test_results.json
├── gemma-3-1b-function-calling-4bit/  # Fine-tuned model weights
├── requirements.txt         # Python dependencies
├── README.md               # Research paper
└── TECHNICAL_MANUAL.md     # This file
```

## 🔧 Configuration Options

### Data Preparation (`prep_data.py`)

```bash
python src/prep_data.py [OPTIONS]

Options:
  --input_file TEXT       Path to input dataset file (required)
  --output_dir TEXT       Output directory for split data (default: root/data)
  --num_examples INTEGER  Maximum number of examples to process
  --train_ratio FLOAT     Proportion for training set (default: 0.8)
  --val_ratio FLOAT       Proportion for validation set (default: 0.1)
  --test_ratio FLOAT      Proportion for test set (default: 0.1)
  --random_seed INTEGER   Random seed for reproducible splits (default: 42)
```

### MLX-LM LoRA Training

```bash
python -m mlx_lm lora [OPTIONS]

Key Options:
  --model TEXT            Base model to fine-tune (use gemma-3-1b-it-4bit)
  --train                 Enable training mode
  --data TEXT             Data directory containing train.jsonl, valid.jsonl, test.jsonl (use data/training)
  --iters INTEGER         Number of training iterations (tested: 500)
  --val-batches INTEGER   Number of validation batches (tested: 20)
  --learning-rate FLOAT   Learning rate (tested: 1e-4)
  --batch-size INTEGER    Training batch size (tested: 2)
  --num-layers INTEGER    Number of layers to apply LoRA to (tested: 8)
  --save-every INTEGER    Save checkpoint every N steps (tested: 100)
  --adapter-path TEXT     Path to save the fine-tuned adapter
  --resume-adapter-file TEXT  Resume from checkpoint
  --grad-checkpoint       Enable gradient checkpointing (saves memory)
  --seed INTEGER          Random seed (default: 42)
```

## 📊 Data Format

The pipeline expects data in the following format:

### Input Format (from HuggingFace)
```json
{
  "query": "What's the weather in New York?",
  "tools": "[{\"name\": \"get_weather\", \"description\": \"Get weather information\", \"parameters\": {\"location\": {\"type\": \"string\", \"description\": \"City name\"}}}]",
  "answers": "[{\"name\": \"get_weather\", \"arguments\": {\"location\": \"New York\"}}]"
}
```

### Output Format (for MLX)
```json
{"text": "<bos><start_of_turn>user\n## Instructions\nYou are a helpful assistant with access to the following tools:\n\n```json\n[{\"name\": \"get_weather\", \"description\": \"Get weather information\", \"parameters\": {\"location\": {\"type\": \"string\", \"description\": \"City name\"}}}]\n```\n\n## User\nWhat's the weather in New York?<end_of_turn>\n<start_of_turn>model\n```json\n[{\"id\": \"call_id_0\", \"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"arguments\": \"{\\\"location\\\": \\\"New York\\\"}\"}}]\n```<end_of_turn><eos>"}
```

## 🎯 Fine-tuning Tips

### 1. **Use 4-bit Quantized Models**
- **Recommended**: `mlx-community/gemma-3-1b-it-4bit` (tested & proven)
- **Avoid**: `gemma-3-1b-it-bf16` (Metal compatibility issues on some systems)
- **Benefits**: Lower memory usage, faster training, better compatibility

### 2. **Start with Proven Configuration**
- **Iterations**: 500 (sufficient for 1000 examples)
- **Learning Rate**: 1e-4 (stable and effective)
- **Batch Size**: 2 (memory efficient)
- **Num Layers**: 8 (good balance of parameters and performance)

### 3. **Monitor Training Progress**
- Watch validation loss decrease from ~2.2 to ~0.8
- Save checkpoints every 100 iterations with `--save-every 100`
- Use `--grad-checkpoint` to save memory (essential for 4-bit models)

### 4. **Memory Management**
- **Peak Usage**: ~8.35 GB with tested configuration
- **Gradient Checkpointing**: Always use `--grad-checkpoint`
- **Batch Size**: Keep at 2 for stability

## 🔍 Troubleshooting

### Common Issues

1. **Metal Driver Errors (macOS)**
   - **Solution**: Use 4-bit quantized models (`gemma-3-1b-it-4bit`)
   - **Avoid**: `gemma-3-1b-it-bf16` on some Apple Silicon systems
   - **Alternative**: Try CPU-only training with `MLX_DEFAULT_DEVICE=cpu`

2. **Out of Memory**
   - **Solution**: Use `--grad-checkpoint` (essential)
   - **Reduce**: `--batch-size` to 1 or 2
   - **Model**: Use 4-bit quantized models

3. **Poor Performance**
   - **Increase**: `--iters` to 500-1000
   - **Learning Rate**: Try 1e-4 (proven effective)
   - **Data**: Ensure proper function calling format

4. **Data Format Issues**
   - **Check**: Data has `query`, `tools`, `answers` fields
   - **Verify**: Tools are valid JSON strings
   - **Format**: Answers are JSON arrays with function calls

5. **Model Loading Issues**
   - **Fine-tuned**: Use `load(model, adapter_path="path")`
   - **Base**: Use `load("mlx-community/gemma-3-1b-it-4bit")`
   - **Check**: Adapter files exist in specified path

## 📈 Monitoring Training

MLX-LM provides detailed logging during training:

```
Loading pretrained model
Loading datasets
Training
Trainable parameters: 0.018% (0.229M/1301.876M)
Starting training..., iters: 500

Iter 1: Val loss 2.211, Val took 18.748s
Iter 10: Train loss 1.819, Learning Rate 1.000e-04, It/sec 0.606
Iter 100: Train loss 0.839, Learning Rate 1.000e-04, It/sec 0.663
Iter 200: Val loss 0.923, Val took 15.760s
Iter 300: Train loss 0.887, Learning Rate 1.000e-04, It/sec 0.588
Iter 400: Val loss 0.911, Val took 16.583s
Iter 500: Val loss 0.840, Val took 15.256s
Iter 500: Train loss 0.783, Learning Rate 1.000e-04, It/sec 0.671

Saved final weights to gemma-3-1b-function-calling-4bit/adapters.safetensors.
```

### **Expected Training Progress**
- **Initial Loss**: ~2.2 (validation)
- **Final Loss**: ~0.8 (validation)
- **Training Speed**: ~0.6-0.8 iterations/second
- **Memory Usage**: ~8.35 GB peak
- **Total Time**: ~15 minutes for 500 iterations

## 🧪 Testing and Evaluation

### Base Model Testing
```bash
python src/inference.py
```
- Tests 50 examples from the test set
- Generates `data/results/base_model_test_results.json`
- Includes success rate and detailed results

### Fine-tuned Model Testing
```bash
python src/test_model.py --num_examples 50 --verbose
```
- Tests the fine-tuned model on the same examples
- Generates `data/results/test_model_results.json`
- Compares against base model performance

### Results Analysis
```bash
python src/analyze_results.py --verbose
```
- Compares base model vs fine-tuned model results
- Calculates improvement metrics
- Provides detailed performance breakdown

## 📚 Additional Resources

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX-LM GitHub](https://github.com/ml-explore/mlx-examples/tree/main/lora)
- [Gemma 3 Model Card](https://huggingface.co/google/gemma-3-1b-it)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Function Calling with LLMs](https://platform.openai.com/docs/guides/function-calling)

## 🤝 Contributing

This is a research project. For questions or issues:
1. Check the troubleshooting section above
2. Review the research paper in README.md
3. Examine the technical implementation in the `src/` directory

## 📄 License

This project is for academic research purposes. Please refer to the individual model and dataset licenses for usage terms.
