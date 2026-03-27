# EEP 567 Lab 8: Malware Detection with Transformer Models

## Overview

This lab teaches you how to detect malware activities using **BERT**, a Transformer-based Natural Language Model. You'll work with the **ADFA-LD dataset** containing system call logs from both benign and malicious processes. The lab walks you through the complete machine learning pipeline: preprocessing, feature extraction, visualization, and classification.

## What You'll Learn

- **System Call Log Analysis**: Understand how system call traces represent process behavior
- **Text Preprocessing**: Convert numeric system call logs into textual representations
- **Run-Length Encoding**: Compress repetitive sequences in malware traces
- **BERT Feature Extraction**: Use transformer models to extract semantic embeddings
- **Dimensionality Reduction**: Project 768-dimensional embeddings to 3D using t-SNE
- **Classification**: Train logistic regression and neural network classifiers
- **Binary vs. Multi-class Detection**: Distinguish between benign/malicious and specific attack types

## Quick Start

### Prerequisites
- Python 3.8 or higher
- At least 8GB RAM (16GB+ recommended for GPU-free execution)
- Optional: NVIDIA GPU with CUDA support or Apple silicon with MPS support for faster inference

### Installation

1. **Install Dependencies**
   ```bash
   pip install torch accelerate matplotlib numpy scikit-learn transformers tqdm
   ```
   
   For NVIDIA GPU support (optional):
   ```bash
   # Check https://pytorch.org for the correct command for your CUDA version
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Verify Installation**
   - Open the notebook in Jupyter or VS Code
   - Run the first two cells to install dependencies
   - The notebook will automatically detect available hardware (GPU or CPU)

### Running the Lab
- Open `lab-8-student-20260222.ipynb` in Jupyter or VS Code
- Execute cells sequentially from top to bottom
- Complete the TODO sections as indicated
- Expected runtime: 15-30 minutes (varies based on hardware and dataset size)

## ML Pipeline Overview

```
Raw System Call Logs (integers)
         ↓
  [Preprocessing]
  - Convert to syscall names
  - Run-length encode repetitions
         ↓
  Text Representations (simple & concise)
         ↓
  [BERT Feature Extraction]
  - Tokenize with BertTokenizer
  - Extract [CLS] token embeddings
  - Output: 768-dimensional vectors
         ↓
  BERT Embeddings (N × 768)
         ↓
  [Dimensionality Reduction]
  - Apply t-SNE to 3D
         ↓
  3D Feature Space (Visualizable)
         ↓
  [Classification]
  - Train: Logistic Regression or MLP
  - Evaluate on binary (benign/malicious) and full (7 classes) tasks
```

## Dataset

### ADFA-LD Structure

| Category | Location | Description |
|----------|----------|-------------|
| **Benign Logs** | `ADFA-LD/Training_Data_Master/*.txt` | 50 system call traces from normal activities |
| **Attack Logs** | `ADFA-LD/Attack_Data_Master/<type>_x/*.txt` | 10 traces per attack type (60 total) |

### Attack Types

| Attack Type | Folder | Description |
|-------------|--------|-------------|
| Adduser | `Adduser_1` to `Adduser_10` | Unauthorized user creation |
| Hydra FTP | `Hydra_FTP_1` to `Hydra_FTP_10` | Brute-force FTP attacks |
| Hydra SSH | `Hydra_SSH_1` to `Hydra_SSH_10` | Brute-force SSH attacks |
| Java Meterpreter | `Java_Meterpreter_1` to `Java_Meterpreter_10` | Java-based remote shell |
| Meterpreter | `Meterpreter_1` to `Meterpreter_10` | Remote shell exploitation |
| Web Shell | `Web_Shell_1` to `Web_Shell_10` | Web-based command execution |

### Log Format

Raw logs contain sequences of system call numbers (integers):
```
4 4 4 221 3 27 9 0 3 3 21 3 4 4 4 4 ...
```

System call mapping comes from `ADFA-LD/syscall-table.csv` (format: `__NR_name,number`)

## Student Exercises

### Exercise 1: Log Preprocessing (Cell 9)

**Task**: Preprocess raw system call logs into two text representations.

**What you need to implement**:
1. Parse raw log string into integer array
2. Create **simple textual log**: Replace each syscall number with its name from `syscall_map`
3. Create **concise textual log**: Compress consecutive identical syscalls using format `name * count`

**Example transformation**:
```
Raw:     4 4 4 221 3 27 9
Simple:  read read read execve open lseek link
Concise: read * 3 execve open lseek link
```

**Hints**:
- `raw_log.split(" ")` → convert to list of integers
- Use `syscall_map.get(number, str(number))` to handle unknown syscalls
- Track previous syscall in a loop to detect transitions (for concise encoding)

### Exercise 2: BERT Feature Extraction (Cell 18)

**Task**: Extract embedding vectors from preprocessed text logs using BERT.

**What you need to implement**:
1. Create a `DataLoader` to batch process logs
2. Tokenize batches with `BertTokenizer` (max 512 tokens, with padding)
3. Run forward pass through BERT model
4. Extract the `[CLS]` token embedding (first token) as the sequence representation

**Key Points**:
- Use `torch.inference_mode()` decorator (already present) to save memory
- Move tokenized inputs to `model_device` before passing to model
- Extract embeddings from `outputs.last_hidden_state[:, 0, :]` (index 0 = CLS token)
- Convert to CPU and float32 before appending to list

**Expected Output**:
- `text_embeds`: shape (N, 768) where N = number of logs
- `concise_embeds`: shape (N, 768) from concise preprocessed logs

## Expected Results

After completing the lab, you should observe:

### Classification Performance (Approximate Baselines)

| Method | Binary Accuracy | Full-Class Accuracy |
|--------|-----------------|---------------------|
| Logistic Regression (Textual) | 85-95% | 60-75% |
| Logistic Regression (Concise) | 85-95% | 60-75% |
| MLP (Textual) | 85-95% | 70-85% |
| MLP (Concise) | 85-95% | 70-85% |

**Note**: Actual results may vary due to randomness in train-test splits and neural network initialization.

### Feature Space Observations

- **Benign vs. Malicious Separation**: Clear clustering visible in t-SNE plots
- **Attack Type Overlap**: Different attack types cluster closely together (harder to distinguish)
- **Representation Comparison**: Textual and concise embeddings produce similar distributions

## Common Issues & Troubleshooting

### 1. **Out of Memory (OOM) Errors**

**Problem**: `torch.cuda.OutOfMemoryError` or `RuntimeError: Unable to allocate X GiB`

**Solutions**:
- Reduce `batch_size` in `make_bert_embeds()` from 16 to 8 or 4
- Use CPU instead of GPU (slower but uses system RAM)
- Reduce `n_samples` if loading subset of data

### 2. **GPU Not Detected (Falls Back to CPU)**

**Problem**: Code runs on CPU despite having a GPU

**Solutions for NVIDIA GPU**:
```python
import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))  # Shows GPU name
```
- Verify PyTorch was installed with CUDA support: `pip list | grep torch`
- Reinstall with correct CUDA version from https://pytorch.org

**Solutions for Apple Silicon (MPS)**:
```python
from torch.backends.mps import is_available
print(is_available())  # Should print True
```
- Requires MacOS 12.3+ and PyTorch 1.12+
- Reinstall: `pip install --upgrade torch`

### 3. **Data Loading Error: Files Not Found**

**Problem**: `FileNotFoundError: [Errno 2] No such file or directory`

**Solutions**:
- Verify you're running notebook from the correct directory (should be `lab-8`)
- Check that `ADFA-LD/` folder exists in the current directory
- Verify `ADFA-LD/syscall-table.csv` exists
- Verify `ADFA-LD/Training_Data_Master/*.txt` and `ADFA-LD/Attack_Data_Master/` folders exist

### 4. **BERT Model Download Fails**

**Problem**: `ConnectionError` when loading `bert-base-uncased`

**Solutions**:
- Check internet connection
- Hugging Face models are cached locally after first download
- Increase timeout: Set environment variable `export TRANSFORMERS_TIMEOUT=120`
- Download manually and specify local path (advanced)

### 5. **Tokenizer/Model Version Mismatch**

**Problem**: `RuntimeError: Error(s) in loading state_dict`

**Solutions**:
- Reinstall transformers: `pip install --upgrade transformers`
- Clear cache: `rm -rf ~/.cache/huggingface/`
- Verify versions match: `pip list | grep -E "torch|transformers"`

### 6. **t-SNE Algorithm Doesn't Converge**

**Problem**: Takes extremely long (30+ minutes) or produces uniform plots

**Solutions**:
- t-SNE is computationally expensive for large datasets; this is normal
- Try reducing dataset size for testing
- Results should still be reasonable even if convergence warnings appear
- For faster dimensionality reduction, try `PCA` instead:
  ```python
  from sklearn.decomposition import PCA
  pca = PCA(n_components=3)
  features_3d = pca.fit_transform(embeddings)
  ```

## References

1. **ADFA-LD Dataset**: https://research.unsw.edu.au/projects/adfa-ids-datasets
2. **BERT Paper**: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2018)
3. **Hugging Face Transformers Documentation**: https://huggingface.co/docs/transformers/
4. **PyTorch Documentation**: https://pytorch.org/docs/stable/index.html
5. **Scikit-learn Classification Metrics**: https://scikit-learn.org/stable/modules/model_evaluation.html
6. **t-SNE Visualization**: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

## Tips for Success

- **Understand the data first**: Examine raw logs and syscall mappings before preprocessing
- **Start simple**: Get textual preprocessing working before implementing concise encoding
- **Check intermediate outputs**: Print examples of preprocessed logs to verify correctness
- **Monitor BERT tokenization**: The 512-token limit may truncate long logs; concise encoding helps
- **Experiment with classifiers**: Try both logistic regression and MLP to see differences
- **Visualize results**: The t-SNE plots reveal how well classes separate