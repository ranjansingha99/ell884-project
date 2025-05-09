# Counterspeech Generation Model

## Overview
This project fine-tunes a transformer-based approach to generate counterspeech responses to hate speech using the DIALOCONAN dataset. It includes data preprocessing, training, evaluation, and visualization of results.

## Requirements
- Python 3.11
- Libraries: `transformers`, `datasets`, `torch`, `pandas`, `numpy`, `nltk`, `sacrebleu`, `rouge-score`, `bert-score`
- GPU (NVIDIA Tesla T4 recommended)

## Dataset
- **DIALOCONAN**: Contains dialogues with hate speech (HS) and counterspeech (CN) across various target groups.
- Path: `/kaggle/input/dialoconan/DIALOCONAN.csv`

## Usage
1. **Setup**: Install dependencies and load libraries as shown in the notebook.
2. **Preprocessing**: Create dialogue pairs and split into train/validation/test sets.
3. **Training**: Fine-tune the approach with 5 epochs, batch size of 2, and FP16 precision.
4. **Evaluation**: Compute BLEU, ROUGE-L, and BERTScore on the test set.
5. **Outputs**:
   - Model checkpoints: `/kaggle/working/counterspeech-final`
   - Logs: `/kaggle/working/training_logs.txt`
   - Evaluation results: `/kaggle/working/evaluation_results.json`
   - Sample predictions: `/kaggle/working/sample_predictions.csv`

## Key Features
- Custom metrics callback for validation (BLEU, ROUGE-L, BERTScore).
- Visualization of training loss.
- Inference function to generate counterspeech for new hate speech inputs.

## Example
```python
sample_hs = "All migrants are criminals and should be deported!"
sample_target = "MIGRANTS"
generated_cn = generate_counterspeech(sample_hs, sample_target)
print(f"Hate Speech: {sample_hs}")
print(f"Generated Counterspeech: {generated_cn}")
```

## Notes
- Ensure internet access for dependency installation on Kaggle.
- Training logs and evaluation metrics are saved for analysis.
- Adjust `max_samples` in the custom metrics callback for faster validation.