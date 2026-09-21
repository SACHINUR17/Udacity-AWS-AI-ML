# IMDB Sentiment Analysis with BERT Embeddings

A PyTorch implementation of a movie review sentiment classifier that combines pre-trained BERT embeddings with a custom transformer architecture.

## 📋 Overview

This project uses pre-trained BERT embeddings to classify IMDB movie reviews as either positive or negative. By leveraging BERT's language understanding capabilities, the model achieves high accuracy without needing to learn language patterns from scratch.

![loss_curves](https://github.com/user-attachments/assets/dad13f30-4c0a-4ce3-bedd-6c58d36258bc)
![accuracy_curve](https://github.com/user-attachments/assets/52022e2e-35d2-4308-bf68-4cb4feccc82f)

## ✨ Features

- Uses pre-trained BERT embeddings for rich word representations
- Custom transformer architecture for sentiment classification
- Achieves high accuracy compared to models trained from scratch
- Complete training pipeline with early stopping and model evaluation

## 🧠 How It Works

The model takes advantage of transfer learning by using:

1. **BERT Embeddings**: Instead of learning word meanings from scratch, we use BERT's pre-trained embeddings that already understand language from billions of texts
2. **Custom Transformer**: We add our own transformer layers on top of these embeddings to focus specifically on sentiment classification
3. **Attention Mechanism**: The model uses self-attention to understand relationships between words in the review

## 🚀 Getting Started

### Prerequisites

```
Python 3.8+
PyTorch 1.8+
Transformers 4.5+
pandas
matplotlib
tqdm
```

### Dataset

The model is trained on the augmented IMDB dataset.

### Training

1. Prepare the data:
```bash
python prepare_data.py
```

2. Tokenize the data:
```bash
python tokenizer.py
```

3. Train the model:
```bash
python train.py
```

4. Evaluate on test data:
```bash
python test.py
```

## 📈 Results

The model achieves significantly better performance compared to models with randomly initialized embeddings:

| Model Type | Test Accuracy |
|------------|---------------|
| Random Embeddings | ~77% |
| BERT Embeddings | ~85% |

## 📁 Project Structure

```
imdb-bert-sentiment/
├── prepare_data.py      # Data preparation script
├── tokenizer.py         # BERT tokenization script
├── model.py             # Model architecture definition
├── config.py            # Configuration and hyperparameters
├── dataloader.py        # Data loading utilities
├── train.py             # Training script
├── test_dataloader.py   # Test data preparation
├── test.py              # Model evaluation script
└── requirements.txt     # Project dependencies
```

## 🔍 Why BERT Embeddings?

BERT embeddings provide several advantages:

- **Pre-trained Knowledge**: They already understand language patterns, idioms, and sentiment expressions
- **Contextual Understanding**: The meaning of a word changes based on surrounding context
- **Handles Rare Words**: Better handling of unusual movie-specific terms
- **Bidirectional Context**: Understands relationships between words in both directions

## 📚 References

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
