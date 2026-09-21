# Transformer-Based IMDB Sentiment Analysis

This repository contains a PyTorch implementation of a simple transformer model for sentiment analysis on the IMDB movie review dataset. The project demonstrates how to build, train, and evaluate a lightweight transformer model from scratch, as well as how to improve performance through various techniques including data augmentation and model scaling.

## Project Overview

The goal of this project is to develop a transformer-based model that can classify movie reviews as either positive or negative. Unlike approaches that rely on pre-trained language models like BERT, this project builds a customizable transformer from the ground up, allowing for flexible experimentation with the architecture while keeping computational requirements modest.

### Key Features

- Custom transformer model implemented in PyTorch
- Configurable architecture (embedding size, number of layers, heads, etc.)
- Data preparation pipeline for the IMDB dataset
- Advanced text augmentation techniques
- Comprehensive evaluation and visualization tools
- Support for Apple Silicon GPU acceleration (MPS)

## Model Architecture

The model consists of a simple transformer architecture with the following components:
- Embedding layer
- Positional encoding
- Transformer encoder blocks
- Linear output layer

The architecture is configurable through various hyperparameters:
- `VOCAB_SIZE`: Size of the vocabulary (derived from the tokenizer)
- `EMBED_DIM`: Dimension of the embedding vectors
- `NUM_HEADS`: Number of attention heads in each transformer layer
- `NUM_LAYERS`: Number of transformer encoder layers
- `MAX_SEQ_LENGTH`: Maximum sequence length for input text

## Dataset

The project uses the IMDB movie review dataset ([Maas et al., 2011](https://ai.stanford.edu/~amaas/data/sentiment/)), which contains 50,000 movie reviews labeled as positive or negative. The dataset is pre-processed and split into training and testing sets. Each set is in a separate folder within which there are sub folders called "pos" and "neg". The ```data_prep.py``` code creates the labeled dataframe that can be tokenized.
Contextual augmentation is also used to increase the quantity of the dataset for increasing test accuracy for same model architecture. Details are described [here](https://github.com/agme2019/Movie-Review-Sentiment-Analysis-Using-Transformers/blob/main/Simple/Contextual%20Augmentation.md).

## Getting Started

### Prerequisites

```bash
pip install torch transformers pandas numpy matplotlib scikit-learn tqdm nltk
```

### Project Structure

```
├── config.py                  # Model configuration and hyperparameters
├── model.py                   # Transformer model definition
├── data_prep.py               # IMDB dataset preparation
├── tokenizer.py               # Text tokenization
├── dataloader.py              # PyTorch data loader
├── train.py                   # Training script
├── validation.py              # Validation script
├── prepare_test_data.py       # Test dataset preparation
├── test_dataloader.py         # Load test data
├── test.py                    # Evaluation script
├── infer_single_review.py     # Inference on a single review
├── gpu_contextual_aug.py      # Contextual text augmentation
└── test_confusion_matrix.py   # Confusion matrix visualization
```

### Training the Model

1. Prepare the dataset:
```bash
python data_prep.py
```

2. Tokenize the dataset:
```bash
python tokenizer.py
```

3. Train the model:
```bash
python train.py
```

For longer training with progress bar:
```bash
python train_m4pro.py
```

4. Evaluate the model:
```bash
python test.py
```

### Data Augmentation

The project implements contextual augmentation techniques to improve model performance. To use data augmentation:

1. Run the augmentation script:
```bash
python gpu_contextual_aug.py
```

2. Tokenize the augmented dataset:
```bash
python tokenizer.py
```

3. Train with the augmented data:
```bash
python train_m4pro.py
```

## Experimental Results

We experimented with various model configurations and training strategies. Here's a summary of the results:

| Model    | Layers | Embed Dim | Heads | Seq Length | Parameters | Train Loss | Val Acc | Test Acc |
|----------|--------|-----------|-------|------------|------------|------------|---------|----------|
| TX-1     | 2      | 128       | 4     | 100        | 4.3M       | 0.091      | 0.994   | 0.754    |
| TX-2     | 4      | 128       | 4     | 100        | 4.7M       | 0.055      | 0.996   | 0.756    |
| TX-3     | 4      | 256       | 4     | 100        | 10.9M      | 0.048      | 0.997   | 0.762    |
| TX-4     | 4      | 256       | 8     | 100        | 10.9M      | 0.049      | 0.991   | 0.761    |
| TX-5     | 4      | 512       | 8     | 100        | 28.23M     | 0.021      | 0.999   | 0.755    |
| TX-aug   | 4      | 256       | 4     | 100        | 10.9M      | 0.031      | 1.000   | 0.743    |
| TX-aug-new2 | 4   | 256       | 4     | 512        | 10.9M      | 0.017      | 1.000   | 0.717    |

### Key Observations

1. **Scaling Impact**: Increasing the model size (layers and embedding dimension) generally improved performance up to a point, with TX-3 showing the best test accuracy.

2. **Overfitting**: Models with very high validation accuracy (0.99+) but lower test accuracy (~0.75) suggest some overfitting to the training data.

3. **Data Augmentation**: While augmentation helped achieve perfect validation accuracy, it didn't consistently improve test accuracy, suggesting that the augmentation techniques may not have provided sufficient diversity for better generalization. TX-aug was fed with additional 20% augmented data, whereas TX-aug-new2 was fed with additional 80% augmented data.

4. **Sequence Length**: Increasing the maximum sequence length from 100 to 512 tokens (TX-aug-new2) provided a small improvement in test accuracy compared to the equivalent model with shorter sequences.

## Advanced Features

### Contextual Augmentation

The project implements a BERT-based contextual augmentation technique that replaces words with contextually similar alternatives. This helps the model learn more robust representations by exposing it to varied but semantically equivalent inputs.

```python
# Example of contextual augmentation
augmenter = ContextualAugmenter()
augmented_text = augmenter.augment(text, percent=0.15)
```

### Apple Silicon Support

The code includes support for training on Apple Silicon GPUs using the MPS (Metal Performance Shaders) backend:

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```
# Limitations and Performance Analysis

## The 75% Test Accuracy Ceiling

A consistent pattern in our experiments is that test accuracy remains around 75% (ranging from 74.3% to 76.2%) regardless of various architecture changes and training strategies. This ceiling effect warrants analysis:

1. **Model Capacity vs. Dataset Complexity**:
   - While increasing model size (TX-1 → TX-5) showed some initial gains, larger models eventually hit diminishing returns and even slight performance degradation.
   - This suggests the primary limitation may not be model capacity but rather the inherent complexity and noise in the dataset.

2. **Training-Test Distribution Mismatch**:
   - The near-perfect validation accuracy contrasted with modest test performance indicates a distribution mismatch between training and test sets.
   - This could be due to different writing styles, vocabulary usage, or review characteristics between the two sets.

3. **Information Bottleneck**:
   - The token-level representations may be losing important contextual information during the dimensionality reduction that occurs in the feed-forward layers.
   - The 75% ceiling might represent the limit of what can be extracted from purely local features without more sophisticated language understanding.

4. **Tokenization Limitations**:
   - Using BERT's tokenizer without its pre-trained weights means we lose the semantic information BERT captures, while still inheriting its subword fragmentation approach.
   - This may create a disconnect between the tokenization strategy and our model's learning capabilities.

5. **Overfitting to Spurious Correlations**:
   - The high training and validation accuracy suggests the model might be learning superficial patterns in the training data that don't generalize to the test distribution.
   - Simple sentiment classification can often rely on the presence of certain keywords rather than understanding review context.
  
An interesting finding is that the model with max sequence length = 512 actually performed worse (71.7% test accuracy) despite achieving a better training loss (0.017) and excellent validation accuracy (0.999). This reinforces the observations about overfitting. The longer sequence length model seems to be overfitting even more severely than the models with shorter sequence lengths. This could be happening for several reasons:

+ **Increased model complexity** - While the parameter count remains the same (10.9M), the longer sequences mean the model is processing more tokens and has more opportunities to find spurious correlations in the training data.

+ **Sparse information density** - Movie reviews might not have relevant sentiment information distributed throughout the entire 512-token sequence, so the model could be learning noise from the additional tokens.

+  **Training dynamics** - The lower training loss (0.017) coupled with worse test performance (71.7%) is a classic sign of overfitting, where the model memorizes training examples rather than learning generalizable patterns.

+  **Attention dilution** - With longer sequences, attention might be diluted across more tokens, making it harder for the model to focus on the most sentiment-relevant parts of the text.

It shows that simply increasing sequence length isn't always beneficial, and can sometimes be detrimental to generalization performance. 

## Comparison with State-of-the-Art

For context, state-of-the-art pre-trained models like BERT and RoBERTa typically achieve 94-96% accuracy [citations](https://github.com/agme2019/Movie-Review-Sentiment-Analysis-Using-Transformers/blob/main/Simple/BERT-RoBERTa_citations.md) on IMDB sentiment classification. Our ~75% accuracy with a smaller custom model represents the trade-off between:

- Lower computational requirements
- No dependency on large pre-trained models
- Simpler architecture and inference
- Reduced accuracy compared to SOTA approaches

This performance gap highlights the value that pre-training on large corpora brings to transformer models, enabling them to capture deeper semantic understanding that smaller models trained from scratch cannot easily achieve.

## Future Work

- Implement more sophisticated data augmentation techniques
- Experiment with different attention mechanisms
- Add support for additional datasets
- Implement model distillation from larger pre-trained models
- Explore knowledge distillation techniques

## References

- Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning word vectors for sentiment analysis. In Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The IMDB dataset creators
- PyTorch and Hugging Face teams for their excellent libraries
