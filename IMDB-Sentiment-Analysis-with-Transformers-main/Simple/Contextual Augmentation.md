# Contextual Augmentation for Sentiment Analysis

This module implements BERT-based contextual augmentation for enhancing sentiment analysis datasets. Unlike traditional text augmentation methods that rely on simple word replacement or synonyms, this approach leverages BERT's masked language modeling capabilities to generate context-aware substitutions.

## How It Works

The contextual augmentation process follows these steps:

1. **Word Selection**: Identifies candidate words in the text for replacement (excluding stop words, short words, and non-alphabetic tokens)
2. **Contextual Masking**: Replaces selected words with the `[MASK]` token
3. **BERT Prediction**: Uses BERT's masked language model to predict contextually appropriate replacements
4. **Substitution**: Replaces the original word with one of the top predictions

This approach preserves the semantic meaning and sentiment of the text while introducing lexical diversity.

## Key Features

- **Context-Aware**: Unlike rule-based substitutions, replacements consider the surrounding context
- **Sentiment Preservation**: Maintains the original sentiment while introducing variation
- **Configurable Intensity**: Adjustable percentage of words to replace (default: 15%)
- **GPU Acceleration**: Supports CUDA, Apple Silicon MPS, and falls back to CPU
- **Length Handling**: Implements a sliding window approach for long reviews

## Usage Example

```python
from gpu_contextual_aug import ContextualAugmenter, augment_imdb_with_contextual

# Simple example on a single text
augmenter = ContextualAugmenter()
original_text = "This movie was fantastic and I enjoyed every minute of it!"
augmented_text = augmenter.augment(original_text, percent=0.15)

# Augment entire dataset
augment_imdb_with_contextual(
    input_file="imdb_train_dataset.csv",
    output_file="imdb_train_contextual_augmented.csv",
    sample_fraction=0.2,           # Augment 20% of the dataset
    augmentations_per_sample=1,    # Create 1 variation per review
    batch_size=5,                  # Process 5 reviews at a time
    max_reviews_length=8000        # Skip reviews longer than 8000 chars
)
```

## Implementation Details

The `ContextualAugmenter` class:
1. Initializes BERT tokenizer and masked language model
2. Implements a sliding window for processing long texts
3. Identifies appropriate word replacements based on context
4. Handles tokenization edge cases (subword tokens, special characters)

The `augment_imdb_with_contextual` function:
1. Loads and processes the IMDB dataset
2. Selects a balanced subset for augmentation
3. Generates augmentations in batches
4. Combines original and augmented data
5. Saves the enhanced dataset

## Performance Impact

In our experiments with a custom transformer model for IMDB sentiment analysis:

| Model | Training Loss | Validation Accuracy | Test Accuracy |
|-------|--------------|---------------------|---------------|
| Base (TX-3) | 0.048 | 0.997 | 0.762 |
| With Augmentation | 0.031 | 1.000 | 0.743 |

While augmentation helped achieve perfect validation accuracy, it didn't improve test accuracy. This suggests the augmentation may be increasing the model's capacity to memorize training patterns rather than improving generalization. However, the technique shows promise and could be refined with:

1. More diverse augmentation strategies
2. Better selection of words to replace
3. Integration with other regularization techniques

## References

- Kobayashi, S. (2018). Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relations. NAACL.
- Wu, X., Lv, S., Zang, L., Han, J., & Hu, S. (2019). Conditional BERT Contextual Augmentation. ICCS.
