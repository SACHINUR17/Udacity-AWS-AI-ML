
# Back-Translation Data Augmentation Tool

## Overview

This tool performs back-translation data augmentation for text classification tasks. Back-translation involves translating text from a source language to a target language and then back to the source language, creating paraphrased variations that preserve the original meaning while introducing linguistic diversity.

## Features

- Translates text between multiple language pairs using pre-trained neural machine translation models
- Supports batch processing to efficiently handle large datasets
- Provides detailed progress tracking with tqdm progress bars
- Creates augmented datasets with original and back-translated samples
- Automatically handles GPU acceleration when available
- Saves intermediate results to prevent data loss during long processing runs
- Configurable parameters for sample selection and batch size

## Requirements

- Python 3.6+
- PyTorch
- Transformers (Hugging Face)
- SentencePiece
- pandas
- tqdm
- GPU for acceleration (I used Google Colab)

## Usage

1. Mount your Google Drive to access and save files:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. Set up input and output paths:
   ```python
   input_file = "/content/drive/MyDrive/backtranslation/imdb_train_dataset.csv"
   output_dir = "/content/drive/MyDrive/backtranslation/output"
   ```

3. Configure the augmentation parameters:
   ```python
   target_langs = ["fr", "de"]  # Languages to use for back-translation
   sample_fraction = 0.85  # Percentage of dataset to augment
   batch_size = 4  # Number of texts to process at once
   ```

4. Run the augmentation:
   ```python
   augment_imdb_with_back_translation(
       input_file=input_file,
       output_dir=output_dir,
       target_langs=target_langs,
       sample_fraction=sample_fraction,
       batch_size=batch_size
   )
   ```

## How It Works

1. The tool initializes pre-trained translation models from Helsinki-NLP's OPUS-MT for each language pair.
2. It loads your dataset and samples a configurable fraction of the data for augmentation.
3. For each target language, it:
   - Translates the original text to the target language
   - Translates it back to the source language
   - Adds the back-translated version to the augmented dataset
4. The tool saves the combined dataset (original + augmented) to the specified output directory.
   
Note : The code displays random examples of back-translated reviews paired with unrelated original reviews, making it impossible to see the true effect of back-translation. Add an ```original_index column``` to track which original review each translation came from, then use this index to retrieve the correct original review when displaying examples for comparison.

## Example

Using the tool on an IMDB dataset:

```python
# Configure parameters
input_file = "/content/drive/MyDrive/backtranslation/imdb_train_dataset.csv"
output_dir = "/content/drive/MyDrive/backtranslation/output"
target_langs = ["fr", "de"]
sample_fraction = 0.85
batch_size = 4

# Run augmentation
augment_imdb_with_back_translation(
    input_file=input_file,
    output_dir=output_dir,
    target_langs=target_langs,
    sample_fraction=sample_fraction,
    batch_size=batch_size
)
```

This will create an augmented dataset with the original reviews plus their French and German back-translated versions.

## Tips

- The process can be time-consuming for large datasets - use a GPU-enabled runtime
- Adjust the `batch_size` based on your available memory
- Use the `sample_fraction` parameter to control the size of the augmented dataset
- If you encounter memory issues, try reducing the maximum review length (`max_review_length`)

## Output

The tool generates a CSV file with the following columns:
- `review`: The text content
- `label`: The original label
- `technique`: The augmentation technique used (original, back_translation_fr, etc.)

# Example Translation Results

Below are examples of back-translations from the output of the back-translation augmentation tool:

## Example 1

**Back-translated (French):**
> This film is an eye-opener for those who can only the glamorous lifestyles of the stars. It tells you how people who would like to do good are not able to do it. Moreover, the scene of the bomb explosion is very real.

**Back-translated (German):**
> This film is an eye opener for those who can only see the glamorous lifestyles of the stars. It tells you how people who want to do good are not able to. Plus the bomb explosion scene is very real.

## Example 2

**Back-translated (French):**
> She is perfect. Her inexperience of action actually works in her favor. We have never seen her before, so it really looks like her story. She also faces a real hardness. It works against her, but because we've never seen her before, she comes across as a tough person rather than an actress playing a tough person.

**Back-translated (German):**
> She is perfect. Her acting inexperience actually works in her favor. We've never seen her before, so it really feels like her story. She also brings real toughness. But that works against her because we've never seen her before, she comes across as a tough person rather than an actress playing a tough person.

## Observations

As you can see from these examples, back-translation creates interesting variations:

1. **Semantic Diversity**: The back-translations often preserve the general sentiment but express it with different vocabulary and phrasing.

2. **Content Divergence**: In some cases, the back-translation significantly diverges from the original content, as seen in Example 1. This is likely due to the multi-step translation process occasionally missing context or meaning.

3. **Linguistic Variation**: Different target languages produce different types of variations, with subtle differences between the French and German back-translations.

4. **Structural Changes**: Back-translation often reorganizes sentence structure while maintaining the overall meaning.

These variations can be valuable for data augmentation in natural language processing tasks, as they help models learn to recognize the same underlying meaning expressed in different ways.
