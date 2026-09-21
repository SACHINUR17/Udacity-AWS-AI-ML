import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForMaskedLM
import random
from tqdm import tqdm
import os
import nltk
from nltk.tokenize import word_tokenize

class ContextualAugmenter:
    """
    Implement contextual augmentation using BERT masked language model.
    This approach replaces words with contextually similar words predicted by BERT.
    """
    
    def __init__(self, device=None):
        """Initialize the augmenter with BERT model and tokenizer."""
        if device is None:
            # Check for available devices in this order: CUDA → MPS → CPU
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("Using Apple Silicon MPS acceleration")
            else:
                self.device = torch.device("cpu")
                print("Using CPU (no GPU acceleration available)")
        else:
            self.device = device
            print(f"Using specified device: {self.device}")
        
        # Load pre-trained model and tokenizer
        print("Loading BERT model and tokenizer...")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.model.to(self.device)
        self.model.eval()
        
        # Define mask token ID
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
        
        # Maximum sequence length for BERT
        self.max_seq_length = 512
    
    def _get_word_replacements(self, text, word_to_replace, num_predictions=5):
        """Get contextual replacements for a specific word in the text."""
        # If text is too long, find the occurrence of the word and use a window around it
        if len(text.split()) > 100:  # Approximate threshold
            words = text.split()
            word_pos = -1
            
            # Find position of the word to replace
            for i, word in enumerate(words):
                if word.lower() == word_to_replace.lower():
                    word_pos = i
                    break
            
            if word_pos == -1:
                return []  # Word not found
            
            # Create a window around the word
            start_pos = max(0, word_pos - 50)
            end_pos = min(len(words), word_pos + 50)
            
            # Create a shorter text with the word in context
            text = ' '.join(words[start_pos:end_pos])
            
            # Adjust word_to_replace if it might have changed (e.g., with punctuation)
            if word_pos >= start_pos and word_pos < end_pos:
                word_to_replace = words[word_pos]
        
        # Tokenize the text
        tokens = self.tokenizer.tokenize(text)
        
        # If the text is still too long, truncate it to fit BERT's limit
        if len(tokens) > self.max_seq_length - 2:  # -2 for [CLS] and [SEP]
            # Find the word in tokens
            word_tokens = self.tokenizer.tokenize(word_to_replace)
            
            if not word_tokens:
                return []
                
            word_token = word_tokens[0]
            
            # Find the positions of the word token
            positions = [i for i, token in enumerate(tokens) if token == word_token]
            
            if not positions:
                return []
                
            # Use the first occurrence and create a window around it
            pos = positions[0]
            left_context = max(0, pos - 200)
            right_context = min(len(tokens), pos + 200)
            
            tokens = tokens[left_context:right_context]
        
        # Find the token(s) corresponding to the word to replace
        word_tokens = self.tokenizer.tokenize(word_to_replace)
        
        # If the word splits into multiple tokens, we'll only replace the first one
        # for simplicity (could be extended to handle multi-token replacements)
        if not word_tokens:
            return []
            
        word_token = word_tokens[0]
        
        # Find positions of the token in the tokenized text
        positions = [i for i, token in enumerate(tokens) if token == word_token]
        if not positions:
            return []
            
        # Randomly select one position to mask
        position = random.choice(positions)
        
        # Create a copy of tokens and replace the selected position with [MASK]
        masked_tokens = tokens.copy()
        masked_tokens[position] = '[MASK]'
        
        # Convert to input IDs and create attention mask
        inputs = self.tokenizer.encode_plus(
            " ".join(masked_tokens),
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Find position of mask token in input_ids
        mask_positions = (input_ids == self.mask_token_id).nonzero()
        if mask_positions.shape[0] == 0:
            return []
            
        mask_position = mask_positions[0, 1]
        
        # Generate predictions
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs.logits
        
        # Get top predictions for masked token
        probs = torch.nn.functional.softmax(predictions[0, mask_position], dim=0)
        top_k_weights, top_k_indices = torch.topk(probs, num_predictions)
        
        # Convert token IDs to tokens
        replacements = []
        for token_id in top_k_indices:
            token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
            
            # Skip special tokens and subword tokens (starting with ##)
            if token.startswith('##') or token in ['[CLS]', '[SEP]', '[MASK]', '[PAD]']:
                continue
                
            # Skip if the replacement is the same as the original
            if token.lower() == word_token.lower():
                continue
                
            replacements.append(token)
        
        return replacements
    
    def augment(self, text, percent=0.15, top_n=5, max_length=10000):
        """
        Augment text by replacing a percentage of words with contextual predictions.
        
        Args:
            text (str): Input text to augment
            percent (float): Percentage of words to replace
            top_n (int): Number of top predictions to consider
            max_length (int): Maximum text length to process
            
        Returns:
            str: Augmented text
        """
        # If text is too long, truncate it (this is a safety measure)
        if len(text) > max_length:
            text = text[:max_length]
        
        # Tokenize into words
        words = word_tokenize(text)
        
        # Determine number of words to replace
        n_to_replace = max(1, int(len(words) * percent))
        
        # Create a copy of words
        new_words = words.copy()
        
        # Get indices of words to potentially replace (exclude stop words, short words)
        valid_indices = []
        for i, word in enumerate(words):
            # Skip words that are too short
            if len(word) <= 3:
                continue
                
            # Skip words that are not alphabetic
            if not word.isalpha():
                continue
                
            valid_indices.append(i)
        
        # Shuffle and select indices to replace
        if valid_indices:
            random.shuffle(valid_indices)
            indices_to_replace = valid_indices[:n_to_replace]
            
            # Replace selected words
            for idx in indices_to_replace:
                word_to_replace = words[idx]
                
                # Get contextual replacements
                replacements = self._get_word_replacements(text, word_to_replace, top_n)
                
                # Replace if we found valid replacements
                if replacements:
                    new_words[idx] = random.choice(replacements)
        
        # Join words back into text
        augmented_text = ' '.join(new_words)
        
        return augmented_text


def augment_imdb_with_contextual(input_file, output_file, sample_fraction=0.2, augmentations_per_sample=1, 
                                batch_size=1, max_reviews_length=10000):
    """
    Augment the IMDB dataset with contextual augmentation.
    
    Args:
        input_file: Path to the original IMDB CSV file
        output_file: Path to save the augmented dataset
        sample_fraction: Fraction of dataset to augment
        augmentations_per_sample: Number of augmentations to create per sample
        batch_size: Process this many reviews at once (for progress reporting)
        max_reviews_length: Skip reviews longer than this to avoid memory issues
    """
    # Check if the input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Load the IMDB dataset
    print(f"Loading dataset from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Check for required columns
    if 'review' not in df.columns or 'label' not in df.columns:
        raise ValueError("The dataset must contain 'review' and 'label' columns!")
    
    # Display dataset statistics
    print(f"Original dataset size: {len(df)} reviews")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    
    # Initialize the augmenter
    augmenter = ContextualAugmenter()
    
    # Sample a subset of the data to augment
    n_samples = int(len(df) * sample_fraction)
    print(f"Sampling {n_samples} reviews for augmentation...")
    
    # Ensure we have equal representation of both classes
    df_pos = df[df['label'] == 1].sample(n=n_samples//2, random_state=42)
    df_neg = df[df['label'] == 0].sample(n=n_samples//2, random_state=42)
    df_to_augment = pd.concat([df_pos, df_neg], ignore_index=True)
    
    # Filter out extremely long reviews
    df_to_augment = df_to_augment[df_to_augment['review'].str.len() < max_reviews_length]
    print(f"Using {len(df_to_augment)} reviews after filtering by length")
    
    # Initialize an empty list to store augmented data
    augmented_data = []
    
    # Augment the sampled data
    print("\nGenerating contextual augmentations...")
    
    # Process in batches for better progress reporting
    num_batches = len(df_to_augment) // batch_size + (1 if len(df_to_augment) % batch_size > 0 else 0)
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(df_to_augment))
        
        batch_df = df_to_augment.iloc[start_idx:end_idx]
        
        for _, row in batch_df.iterrows():
            review = row['review']
            label = row['label']
            
            # Create multiple augmentations per sample
            for i in range(augmentations_per_sample):
                try:
                    augmented_review = augmenter.augment(review, percent=0.15)
                    
                    # Only add if the augmentation is different from the original
                    if augmented_review != review:
                        augmented_data.append({
                            'review': augmented_review,
                            'label': label,
                            'technique': f'contextual_aug_{i+1}'
                        })
                except Exception as e:
                    print(f"Error during augmentation: {str(e)[:100]}...")
                    continue
    
    # Add technique column to original data
    df_original = df.copy()
    df_original['technique'] = 'original'
    
    # Create a DataFrame with augmented data
    if augmented_data:
        df_augmented = pd.DataFrame(augmented_data)
        
        # Combine original and augmented data
        df_combined = pd.concat([df_original, df_augmented], ignore_index=True)
        
        # Display augmented dataset statistics
        print(f"\nAugmented dataset size: {len(df_combined)} reviews")
        print(f"Added {len(df_augmented)} augmented samples")
        print(f"Percentage increase: {(len(df_combined) - len(df)) / len(df) * 100:.2f}%")
        
        # Save the augmented dataset
        print(f"Saving augmented dataset to {output_file}...")
        df_combined.to_csv(output_file, index=False)
        print("Done!")
        
        # Display a few examples of original and augmented reviews
        print("\nExamples of original and augmented reviews:")
        for i in range(1, augmentations_per_sample + 1):
            technique = f'contextual_aug_{i}'
            aug_examples = df_combined[df_combined['technique'] == technique].head(2)
            
            for _, row in aug_examples.iterrows():
                print(f"\nAugmented (contextual):")
                print(row['review'][:200] + "..." if len(row['review']) > 200 else row['review'])
                
                # Find the original review
                original_idx = df_to_augment.loc[
                    (df_to_augment['label'] == row['label']) &
                    (df_to_augment.index % len(df_to_augment) == 
                     row.name % len(df_to_augment))
                ].index
                
                if not original_idx.empty:
                    original_review = df.loc[original_idx[0], 'review']
                    print("\nOriginal:")
                    print(original_review[:200] + "..." if len(original_review) > 200 else original_review)
    else:
        print("No augmented data was generated. Saving original dataset only.")
        df_original.to_csv(output_file, index=False)


if __name__ == "__main__":
    # Set the paths for input and output files
    input_file = "imdb_train_dataset.csv"
    output_file = "imdb_train_contextual_augmented.csv"
    
    # Set the fraction of the dataset to augment
    sample_fraction = 0.2  # Augment 20% of the dataset
    
    # Set number of augmentations per sample
    augmentations_per_sample = 1
    
    # Set batch size for progress reporting
    batch_size = 5
    
    # Set maximum review length (characters) to process
    max_reviews_length = 8000
    
    # Run the augmentation
    augment_imdb_with_contextual(
        input_file=input_file,
        output_file=output_file,
        sample_fraction=sample_fraction,
        augmentations_per_sample=augmentations_per_sample,
        batch_size=batch_size,
        max_reviews_length=max_reviews_length
    )