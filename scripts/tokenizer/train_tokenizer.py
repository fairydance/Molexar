import argparse
import os

from molexar.modeling import MolexarConfig
from molexar.tokenizer import train_and_save_tokenizer

def main():
    parser = argparse.ArgumentParser(description="Train the Molexar Fragment-SELFIES tokenizer")
    
    parser.add_argument(
        "--data_file", 
        type=str, 
        required=True,
        help="Path to Fragment-SELFIES data. If IDs are present after whitespace, only the first field is used."
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="models/tokenizer",
        help="Directory where the tokenizer.json will be saved"
    )
    
    parser.add_argument(
        "--vocab_size", 
        type=int, 
        default=1000,
        help="Retained for CLI visibility; WordLevel training keeps all observed Fragment-SELFIES tokens"
    )

    args = parser.parse_args()

    # 1. Validation
    if not os.path.exists(args.data_file):
        print(f"Error: Data file not found at {args.data_file}")
        print("Please ensure you have a text file with Fragment-SELFIES strings for training.")
        return

    # 2. Setup Config
    config = MolexarConfig()
    
    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    print(f"--- Starting Tokenizer Training ---")
    print(f"Data Source: {args.data_file}")
    print(f"Vocab Size:  all observed Fragment-SELFIES tokens")
    print(f"Output Dir:  {args.output_dir}")
    
    # 3. Train
    # Note: We pass the config to ensure Special Tokens (Conditions) are added
    tokenizer = train_and_save_tokenizer(
        data_files=[args.data_file], 
        save_path=args.output_dir, 
        config=config
    )
    
    print(f"Observed Vocabulary Size: {tokenizer.get_vocab_size()}")
    print(f"Success! Tokenizer saved to: {os.path.join(args.output_dir, 'tokenizer.json')}")

if __name__ == "__main__":
    main()
