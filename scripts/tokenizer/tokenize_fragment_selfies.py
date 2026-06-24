#!/usr/bin/env python3
"""
Fragment-SELFIES Tokenization Script

This script tokenizes Fragment-SELFIES strings using a pre-trained tokenizer.
It can be used as a standalone CLI tool or imported as a module.
"""

import sys
import os
import argparse
from typing import List, Dict, Union

try:
    from transformers import PreTrainedTokenizerFast
    from molexar.tokenizer import MolexarTokenizerFast
    from molexar.modeling import MolexarConfig
except ImportError as e:
    print(f"Error: Required module not found: {e}")
    print("Please ensure you have the necessary dependencies installed.")
    print("Try: pip install transformers")
    sys.exit(1)


def tokenize_fragment_selfies_string(fragment_selfies_string: str, tokenizer_path: str) -> Dict[str, Union[List[str], List[int]]]:
    """
    Tokenize a Fragment-SELFIES string using the tokenizer from the specified path.
    
    Args:
        fragment_selfies_string: The Fragment-SELFIES string to tokenize
        tokenizer_path: Path to the tokenizer.json file or directory containing it
    
    Returns:
        Dictionary containing:
        - 'tokens': List of token strings
        - 'token_ids': List of token IDs
        - 'input_ids': List of input IDs with BOS/EOS tokens
        - 'attention_mask': List of attention masks
    
    Raises:
        FileNotFoundError: If tokenizer file doesn't exist
        ValueError: If tokenization fails
    
    Example:
        >>> result = tokenize_fragment_selfies_string("[Frag][C]", "/path/to/tokenizer")
        >>> print(result['tokens'])
        ['<BOS>', '[Frag]', '[C]', '<EOS>']
    """
    # Determine tokenizer file path
    if os.path.isdir(tokenizer_path):
        tokenizer_file = os.path.join(tokenizer_path, "tokenizer.json")
    else:
        tokenizer_file = tokenizer_path
    
    if not os.path.exists(tokenizer_file):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_file}")
    
    try:
        # Load tokenizer
        tokenizer = MolexarTokenizerFast(tokenizer_file=tokenizer_file)
        
        # Tokenize the string
        # First, get the raw tokens (without BOS/EOS)
        encoding = tokenizer(
            fragment_selfies_string,
            truncation=False,
            padding=False,
            return_tensors=None
        )
        
        # Get tokens for display
        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        
        # Return comprehensive results
        return {
            'tokens': tokens,
            'token_ids': encoding['input_ids'],
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding.get('attention_mask', [1] * len(encoding['input_ids']))
        }
        
    except Exception as e:
        raise ValueError(f"Failed to tokenize: {e}")


def print_tokenization_result(fragment_selfies_string: str, result: Dict[str, Union[List[str], List[int]]]):
    """
    Print the tokenization result in a formatted way.
    """
    print("\n" + "="*70)
    print("FRAGMENT-SELFIES TOKENIZATION RESULT")
    print("="*70)
    
    print(f"\nInput Fragment-SELFIES String:")
    print(f"  {fragment_selfies_string}")
    
    print(f"\nTokenized into {len(result['tokens'])} tokens:")
    print(f"  Tokens: {result['tokens']}")
    print(f"  Token IDs: {result['token_ids']}")
    
    print(f"\nDetailed Breakdown:")
    for i, (token, token_id) in enumerate(zip(result['tokens'], result['token_ids'])):
        print(f"  [{i:3d}] {token:20s} -> {token_id:4d}")
    
    print(f"\nWith BOS/EOS (for model input):")
    print(f"  Input IDs: {result['input_ids']}")
    print(f"  Attention Mask: {result['attention_mask']}")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize a Fragment-SELFIES string using a pre-trained tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tokenize a simple Fragment-SELFIES string
  python tokenize_fragment_selfies.py -s "[Frag][C]" -t /path/to/tokenizer.json
  
  # Use custom tokenizer file
  python tokenize_fragment_selfies.py -s "[Frag][C]" -t /path/to/tokenizer.json
  
  # Use tokenizer from directory
  python tokenize_fragment_selfies.py -s "[Frag][C]" -t /path/to/tokenizer_dir
  
  # Get just the token IDs (for scripting)
  python tokenize_fragment_selfies.py -s "[Frag][C]" -t /path/to/tokenizer.json --quiet --format ids
        """
    )
    
    parser.add_argument(
        '-s', '--string',
        type=str,
        default='[Frag][C][C][O]',
        help='Fragment-SELFIES string to tokenize (default: %(default)s)'
    )
    
    parser.add_argument(
        '-t', '--tokenizer',
        type=str,
        default='models/tokenizer',
        help='Path to tokenizer.json file or directory (default: models/tokenizer)'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress verbose output, only print tokens'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['full', 'tokens', 'ids', 'json'],
        default='full',
        help='Output format: full (detailed), tokens, ids, or json'
    )
    
    args = parser.parse_args()
    
    try:
        # Perform tokenization
        result = tokenize_fragment_selfies_string(args.string, args.tokenizer)
        
        # Output based on format
        if args.quiet:
            if args.format == 'tokens':
                print(' '.join(result['tokens']))
            elif args.format == 'ids':
                print(' '.join(map(str, result['token_ids'])))
            elif args.format == 'json':
                import json
                print(json.dumps(result, indent=2))
            else:
                print(result['token_ids'])
        else:
            if args.format == 'json':
                import json
                print(json.dumps(result, indent=2))
            else:
                print_tokenization_result(args.string, result)
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
