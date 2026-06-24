import re
import logging
import os
from typing import Optional, List
from tokenizers import Tokenizer, pre_tokenizers, models, trainers, processors, Regex, normalizers, decoders
from transformers import PreTrainedTokenizerFast
from loguru import logger
from .modeling import MolexarConfig

class MolexarSplitter:
    """
    Standard splitter for Fragment-SELFIES and Molexar template tokens.
    Holds the regex pattern used for native tokenization.
    """
    # Fragment-SELFIES and SELFIES tokens are bracketed. Molexar control tokens
    # remain angle-bracketed, for example <BOS>, <MOL>, and </MOL>.
    REGEX_PATTERN = r"(\[[^\]]+\]|</?[^>]+>)"

    def __init__(self, pattern: Optional[str] = None):
        self.pattern = pattern if pattern else self.REGEX_PATTERN
        self.regex = re.compile(self.pattern)

    def split(self, i: int, normalized_string: str) -> List[str]:
        """
        Helper method to verify splits in Python if needed.
        """
        text = str(normalized_string)
        tokens = list(self.regex.findall(text))
        return [t for t in tokens if t]

def get_molexar_tokenizer_object(config: MolexarConfig):
    """
    Builds the backend tokenizer object using only Split pre-tokenizer.
    No BPE processing - just regex splitting.
    
    The post-processor is configured to handle our conditional format:
    - Single sequence: <BOS> $A <EOS>
    - Where $A represents the tokenized content (which may include condition keys and values)
    
    All special tokens from config.special_tokens_list are included in the post_processor
    with their indices as IDs (matching what training will produce).
    """
    # 1. WordLevel model (simple vocabulary lookup)
    tokenizer = Tokenizer(models.WordLevel(unk_token=config.UNK_TOKEN))
    
    # 2. Normalizer: Validates input and strips whitespace/newlines
    # This runs BEFORE pre-tokenization. It removes the trailing '\n' from file lines.
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Strip()  # Removes leading and trailing whitespace/newlines
    ])

    # 3. Pre-Tokenizer: Split using regex pattern ONLY
    splitter_inst = MolexarSplitter()
    
    # Pattern must be wrapped in Regex() object
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        pattern=Regex(splitter_inst.pattern), 
        behavior="isolated"
    )

    # 4. Decoder
    tokenizer.decoder = decoders.ByteLevel()

    # 5. Post Processor (Templates for BOS/EOS)
    # For unconditional generation: <BOS> $A <EOS>
    # For conditional generation: <BOS><KEY1>{value1}<SEP>{molecule}<EOS>
    # The $A placeholder represents the tokenized content
    # Include ALL special tokens with their index as ID (matching training)
    special_tokens_with_ids = [
        (token, idx) for idx, token in enumerate(config.special_tokens_list)
    ]
    
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{config.BOS_TOKEN} $A {config.EOS_TOKEN}",
        special_tokens=special_tokens_with_ids,
    )
    return tokenizer

def train_and_save_tokenizer(data_files: List[str], save_path: str, config: MolexarConfig):
    """
    Build tokenizer vocabulary from data files using WordLevel training.
    """
    # 1. Create tokenizer with core components
    tokenizer = Tokenizer(models.WordLevel(unk_token=config.UNK_TOKEN))
    tokenizer.normalizer = normalizers.Sequence([normalizers.Strip()])
    
    splitter_inst = MolexarSplitter()
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        pattern=Regex(splitter_inst.pattern), 
        behavior="isolated"
    )
    tokenizer.decoder = decoders.ByteLevel()
    
    # 2. Configure trainer
    trainer = trainers.WordLevelTrainer(
        vocab_size=100_000_000,
        special_tokens=config.special_tokens_list,
        show_progress=True
    )
    
    # 3. Train on Fragment-SELFIES records. Conversion outputs may preserve IDs
    # after whitespace, so only the first field belongs to the molecular language.
    def fragment_selfies_records():
        for data_file in data_files:
            with open(data_file, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        yield line.split(maxsplit=1)[0]

    tokenizer.train_from_iterator(fragment_selfies_records(), trainer=trainer)
    
    # 4. Set post-processor AFTER training (best practice)
    # Use token_to_id() to get correct IDs from the trained vocabulary
    # Include ALL special tokens from config
    special_tokens_with_ids = [
        (token, tokenizer.token_to_id(token)) for token in config.special_tokens_list
    ]
    
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{config.BOS_TOKEN} $A {config.EOS_TOKEN}",
        special_tokens=special_tokens_with_ids,
    )
    
    # 5. Save
    os.makedirs(save_path, exist_ok=True)
    tokenizer.save(f"{save_path}/tokenizer.json")
    return tokenizer

class MolexarTokenizerFast(PreTrainedTokenizerFast):
    """
    Hugging Face Wrapper for the Molexar Tokenizer.
    Enables `AutoTokenizer` compatibility and fast loading.
    """
    def __init__(self, tokenizer_file=None, **kwargs):
        config = MolexarConfig()
        super().__init__(tokenizer_file=tokenizer_file, **kwargs)
        
        # Set Special Tokens attributes for HF mapping
        self.pad_token = config.PAD_TOKEN
        self.bos_token = config.BOS_TOKEN
        self.eos_token = config.EOS_TOKEN
        self.unk_token = config.UNK_TOKEN
        self.sep_token = config.SEP_TOKEN
        
        self.molexar_config = config
