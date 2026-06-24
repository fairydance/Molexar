"""
Molexar base model using transformers Gemma2 architecture.

This module provides the core Molexar model components that inherit from
Gemma2 for full compatibility with the transformers ecosystem.

Key Features:
- Inherits all Gemma2 features (sliding window, RoPE, GQA, softcapping)
- In-place condition embedding replacement
- Compatible with transformers Trainer and GenerationMixin
- Proper KV caching with DynamicCache
- Optimized for molecular data with Fragment-SELFIES tokenizers
"""
import torch
import torch.nn as nn
from typing import Optional, Union, List, Dict, Any, Tuple
import os
import json
from loguru import logger

from transformers import Gemma2Config, Gemma2Model, Gemma2ForCausalLM, Gemma2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.cache_utils import DynamicCache, Cache


class MolexarConfig(Gemma2Config):
    """
    Molexar configuration that extends Gemma2Config with Molexar-specific defaults.
    
    Optimized for molecular pre-training with:
    - Fragment-SELFIES molecular representation
    - Alternating sliding window attention
    - In-place condition embedding replacement
    """
    
    model_type = "molexar"
    architectures = ["MolexarForCausalLM"]
    
    PAD_TOKEN: str = "<PAD>"
    BOS_TOKEN: str = "<BOS>"
    EOS_TOKEN: str = "<EOS>"
    UNK_TOKEN: str = "<UNK>"
    SEP_TOKEN: str = "<SEP>"
    MASK_TOKEN: str = "<MASK>"
    COND_TOKEN: str = "<COND>"
    COND_END_TOKEN: str = "</COND>"
    MOL_TOKEN: str = "<MOL>"
    MOL_END_TOKEN: str = "</MOL>"
    
    MOL_HAC_TOKEN: str = "<MOL_HAC>"
    MOL_HBDC_TOKEN: str = "<MOL_HBDC>"
    MOL_HBAC_TOKEN: str = "<MOL_HBAC>"
    MOL_ROTBC_TOKEN: str = "<MOL_ROTBC>"
    
    MOL_WT_TOKEN: str = "<MOL_WT>"
    MOL_LOGP_TOKEN: str = "<MOL_LOGP>"
    MOL_TPSA_TOKEN: str = "<MOL_TPSA>"
    MOL_QED_TOKEN: str = "<MOL_QED>"
    MOL_SAS_TOKEN: str = "<MOL_SAS>"
    
    MOL_PHARMA_FP_TOKEN: str = "<MOL_PHARMA_FP>"
    PROT_SEQ_ESM_EMB_TOKEN: str = "<PROT_SEQ_ESM_EMB>"
    PROT_POC_GVP_EMB_TOKEN: str = "<PROT_POC_GVP_EMB>"
    
    VALUE_TOKEN: str = "<VALUE>"
    
    UNUSED0_TOKEN: str = "<UNUSED0>"
    UNUSED1_TOKEN: str = "<UNUSED1>"
    UNUSED2_TOKEN: str = "<UNUSED2>"
    UNUSED3_TOKEN: str = "<UNUSED3>"
    UNUSED4_TOKEN: str = "<UNUSED4>"
    UNUSED5_TOKEN: str = "<UNUSED5>"
    UNUSED6_TOKEN: str = "<UNUSED6>"
    UNUSED7_TOKEN: str = "<UNUSED7>"
    UNUSED8_TOKEN: str = "<UNUSED8>"
    UNUSED9_TOKEN: str = "<UNUSED9>"
    
    def __init__(
        self,
        vocab_size: int = 127,
        hidden_size: int = 256,
        intermediate_size: int = 640,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 4,
        num_key_value_heads: int = 1,
        max_position_embeddings: int = 256,
        head_dim: int = 64,
        query_pre_attn_scalar: Optional[int] = None,
        sliding_window: Optional[int] = None,
        attn_logit_softcapping: float = 50.0,
        final_logit_softcapping: float = 30.0,
        rope_theta: float = 10000.0,
        hidden_activation: str = "gelu_pytorch_tanh",
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        tie_word_embeddings: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        architectures: Optional[List[str]] = None,
        condition_settings: Optional[Dict[str, Any]] = None,
        mol_pharma_fp_dim: int = 1032,
        prot_seq_esm_emb_dim: int = 1152,
        prot_poc_gvp_emb_dim: int = 256,
        gvp_node_in_dim: Tuple[int, int] = (11, 3),
        gvp_edge_in_dim: Tuple[int, int] = (32, 1),
        gvp_node_h_dim: Tuple[int, int] = (256, 16),
        gvp_edge_h_dim: Tuple[int, int] = (32, 1),
        gvp_n_layers: int = 3,
        gvp_drop_rate: float = 0.1,
        condition_projector_layers: int = 2,
        **kwargs,
    ):
        
        if query_pre_attn_scalar is None:
            query_pre_attn_scalar = head_dim
        
        if sliding_window is None:
            sliding_window = max_position_embeddings // 2
        
        layer_types = kwargs.pop("layer_types", None)
        if layer_types is None:
            layer_types = [
                "sliding_attention" if i % 2 == 1 else "full_attention"
                for i in range(num_hidden_layers)
            ]
        
        if 'prot_esm_emb_dim' in kwargs or 'pock_zernike_dim' in kwargs:
            raise TypeError('MolexarConfig does not support old parameter names prot_esm_emb_dim/pock_zernike_dim.')
        
        if kwargs:
            raise TypeError(f'MolexarConfig() got unexpected keyword arguments: {list(kwargs.keys())}')
        
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            query_pre_attn_scalar=query_pre_attn_scalar,
            sliding_window=sliding_window,
            attn_logit_softcapping=attn_logit_softcapping,
            final_logit_softcapping=final_logit_softcapping,
            rope_theta=rope_theta,
            hidden_activation=hidden_activation,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            initializer_range=initializer_range,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            architectures=architectures,
            layer_types=layer_types,
            **kwargs,
        )
        
        from collections import OrderedDict
        
        self.condition_settings = condition_settings if condition_settings is not None else OrderedDict([
            ('mol_hac',  {'method': 'onehot', 'allowable_set': list(range(2, 51)), 'with_unknown': False}),
            ('mol_hbdc', {'method': 'onehot', 'allowable_set': list(range(0, 11)), 'with_unknown': False}),
            ('mol_hbac', {'method': 'onehot', 'allowable_set': list(range(0, 23)), 'with_unknown': False}),
            ('mol_rotbc', {'method': 'onehot', 'allowable_set': list(range(0, 21)), 'with_unknown': False}),
            ('mol_wt',   {'method': 'rbf', 'min': 30.0, 'max': 750.0, 'steps': 128}),
            ('mol_logp', {'method': 'rbf', 'min': -6.0, 'max': 12.0, 'steps': 96}),
            ('mol_tpsa', {'method': 'rbf', 'min': 0.0, 'max': 200.0, 'steps': 96}),
            ('mol_qed',  {'method': 'rbf', 'min': 0.3, 'max': 1.0, 'steps': 64}),
            ('mol_sas',  {'method': 'rbf', 'min': 1.0, 'max': 5.0, 'steps': 64}),
            ('mol_pharma_fp', {'method': 'direct', 'dim': mol_pharma_fp_dim}),
            ('prot_seq_esm_emb', {'method': 'direct', 'dim': prot_seq_esm_emb_dim}),
            ('prot_poc_gvp_emb', {'method': 'direct', 'dim': prot_poc_gvp_emb_dim}),
        ])
        
        self.gvp_node_in_dim = gvp_node_in_dim
        self.gvp_edge_in_dim = gvp_edge_in_dim
        self.gvp_node_h_dim = gvp_node_h_dim
        self.gvp_edge_h_dim = gvp_edge_h_dim
        self.gvp_n_layers = gvp_n_layers
        self.gvp_drop_rate = gvp_drop_rate
        if self.gvp_node_h_dim[0] != prot_poc_gvp_emb_dim:
            raise ValueError(
                f"gvp_node_h_dim[0] ({self.gvp_node_h_dim[0]}) must equal prot_poc_gvp_emb_dim ({prot_poc_gvp_emb_dim})"
            )
        self.prot_poc_gvp_emb_dim = prot_poc_gvp_emb_dim
        self.condition_projector_layers = condition_projector_layers
    
    @property
    def special_tokens_list(self) -> List[str]:
        return [
            self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN, self.SEP_TOKEN, self.MASK_TOKEN,
            self.COND_TOKEN, self.COND_END_TOKEN, self.MOL_TOKEN, self.MOL_END_TOKEN,
            self.MOL_HAC_TOKEN, self.MOL_HBDC_TOKEN, self.MOL_HBAC_TOKEN, self.MOL_ROTBC_TOKEN,
            self.MOL_WT_TOKEN, self.MOL_LOGP_TOKEN, self.MOL_TPSA_TOKEN, self.MOL_QED_TOKEN, self.MOL_SAS_TOKEN,
            self.MOL_PHARMA_FP_TOKEN, self.PROT_SEQ_ESM_EMB_TOKEN, self.PROT_POC_GVP_EMB_TOKEN,
            self.VALUE_TOKEN,
            self.UNUSED0_TOKEN, self.UNUSED1_TOKEN, self.UNUSED2_TOKEN, self.UNUSED3_TOKEN, self.UNUSED4_TOKEN,
            self.UNUSED5_TOKEN, self.UNUSED6_TOKEN, self.UNUSED7_TOKEN, self.UNUSED8_TOKEN, self.UNUSED9_TOKEN,
        ]
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        if os.path.isfile(pretrained_model_name_or_path):
            config_path = pretrained_model_name_or_path
        else:
            config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        
        if not os.path.exists(config_path):
            config_path = os.path.join(os.path.dirname(pretrained_model_name_or_path), "config.json")
        
        TRANSFORMERS_PARAMS = {
            'return_dict', 'output_hidden_states', 'torchscript', 'dtype', 'pruned_heads',
            'chunk_size_feed_forward', 'is_encoder_decoder', 'is_decoder',
            'cross_attention_hidden_size', 'add_cross_attention', 'tie_encoder_decoder',
            'finetuning_task', 'id2label', 'label2id', 'task_specific_params',
            'problem_type', 'tokenizer_class', 'prefix', 'sep_token_id', 'decoder_start_token_id',
            'max_length', 'min_length', 'do_sample', 'early_stopping', 'num_beams', 'temperature',
            'top_k', 'top_p', 'typical_p', 'repetition_penalty', 'length_penalty',
            'no_repeat_ngram_size', 'encoder_no_repeat_ngram_size', 'bad_words_ids',
            'num_return_sequences', 'output_scores', 'return_dict_in_generate',
            'forced_bos_token_id', 'forced_eos_token_id', 'remove_invalid_values',
            'exponential_decay_length_penalty', 'suppress_tokens', 'begin_suppress_tokens',
            'num_beam_groups', 'diversity_penalty', '_name_or_path', 'transformers_version',
            'tf_legacy_loss', 'use_bfloat16', 'output_attentions', 'model_type',
            'use_bidirectional_attention', 'rope_parameters'
        }
        removed_config_params = {
            "use" + "_conditions",
            "use" + "_gvp" + "_network",
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            filtered_dict = {
                k: v for k, v in config_dict.items()
                if k not in TRANSFORMERS_PARAMS and k not in removed_config_params
            }
            for key in removed_config_params:
                kwargs.pop(key, None)
            filtered_dict.update(kwargs)
            config = cls(**filtered_dict)
        else:
            config = cls(**kwargs)
        
        token_attrs = [
            'BOS_TOKEN', 'EOS_TOKEN', 'PAD_TOKEN', 'UNK_TOKEN', 'SEP_TOKEN', 'MASK_TOKEN',
            'COND_TOKEN', 'COND_END_TOKEN', 'MOL_TOKEN', 'MOL_END_TOKEN',
            'MOL_HAC_TOKEN', 'MOL_HBDC_TOKEN', 'MOL_HBAC_TOKEN', 'MOL_ROTBC_TOKEN',
            'MOL_WT_TOKEN', 'MOL_LOGP_TOKEN', 'MOL_TPSA_TOKEN', 'MOL_QED_TOKEN', 'MOL_SAS_TOKEN',
            'MOL_PHARMA_FP_TOKEN', 'PROT_SEQ_ESM_EMB_TOKEN', 'PROT_POC_GVP_EMB_TOKEN',
            'VALUE_TOKEN',
            'UNUSED0_TOKEN', 'UNUSED1_TOKEN', 'UNUSED2_TOKEN', 'UNUSED3_TOKEN', 'UNUSED4_TOKEN',
            'UNUSED5_TOKEN', 'UNUSED6_TOKEN', 'UNUSED7_TOKEN', 'UNUSED8_TOKEN', 'UNUSED9_TOKEN',
        ]
        for attr in token_attrs:
            if not hasattr(config, attr):
                setattr(config, attr, getattr(cls, attr))
        
        return config
    
    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        config_dict = self.to_dict()
        
        output_path = os.path.join(save_directory, "config.json")
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)


class GaussianSmearing(nn.Module):
    def __init__(self, start: float, end: float, steps: int, width: Optional[float] = None):
        super().__init__()
        self.start = start
        self.end = end
        self.steps = steps
        
        offset = torch.linspace(start, end, steps)
        self.register_buffer("offsets", offset)
        
        if width is None:
            width = float(offset[1] - offset[0])
        self.register_buffer("width", torch.tensor(width, dtype=torch.float32))

    def forward(self, values: torch.Tensor):
        if values.dim() == 1:
            values = values.unsqueeze(1)
            
        diff = values - self.offsets 
        rbf = torch.exp(-0.5 / torch.pow(self.width, 2) * torch.pow(diff, 2))
        return rbf


class DiscreteOneHotEncoder(nn.Module):
    def __init__(self, allowable_set: list, with_unknown: bool = False):
        super().__init__()
        self.allowable_set_list = allowable_set
        self.with_unknown = with_unknown
        
        self.register_buffer("vocab", torch.tensor(allowable_set))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(1)
        
        matches = (x == self.vocab).float()
        
        if self.with_unknown:
            found = matches.sum(dim=1, keepdim=True) 
            unknown = 1.0 - found
            matches = torch.cat([matches, unknown], dim=1)
            
        return matches


class ConditionEncoder(nn.Module):
    """
    Encodes condition values into hidden_size embeddings for in-place replacement.
    
    Input: [Batch, InputDim] or scalar
    Output: [Batch, HiddenSize]
    """
    def __init__(self, hidden_dim: int, settings: Dict[str, Any], activation: str = "gelu_pytorch_tanh", num_layers: int = 2):
        super().__init__()
        self.method = settings.get('method', 'direct')
        input_dim = 0
        
        if self.method == 'rbf':
            self.encoder = GaussianSmearing(
                start=settings['min'],
                end=settings['max'],
                steps=settings['steps'],
                width=settings.get('width', None)
            )
            input_dim = settings['steps']
            
        elif self.method == 'onehot':
            self.encoder = DiscreteOneHotEncoder(
                allowable_set=settings['allowable_set'],
                with_unknown=settings.get('with_unknown', False)
            )
            input_dim = len(settings['allowable_set']) + (1 if settings.get('with_unknown') else 0)
            
        else:
            input_dim = settings.get('dim', hidden_dim)
            self.encoder = nn.Identity()

        from transformers.activations import get_activation
        activation_fn = get_activation(activation)
        
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(activation_fn)
        
        self.projector = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expanded = self.encoder(x)
        features = self.projector(expanded)
        return features


class MolexarModel(Gemma2Model):
    """
    Molexar model based on Gemma2 architecture with in-place condition replacement.
    
    Conditions are encoded and replace placeholder token embeddings in-place,
    avoiding sequence length changes and attention mask modifications.
    """
    
    def __init__(self, config: MolexarConfig):
        super().__init__(config)
        self.config = config
        
        self.condition_encoders = nn.ModuleDict()
        for key, settings in config.condition_settings.items():
            if settings.get('method') != 'string':
                self.condition_encoders[key] = ConditionEncoder(
                    config.hidden_size,
                    settings,
                    config.hidden_activation,
                    config.condition_projector_layers,
                )

        from .nn.gvp import GVPEncoder

        self.gvp_encoder = GVPEncoder(
            node_in_dim=config.gvp_node_in_dim,
            edge_in_dim=config.gvp_edge_in_dim,
            node_h_dim=config.gvp_node_h_dim,
            edge_h_dim=config.gvp_edge_h_dim,
            n_layers=config.gvp_n_layers,
            drop_rate=config.gvp_drop_rate,
        )

        self.condition_key_to_token_id = {
            'mol_hac': config.special_tokens_list.index(config.MOL_HAC_TOKEN),
            'mol_hbdc': config.special_tokens_list.index(config.MOL_HBDC_TOKEN),
            'mol_hbac': config.special_tokens_list.index(config.MOL_HBAC_TOKEN),
            'mol_rotbc': config.special_tokens_list.index(config.MOL_ROTBC_TOKEN),
            'mol_wt': config.special_tokens_list.index(config.MOL_WT_TOKEN),
            'mol_logp': config.special_tokens_list.index(config.MOL_LOGP_TOKEN),
            'mol_tpsa': config.special_tokens_list.index(config.MOL_TPSA_TOKEN),
            'mol_qed': config.special_tokens_list.index(config.MOL_QED_TOKEN),
            'mol_sas': config.special_tokens_list.index(config.MOL_SAS_TOKEN),
            'mol_pharma_fp': config.special_tokens_list.index(config.MOL_PHARMA_FP_TOKEN),
            'prot_seq_esm_emb': config.special_tokens_list.index(config.PROT_SEQ_ESM_EMB_TOKEN),
            'prot_poc_gvp_emb': config.special_tokens_list.index(config.PROT_POC_GVP_EMB_TOKEN),
        }
        self.value_token_id = config.special_tokens_list.index(config.VALUE_TOKEN)
        
        self.mol_token_id = config.special_tokens_list.index(config.MOL_TOKEN)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        condition_indices: Optional[Dict[str, List[int]]] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        sequence_length = None
        if inputs_embeds is not None:
            sequence_length = inputs_embeds.shape[1]
        elif input_ids is not None:
            sequence_length = input_ids.shape[1]

        has_condition_positions = False
        if condition_indices is not None and sequence_length is not None:
            has_condition_positions = any(
                0 <= int(pos_idx) < sequence_length
                for indices in condition_indices.values()
                for pos_idx in indices
            )

        should_replace = (
            condition_values is not None
            and condition_indices is not None
            and has_condition_positions
        )
        
        if should_replace:
            if inputs_embeds is None:
                if input_ids is not None:
                    inputs_embeds = self.embed_tokens(input_ids)
                    input_ids = None
                else:
                    raise ValueError("Either input_ids or inputs_embeds must be provided")
            
            batch_size = inputs_embeds.shape[0]
            device = inputs_embeds.device
            dtype = inputs_embeds.dtype
            
            for key, indices in condition_indices.items():
                if key not in condition_values or condition_values[key] is None:
                    continue
                if key not in self.condition_encoders:
                    continue
                
                raw_val = condition_values[key]
                
                if isinstance(raw_val, (int, float)):
                    raw_val = torch.tensor([[raw_val]], dtype=torch.float32, device=device)
                elif isinstance(raw_val, torch.Tensor):
                    if raw_val.dim() == 0:
                        raw_val = raw_val.unsqueeze(0).unsqueeze(0)
                    elif raw_val.dim() == 1:
                        raw_val = raw_val.unsqueeze(0)
                    if raw_val.shape[0] != batch_size:
                        raw_val = raw_val.expand(batch_size, -1)
                    raw_val = raw_val.to(device=device)
                    if not raw_val.is_floating_point():
                        raw_val = raw_val.to(dtype=torch.float32)
                
                if key == 'prot_poc_gvp_emb' and self.gvp_encoder is not None:
                    if isinstance(raw_val, dict) and 'node_scalar' in raw_val:
                        with torch.amp.autocast('cuda', enabled=False):
                            raw_val = self.encode_pocket_geometry(
                                node_features=(raw_val['node_scalar'], raw_val['node_vector']),
                                edge_index=raw_val['edge_index'],
                                edge_features=(raw_val['edge_scalar'], raw_val['edge_vector']),
                                batch=raw_val.get('batch')
                            )

                if isinstance(raw_val, torch.Tensor) and raw_val.shape[0] != batch_size:
                    raw_val = raw_val.expand(batch_size, -1)

                raw_val = raw_val.to(dtype=dtype)
                
                cond_embeds = self.condition_encoders[key](raw_val)
                
                for batch_idx, pos_idx in enumerate(indices):
                    pos_idx = int(pos_idx)
                    if batch_idx < batch_size and 0 <= pos_idx < inputs_embeds.shape[1]:
                        inputs_embeds[batch_idx, pos_idx, :] = cond_embeds[batch_idx]
                
                logger.debug(f"Replaced condition '{key}' at indices {indices}, embed shape: {cond_embeds.shape}")
        
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
    
    def encode_pocket_geometry(
        self,
        node_features: Tuple[torch.Tensor, torch.Tensor],
        edge_index: torch.Tensor,
        edge_features: Tuple[torch.Tensor, torch.Tensor],
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pocket_embedding = self.gvp_encoder(node_features, edge_index, edge_features, batch)
        if batch is None:
            pocket_embedding = pocket_embedding.mean(dim=0, keepdim=True)
        return pocket_embedding


class MolexarForCausalLM(Gemma2ForCausalLM):
    """
    Molexar model for causal language modeling with in-place condition replacement.
    """
    
    config_class = MolexarConfig
    _forward_call_count = 0
    
    def __init__(self, config: MolexarConfig):
        super().__init__(config)
        self.model = MolexarModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tie_weights()
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        condition_indices: Optional[Dict[str, List[int]]] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        MolexarForCausalLM._forward_call_count += 1
        should_log = MolexarForCausalLM._forward_call_count <= 3
        
        if should_log:
            logger.debug(f"Forward call #{MolexarForCausalLM._forward_call_count}")
            if input_ids is not None:
                logger.debug(f"  input_ids shape: {input_ids.shape}, dtype: {input_ids.dtype}")
                logger.debug(f"  input_ids range: [{input_ids.min().item()}, {input_ids.max().item()}]")
                logger.debug(f"  First 20 tokens (sample 0): {input_ids[0, :20].tolist()}")
            if attention_mask is not None:
                logger.debug(f"  attention_mask shape: {attention_mask.shape}")
                logger.debug(f"  attention_mask sum per sample: {attention_mask.sum(dim=1)[:3].tolist()}")
            if labels is not None:
                logger.debug(f"  labels shape: {labels.shape}")
                non_ignore = (labels != -100).sum().item()
                total = labels.numel()
                logger.debug(f"  labels non-ignore ratio: {non_ignore}/{total} ({100*non_ignore/total:.1f}%)")
        
        if isinstance(past_key_values, (list, tuple)) and not isinstance(past_key_values, Cache):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        
        if return_dict is None:
            return_dict = True
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            condition_values=condition_values,
            condition_indices=condition_indices,
            return_dict=return_dict,
            **kwargs,
        )
        
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        
        if should_log:
            logger.debug(f"  hidden_states shape: {hidden_states.shape}")
            logger.debug(f"  logits shape: {logits.shape}")
        
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            
            if should_log:
                logger.debug(f"  shift_logits shape: {shift_logits.shape}")
                logger.debug(f"  shift_labels shape: {shift_labels.shape}")
                logger.debug(f"  loss: {loss.item():.4f}")
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        
        if "condition_values" in kwargs:
            model_inputs["condition_values"] = kwargs["condition_values"]
        if "condition_indices" in kwargs:
            model_inputs["condition_indices"] = kwargs["condition_indices"]
        
        return model_inputs
    
    @classmethod
    def from_pretrained(cls, model_path, config=None):
        if config is None:
            config = MolexarConfig.from_pretrained(model_path)
        
        model = cls(config)
        
        model_files = [
            f"{model_path}/pytorch_model_fsdp.bin",
            f"{model_path}/pytorch_model.bin",
            f"{model_path}/model.safetensors",
            f"{model_path}/pytorch_model_safe.bin"
        ]
        
        loaded = False
        for model_file in model_files:
            if os.path.exists(model_file):
                print(f"Loading model weights from: {model_file}")
                if model_file.endswith('.safetensors'):
                    from safetensors import safe_open
                    with safe_open(model_file, framework="pt") as f:
                        state_dict = {k: f.get_tensor(k) for k in f.keys()}
                else:
                    state_dict = torch.load(model_file, map_location='cpu')
                
                embed_weight = state_dict.get('model.embed_tokens.weight')
                if embed_weight is not None:
                    expected_vocab = config.vocab_size
                    actual_vocab = embed_weight.shape[0]
                    if actual_vocab != expected_vocab:
                        print(f"Warning: Model vocab size ({actual_vocab}) doesn't match config ({expected_vocab})")
                        continue
                
                model.load_state_dict(state_dict, strict=False)
                print(f"Successfully loaded weights")
                loaded = True
                break
        
        if not loaded:
            raise FileNotFoundError(f"No compatible model weights found in {model_path}")
        
        return model
    
    def save_pretrained(self, save_directory, state_dict=None, save_safetensors: bool = False, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        if state_dict is None:
            state_dict = self.state_dict()
        if save_safetensors:
            try:
                from safetensors.torch import save_file
                save_file(state_dict, f"{save_directory}/model.safetensors")
            except ImportError:
                torch.save(state_dict, f"{save_directory}/pytorch_model.bin")
        else:
            torch.save(state_dict, f"{save_directory}/pytorch_model.bin")

__all__ = [
    "MolexarConfig",
    "MolexarModel",
    "MolexarForCausalLM",
    "GaussianSmearing",
    "DiscreteOneHotEncoder",
    "ConditionEncoder",
]
