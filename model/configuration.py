import logging

from transformers import BertConfig

logger = logging.getLogger(__name__)

#KoBERT
kobert_config = {
    'attention_probs_dropout_prob': 0.1,
    'hidden_act': 'gelu',
    'hidden_dropout_prob': 0.1,
    'hidden_size': 768,
    'initializer_range': 0.02,
    'intermediate_size': 3072,
    'max_position_embeddings': 512,
    'num_attention_heads': 12,
    'num_hidden_layers': 12,
    'type_vocab_size': 2,
    'vocab_size': 8002
}

def get_kobert_config():
    return BertConfig.from_dict(kobert_config)