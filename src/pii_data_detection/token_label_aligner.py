from typing import Any, List, Tuple
from transformers import AutoTokenizer


class ProducedLabelAligner:
    """
    Align the labels as per the tokens produced by the tokenizer

    Context: In the given dataset, the given tokens and labels may not align with the tokens produced by the tokenizer
    This code references the following: https://www.kaggle.com/code/nbroad/transformer-ner-baseline-lb-0-854
    """

    def __call__(
            self, 
            old_tokens: List[str], 
            old_labels: List[str], 
            trailing_whitespace: List[bool],
            tokenizer: AutoTokenizer,
            max_length: int
            # new_tokens: List[int], 
            # word_ids: List[int], 
            # offset_mapping: List[Tuple[int]]
        ) -> Any:

        text = []
        characterwise_labels = [] # Labels per character
        
        for token, label, has_ws in zip(old_tokens, old_labels, trailing_whitespace):
            text.append(token + (" " if has_ws else ""))
            characterwise_labels.extend(([label] * len(token)) + (["O"] if has_ws else ""))
            text = "".join(text)
        
        tokenized = tokenizer("".join(text), return_offsets_mapping=True, max_length=max_length)

        offset_mapping = tokenized.offset_mapping
        
        new_token_labels = []

        for start_index, end_index in offset_mapping:

            # Upon Encountering [CLS] token, append label as O
            if start_index + end_index == 0:
                new_token_labels.append("O")
                continue
            
            # Skip Spaces
            if text[start_index].isspace():
                start_index += 1 
            
            # Make sure that start_index is always less than the character length
            if start_index >= len(characterwise_labels):
                start_index = len(characterwise_labels) - 1

            new_token_labels.append(characterwise_labels[start_index])
        
        length = len(tokenized.input_ids)

        return {
            **tokenized,
            "labels": new_token_labels,
            "length": length
        }
