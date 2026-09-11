"""Pinned Qwen3 reranker with the model's official yes/no scoring template."""

from collections.abc import Sequence

from arkb.retrieval.models import SearchResult, validate_options


QWEN_MODEL = 'Qwen/Qwen3-Reranker-0.6B'
QWEN_REVISION = 'e61197ed45024b0ed8a2d74b80b4d909f1255473'
INSTRUCTION = 'Given a web search query, retrieve relevant passages that answer the query'
PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
          'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
          '<|im_start|>user\n')
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


class QwenRerankerScorer:
    """CPU float32, title+body candidates; higher logit(yes)-logit(no) is better.

    Uses the official model-card Transformers template. Prefix/suffix are
    reserved before allocating query/title/body, preserving the scoring position.
    Logit difference orders candidates identically to the official two-token
    softmax, without saturation from converting large logits to probabilities.
    No generation, sampling, remote code, or label-dependent instruction. Empty
    document inputs are reported explicitly for the Reranker fallback boundary.
    """
    score_type = 'yes_no_logit_difference'

    def __init__(self, *, max_length: int = 512, batch_size: int = 16,
                 cache_folder: str | None = None, local_files_only: bool = False,
                 query_cap: int | None = 128, title_cap: int = 64,
                 query_strategy: str = 'head_tail'):
        validate_options(max_length, None)
        validate_options(batch_size, None)
        if query_cap is not None:
            validate_options(query_cap, None)
        validate_options(title_cap, None)
        if query_strategy not in ('head', 'tail', 'head_tail'):
            raise ValueError('Unknown query allocation strategy.')
        self.query_cap, self.title_cap, self.query_strategy = query_cap, title_cap, query_strategy
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise ValueError('Install the optional reranker with: uv sync --locked --extra rerank') from error

        options = dict(revision=QWEN_REVISION, cache_dir=cache_folder,
                       local_files_only=local_files_only, trust_remote_code=False)
        self.tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, padding_side='left', **options)
        self._prefix = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self._suffix = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        if max_length <= len(self._prefix) + len(self._suffix):
            raise ValueError('max_length must leave room for query/document text after the Qwen template.')
        self._true = self.tokenizer.convert_tokens_to_ids('yes')
        self._false = self.tokenizer.convert_tokens_to_ids('no')
        if any(type(token) is not int or token == self.tokenizer.unk_token_id
               for token in (self._true, self._false)) or self._true == self._false:
            raise ValueError('Qwen tokenizer must provide distinct yes/no tokens.')
        self._model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL, dtype=torch.float32, attn_implementation='sdpa', **options).to('cpu').eval()
        self.max_length, self.batch_size = max_length, batch_size
        self.identity = (f'{QWEN_MODEL}@{QWEN_REVISION}/cpu/float32/max_length={max_length}/batch_size={batch_size}'
                         '/title-body-v1/official-instruction-v1/yes-no-logit-difference')
        if query_cap is not None:
            self.identity += f'/allocation-v2/query_cap={query_cap}/title_cap={title_cap}/query={query_strategy}/body_min=128'

    def prepare_inputs(self, query: str, candidates: Sequence[SearchResult]) -> list[dict]:
        """Lossless unpadded inputs and field diagnostics, without model inference."""
        from arkb.retrieval.qwen_inputs import QwenInputBuilder
        builder = QwenInputBuilder(self.tokenizer, max_length=self.max_length,
                                  query_cap=self.query_cap,
                                  title_cap=self.title_cap if self.query_cap is not None else None,
                                  query_strategy=self.query_strategy)
        return [builder.prepare(query, hit) for hit in candidates]

    def _inputs(self, query, candidates):
        return self._pad(self.prepare_inputs(query, candidates))

    def _pad(self, prepared):
        inputs = {'input_ids': [row['input_ids'] for row in prepared]}
        return self.tokenizer.pad(inputs, padding=True, return_tensors='pt',
                                  return_attention_mask=True)

    def score(self, query: str, candidates: Sequence[SearchResult]) -> list[float]:
        import torch

        if not candidates:
            return []
        prepared = self.prepare_inputs(query, candidates)
        if all(row['body_empty'] for row in prepared):
            from arkb.retrieval.rerank import EmptyRerankerInput
            raise EmptyRerankerInput('No candidate retains document body tokens.')
        scores = []
        with torch.inference_mode():
            for offset in range(0, len(candidates), self.batch_size):
                inputs = self._pad(prepared[offset:offset + self.batch_size])
                logits = self._model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :]
                scores.extend((logits[:, self._true] - logits[:, self._false]).tolist())
        return scores
