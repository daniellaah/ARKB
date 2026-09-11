"""Measurable Qwen reranker inputs; evidence text itself is never modified."""

from collections import Counter
import hashlib
import json

from arkb.retrieval.models import SearchResult, validate_options
from arkb.retrieval.qwen_rerank import INSTRUCTION, PREFIX, SUFFIX


class QwenInputBuilder:
    """Build the exact unpadded scoring sequence and its token-allocation record.

    Token counts refer to field spans in the actual joined sequence, not a
    separate approximation. A boundary-crossing token belongs to the last field
    it overlaps. Template/special-token counts therefore need not be additive:
    special tokens are a subset of the complete sequence.

    Before-counts describe the original query/title and the full body in its
    allocated frame. Query gap tokens are reported separately and count toward
    the cap. Explicit allocation reserves at least min(full body tokens, 128);
    configurations that cannot provide that reservation are rejected.
    """

    def __init__(self, tokenizer, *, max_length=512, query_cap=None, title_cap=None,
                 query_strategy='head_tail', min_body_tokens=128):
        validate_options(max_length, None)
        if (query_cap is None) != (title_cap is None):
            raise ValueError('Query and title caps must be configured together.')
        validate_options(min_body_tokens, None)
        for value in (query_cap, title_cap):
            if value is not None:
                validate_options(value, None)
        if query_strategy not in ('head', 'tail', 'head_tail'):
            raise ValueError('Unknown query allocation strategy.')
        self.tokenizer, self.max_length = tokenizer, max_length
        self.query_cap, self.title_cap = query_cap, title_cap
        self.query_strategy, self.min_body_tokens = query_strategy, min_body_tokens
        self.prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.gap = tokenizer.encode('\n...\n', add_special_tokens=False) if query_cap else []
        self.capacity = max_length - len(self.prefix) - len(self.suffix)
        if self.capacity <= 0:
            raise ValueError('max_length must leave room after the Qwen template.')

    def _encode(self, query, title, body, gap=None):
        header = f'<Instruct>: {INSTRUCTION}\n<Query>: '
        middle = '\n<Document>: '
        pair = header + query + middle + title + '\n\n' + body
        query_start = len(header); query_end = query_start + len(query)
        title_start = query_end + len(middle); title_end = title_start + len(title)
        body_start = title_end + 2
        encoded = self.tokenizer(pair, truncation=False, add_special_tokens=False,
                                 return_offsets_mapping=True, return_attention_mask=False)
        groups = []
        for start, end in encoded['offset_mapping']:
            if end > body_start:
                groups.append('body')
            elif title_start < title_end and title_start < end and start < title_end:
                groups.append('title')
            elif query_start < end and start < query_end:
                groups.append('query_gap' if gap and query_start+gap[0] < end
                              and start < query_start+gap[1] else 'query')
            else:
                groups.append('template')
        return encoded['input_ids'], encoded['offset_mapping'], groups

    def _trim(self, text, cap, strategy):
        encoded = self.tokenizer(text, add_special_tokens=False, truncation=False,
                                 return_offsets_mapping=True, return_attention_mask=False)
        offsets = encoded['offset_mapping']; indexes = list(range(len(offsets)))
        if len(indexes) <= cap:
            return text, None
        if strategy == 'tail':
            tail = self._tail(indexes, cap, offsets)
            return text[offsets[tail[0]][0]:] if tail else '', None
        if strategy == 'head_tail' and cap > len(self.gap):
            available = cap - len(self.gap)
            head = self._head(indexes, (available+1)//2, offsets)
            tail = self._tail(indexes, available//2, offsets)
            first = text[:offsets[head[-1]][1]] if head else ''
            last = text[offsets[tail[0]][0]:] if tail else ''
            marker = '\n...\n'
            return first+marker+last, (len(first), len(first)+len(marker))
        head = self._head(indexes, cap, offsets)
        return text[:offsets[head[-1]][1]] if head else '', None

    def prepare(self, query: str, candidate: SearchResult) -> dict:
        from arkb.retrieval.models import validate_request
        validate_request(query, 1, None)
        if not isinstance(candidate, SearchResult):
            raise ValueError('Input construction requires a SearchResult.')
        title = candidate.metadata.get('title') or ''
        if not isinstance(title, str):
            raise ValueError('Candidate title must be text.')
        ids, offsets, groups = self._encode(query, title, candidate.content)
        original = Counter(groups)
        selected_query, selected_title = query, title
        if self.query_cap is not None:
            query_limit, title_limit = self.query_cap, self.title_cap
            # Reassemble the text after selecting complete character spans. This
            # preserves mandatory delimiters even when BPE merges a field's last
            # punctuation with its following newlines. Recount in joined context.
            while True:
                selected_query, gap = self._trim(query, query_limit, self.query_strategy)
                selected_title, _ = self._trim(title, title_limit, 'head')
                ids, offsets, groups = self._encode(selected_query, selected_title, candidate.content, gap)
                counts = Counter(groups)
                query_excess = counts['query'] + counts['query_gap'] - self.query_cap
                title_excess = counts['title'] - self.title_cap
                if query_excess <= 0 and title_excess <= 0:
                    break
                query_limit -= max(0, query_excess)
                title_limit -= max(0, title_excess)
                if min(query_limit, title_limit) < 0:
                    raise ValueError('Field allocation cannot fit the requested caps.')
            if not counts['query']:
                raise ValueError('The query allocation retains no query characters; increase query_cap.')
            body_capacity = self.capacity - sum(g != 'body' for g in groups)
            if body_capacity < min(self.min_body_tokens, counts['body']):
                raise ValueError('Sequence budget cannot meet the configured query/title caps and minimum body allocation.')
            selected = self._head(list(range(len(ids))), self.capacity, offsets)
            retained_ids = [ids[i] for i in selected]
            retained_groups = [groups[i] for i in selected]
        else:
            retained_ids, retained_groups = ids[:self.capacity], groups[:self.capacity]
        model_ids = self.prefix + retained_ids + self.suffix
        before, retained = Counter(groups), Counter(retained_groups)
        if self.query_cap is not None and retained['body'] < min(self.min_body_tokens, before['body']):
            raise ValueError('Unicode boundary cannot meet the minimum body allocation within the sequence limit.')
        special = set(self.tokenizer.all_special_ids)
        record = {'input_ids': model_ids, 'total_model_tokens': len(model_ids),
                  'truncated': selected_query != query or selected_title != title or len(ids) > len(retained_ids),
                  'body_empty': retained['body'] == 0,
                  'query_gap_tokens': retained['query_gap'],
                  'body_retained_fraction': retained['body']/before['body'] if before['body'] else None,
                  'template_tokens': len(self.prefix) + len(self.suffix) + retained['template'],
                  'template_tokens_before': len(self.prefix) + len(self.suffix) + before['template'],
                  'special_tokens': sum(i in special for i in model_ids),
                  'input_ids_sha256': hashlib.sha256(json.dumps(model_ids, separators=(',', ':')).encode()).hexdigest()}
        for field in ('query', 'title', 'body'):
            record[field + '_tokens_before'] = (before if field == 'body' else original)[field]
            record[field + '_tokens_retained'] = retained[field]
        return record

    @staticmethod
    def _head(indexes, cap, offsets):
        """Do not cut between byte tokens that overlap one Unicode character."""
        size = min(cap, len(indexes))
        while 0 < size < len(indexes) and offsets[indexes[size-1]][1] > offsets[indexes[size]][0]:
            size -= 1
        return indexes[:size]

    @staticmethod
    def _tail(indexes, cap, offsets):
        start = max(0, len(indexes)-cap)
        while 0 < start < len(indexes) and offsets[indexes[start-1]][1] > offsets[indexes[start]][0]:
            start += 1
        return indexes[start:]
