from copy import deepcopy
import threading

from .probe import workload, blocks, normalized_message, summarize, one_call, ollama_process_memory


def fixture_rows():
    rows = []
    for ds in ('browsecomp-plus', 'fiqa', 'nfcorpus', 'musique'):
        for q in ('first', 'second'):
            for arm in ('A-M', 'F-S', 'A-All'):
                for variant in (('v0', 'v1') if ds == 'musique' else ('v0',)):
                    rows.append({'key': f'{ds}-{q}-{arm}-{variant}', 'schedule': {'dataset': ds, 'id': q, 'arm': arm, 'variant': variant},
                                 'result': {'final': {'status': 'error'}, 'observation': {'models': [{'request': {'query': q}}]}}})
    return rows


def test_selection_retains_failed_outcomes_and_musique_pair():
    rows = fixture_rows()
    selected = workload(rows)
    assert len(selected) == 10
    assert {r['question_id'] for r in selected} == {'first'}
    assert len([r for r in selected if r['dataset'] == 'musique']) == 4
    changed = deepcopy(rows)
    for r in changed: r['result']['final']['status'] = 'answered'
    assert workload(changed) == selected


def test_each_request_gets_three_repetitions_per_concurrency_with_shared_order():
    schedule = blocks(workload(fixture_rows()))
    assert [b['concurrency'] for b in schedule] == [1, 2, 4, 2, 4, 1, 4, 1, 2]
    assert sum(len(b['request_ids']) for b in schedule) == 90
    for repeat in range(3):
        selected = [b for b in schedule if b['repeat'] == repeat]
        assert selected[0]['request_ids'] == selected[1]['request_ids'] == selected[2]['request_ids']


def test_output_hash_ignores_json_formatting_but_preserves_tool_order():
    assert normalized_message({'message': {'content': '{"a":1,"b":2}'}}) == normalized_message({'message': {'content': '{ "b": 2, "a": 1 }'}})
    a = {'message': {'tool_calls': ['search', 'read']}}
    b = {'message': {'tool_calls': ['read', 'search']}}
    assert normalized_message(a) != normalized_message(b)


def test_administratively_unstarted_calls_never_touch_provider_or_disk(tmp_path):
    cancel = threading.Event()
    cancel.set()
    assert one_call({'id': 'q'}, tmp_path, 'c', {}, 0, cancel)['status'] == 'unstarted'
    assert not list(tmp_path.iterdir())


def test_screen_rejects_extra_length_outputs_despite_faster_throughput():
    batches = []
    for level in (1, 2, 4):
        for repeat in range(3):
            row = {'request_id': 'q', 'status': 'completed', 'client_seconds': 1, 'message_sha256': 'same',
                   'done_reason': 'length' if level == 4 else 'stop'}
            batches.append({'concurrency': level, 'throughput_per_second': level, 'calls': [row]})
    result = summarize(batches)
    assert result[1]['eligible_for_agent_confirmation']
    assert not result[2]['eligible_for_agent_confirmation']
    assert result[1]['identical_output_requests'] == 1


def test_partial_repetition_does_not_qualify_concurrency():
    row = {'request_id': 'q', 'status': 'completed', 'client_seconds': 1, 'message_sha256': 'same', 'done_reason': 'stop'}
    assert not summarize([{'concurrency': 1, 'throughput_per_second': 1, 'calls': [row]},
                          {'concurrency': 2, 'throughput_per_second': 2, 'calls': [row]}])[1]['eligible_for_agent_confirmation']


def test_memory_includes_backend_descendants_but_not_unrelated_llama_servers():
    processes = ollama_process_memory([
        ' 10 1 100 /Applications/Ollama.app/Contents/Resources/ollama serve',
        ' 11 10 8000 /Applications/Ollama.app/Contents/Resources/llama-server --model x',
        ' 12 11 30 helper',
        ' 20 1 9000 /unrelated/llama-server --model other',
    ])
    assert {p['pid'] for p in processes} == {10, 11, 12}
    assert sum(p['rss_kib'] for p in processes) == 8130
