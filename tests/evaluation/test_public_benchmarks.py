"""Official IDs/labels and full-corpus external adapter boundary."""
import json
from pathlib import Path
import pytest
from arkb.evaluation.external import digest,import_beir,load_external


def beir_input(tmp_path,name):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root=tmp_path/name
    for kind,rows in [('corpus',[{'_id':'doc/1','title':'Raw\nTitle','text':' raw text '},
                               {'_id':'unjudged','title':'','text':'complete corpus distractor'}]),
                      ('queries',[{'_id':'test/query','text':'original question'},
                                  {'_id':'training','text':'unselected query'}])]:
        path=root/kind/(kind+'-00000-of-00001.parquet');path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows),path)
    (root/'README.md').write_text('official fixture metadata')
    qrels=tmp_path/(name+'-qrels');qrels.mkdir()
    (qrels/'test.tsv').write_text('query-id\tcorpus-id\tscore\ntest/query\tdoc/1\t2\n')
    files={p.relative_to(tmp_path).as_posix():{'sha256':digest(p),'revision':'pinned-revision'} for p in tmp_path.rglob('*') if p.is_file()}
    (tmp_path/'downloads.json').write_text(json.dumps({'files':files}))
    return tmp_path


@pytest.mark.parametrize('name',['scifact','nfcorpus','fiqa'])
def test_generic_beir_keeps_full_corpus_original_ids_and_test_labels(tmp_path,name):
    result=import_beir(beir_input(tmp_path,name),name)
    assert len(result.corpus)==2
    assert result.corpus[0]=={'id':'doc/1','title':'Raw\nTitle','text':' raw text '}
    assert result.queries=={'test/query':'original question'}
    assert result.qrels=={'test/query':{'doc/1':2}}
    assert result.provenance['split']=='test'
    result.save(tmp_path/'normalized')
    assert load_external(tmp_path/'normalized')==result


def test_browsecomp_adapter_separates_queries_evidence_gold_and_answers(tmp_path):
    import base64,hashlib
    import pyarrow as pa
    import pyarrow.parquet as pq
    from arkb.evaluation.browsecomp import import_browsecomp
    password='fixture key'
    def enc(text):
        key=hashlib.sha256(password.encode()).digest()
        return base64.b64encode(bytes(b^key[i%32] for i,b in enumerate(text.encode()))).decode()
    corpus=tmp_path/'browsecomp-plus-corpus/data';corpus.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{'docid':'10','text':'support text','url':'https://example.com/10'},
        {'docid':'11','text':'answer-bearing text','url':'https://example.com/11'},
        {'docid':'12','text':'unjudged distractor','url':'https://example.com/12'}]),corpus/'train-0.parquet')
    queries=tmp_path/'browsecomp-plus/data';queries.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{'query_id':'007','query':enc('literal question'),
        'answer':enc('answer must stay out'),'evidence_docs':[{'docid':enc('10')},{'docid':enc('11')}],
        'gold_docs':[{'docid':enc('11')}]}]),queries/'test-0.parquet')
    official=tmp_path/'browsecomp-official/topics-qrels';official.mkdir(parents=True)
    (official/'qrel_evidence.txt').write_text('007 0 10 1\n007 0 11 1\n')
    (official/'qrel_golds.txt').write_text('007 0 11 1\n')
    files={p.relative_to(tmp_path).as_posix():{'sha256':digest(p),'revision':'fixed'} for p in tmp_path.rglob('*') if p.is_file()}
    (tmp_path/'downloads.json').write_text(json.dumps({'files':files}))
    data,gold,urls=import_browsecomp(tmp_path,password=password,expected_counts=(3,1))
    assert data.queries=={'007':'literal question'}
    assert data.qrels=={'007':{'10':1,'11':1}}
    assert gold=={'007':{'11':1}}
    assert len(data.corpus)==3
    assert all(d['title']=='' for d in data.corpus)
    assert urls['10']=='https://example.com/10'
    assert 'answer must stay out' not in json.dumps(data.queries)
    assert all(set(d)=={'id','title','text'} for d in data.corpus)


def test_explicit_empty_document_exception_keeps_full_corpus_and_recall_denominator(tmp_path):
    from arkb.evaluation.external import ExternalDataset,rank_metrics
    data=ExternalDataset('fixture',[{'id':'empty','title':'','text':' \n '},
        {'id':'readable','title':'','text':'useful evidence'}],{'q':'question'},
        {'q':{'empty':1,'readable':1}},{},{})
    data.save(tmp_path/'full',materialize=True)
    data.materialize(tmp_path/'indexable',omit_empty=True)
    data.verify_materialized(tmp_path/'indexable',omit_empty=True)
    assert len(list((tmp_path/'full/corpus').iterdir()))==2
    assert len(list((tmp_path/'indexable').iterdir()))==1
    assert len(load_external(tmp_path/'full').corpus)==2
    assert data.qrels=={'q':{'empty':1,'readable':1}}
    assert rank_metrics(data.qrels['q'],['readable'])['recall@10']==.5
    with pytest.raises(ValueError):data.verify_materialized(tmp_path/'indexable')
