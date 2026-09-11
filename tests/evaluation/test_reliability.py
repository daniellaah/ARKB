from arkb.evaluation.reliability import summarize_reliability
from arkb.evaluation.multihop import canonical_prediction
from arkb.agent.state import AgentFinal
import pytest


def test_failure_envelopes_are_not_counted_as_final_answers():
    row={'track':'musique','parse_error':'failed','stop_reason':'error','elapsed_ms':10,
         'result':{'response':None,'final':{'schema_version':'arkb-agent-final-v1','status':'error',
                   'citations':[],'termination_reason':'invalid_final_output'},
                   'observation':{'models':[{}], 'tools':[{'name':'read','executed':True,'status':'recoverable_error',
                                              'error':{'code':'invalid_reference'}}]}}}
    summary=summarize_reliability([row])
    assert summary['canonical_envelope_validity']['rate']==1
    assert summary['runs_with_final_output']['rate']==0
    assert summary['collection_tool_validation_errors']['rate']==1
    assert summary['recoverable_tool_errors']['rate']==1 and summary['fatal_tool_errors']['rate']==0
    assert summary['exact_match_errors']['rate'] is None


def test_canonical_benchmark_mapping_does_not_parse_answer_text():
    answer='Answer with ```json and literal {"answer": "inside"} text.'
    final=AgentFinal(answer,'answered',[{'ref':'ev_1','source':'a.md'}],'finish')
    predicted,error=canonical_prediction(final,{'a.md':7})
    assert error is None and predicted=={'predicted_answer':answer,'predicted_answerable':True,'predicted_support_idxs':[7]}
    with pytest.raises(ValueError):canonical_prediction(final,{'b.md':8})


def test_fatal_outcome_cannot_earn_abstention_credit():
    prediction,error=canonical_prediction(AgentFinal(None,'error',[],'fatal_error'),{})
    assert error and prediction['predicted_answerable'] is None
    prediction,error=canonical_prediction(AgentFinal('Not enough evidence','insufficient_evidence',[],'max_turns'),{})
    assert error is None and prediction['predicted_answerable'] is False
