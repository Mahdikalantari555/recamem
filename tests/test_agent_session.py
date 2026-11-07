"""The initial coordinator preserves the established runner contract."""
import pytest
from recamem.agent_session import AgentSession
from recamem.runner import RunnerError, run_once


class Engine:
    name = 'example'
    def recall(self, prompt, top_k=3):
        return [{'text': 'A known fact.', 'source': 'user', 'score': 0.8, 'channels': ['episodic']}]
    def chat(self, prompt):
        return 'A known fact.'
    def teach(self, prompt):
        return {'propositions': 1}


@pytest.mark.parametrize('mode', ['chat', 'recall', 'teach'])
def test_session_preserves_runner_payload(mode, monkeypatch):
    monkeypatch.setenv('RECAMEM_ANCHOR', 'mock')
    engine = Engine()
    assert AgentSession(engine, profile='example').handle('hello', mode=mode) == run_once('hello', engine, mode=mode, profile='example')


def test_session_does_not_silently_accept_invalid_input():
    with pytest.raises(RunnerError):
        AgentSession(Engine()).handle('')
