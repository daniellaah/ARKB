"""Administrative stops finish the current attempt and persist across restarts."""
import json
import signal
from pathlib import Path


class StopController:
    def __init__(self, path):
        self.path = Path(path)
        self.reason = None
        self.previous = {}

    def __enter__(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            self.previous[sig] = signal.signal(sig, self._signal)
        return self

    def _signal(self, signum, frame):
        # Do not raise into a provider/tool call or interrupt its journal.
        self.reason = self.reason or signal.Signals(signum).name

    def request(self, reason):
        self.reason = self.reason or reason

    def requested(self):
        return self.reason is not None or self.path.exists()

    def persist(self):
        if self.reason and not self.path.exists():
            try:
                with self.path.open('x') as stream:
                    json.dump({'reason': self.reason, 'scope': 'administrative_between_attempts'}, stream)
                    stream.write('\n')
            except FileExistsError:
                pass
        return {'reason': self.reason or 'stop_request_file', 'request_path': str(self.path),
                'resume': 'Review the cause and explicitly archive the stop request before resuming.'}

    def __exit__(self, *exc):
        for sig, previous in self.previous.items():
            signal.signal(sig, previous)
