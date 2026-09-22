"""Small shared helpers: paths, JSON with checksums, hashes, the run deadline."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
from threading import current_thread, main_thread

ROOT = Path(__file__).resolve().parents[1]


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest_text(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def hash_key(*parts):
    """Order-independent selection key from identities only; never from outcomes."""
    return digest_text('|'.join(str(p) for p in parts))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON field: {key}.')
        result[key] = value
    return result


def read_json(path):
    """Strict JSON: duplicate keys and nonfinite numbers are errors."""
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'Nonfinite JSON: {value}')))


def read_jsonl(path):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line, object_pairs_hook=unique_object,
                                   parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'Nonfinite JSON: {value}'))))
        except ValueError as error:
            raise ValueError(f'{Path(path).name}:{number}: {error}') from error
    return rows


def resolve(path):
    """Repository-relative paths in the devset resolve against the checkout; absolute ones stay."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


class DeadlineExceeded(RuntimeError):
    """One scenario exhausted its hard wall-clock limit (not an OSError, so transports cannot swallow it)."""


@contextmanager
def deadline(seconds):
    """Interrupt a synchronous model or tool call on the main thread; the trace still finalizes."""
    if seconds <= 0 or current_thread() is not main_thread():
        raise ValueError('A positive deadline on the main thread is required.')
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise RuntimeError('Refusing to replace an existing alarm timer.')
    previous = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise DeadlineExceeded(f'Hard deadline of {seconds:g} seconds exceeded.')

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
