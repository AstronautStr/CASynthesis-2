"""Catalog check (S7): recompute EVERY record of a catalog with the current
code, compare each of its three tracks (A, B, monitor) with the stored WAVs,
and keep the outcome as a REPORT that survives the application.

One run = one target implementation: the code the worker process actually
executes (its S6 provenance, captured once at start) + its environment.  The
runtime set is fingerprinted again before and after every record; a change
stops the run (the unfinished record is 'unchecked', finished ones stay).

Layout (inside the catalog root, user data):
    <root>/.verify/<run_id>/run.json          the report (rewritten after every record)
    <root>/.verify/<run_id>/<record_id>/<track>.wav   recomputed PCM of DIFFERING tracks
    <root>/.verify/<run_id>/marks.json        listening marks of this report only
Records (WAVs, snapshots, record.json, <record>/replay/) are never modified;
a new run makes a new report, a single check (Catalog.replay) never touches
a report's pair.

Per track one of: 'exact' (same format, length and every PCM sample),
'differs' (a comparable pair with sample or length differences), 'failed'
(no valid pair -- with the reason), 'unchecked' (not reached / cancelled /
target version lost).  A small numeric difference is never labelled
"inaudible": the listening mark is a separate, optional, per-pair judgement.

The recompute itself is Catalog.recompute() -- the same journal replay the
single check uses; there is no second algorithm here.
"""
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import uuid

import numpy as np

from casynth_config import SR
from . import provenance as prov
from .catalog import Catalog, CatalogError, read_wav, _pcm_to_wav
from .runner import BLOCK, CHANNELS, OUTPUTS

RUNS_DIRNAME = '.verify'
RUN_FORMAT = 1
TRACK_EXACT, TRACK_DIFFERS, TRACK_FAILED, TRACK_UNCHECKED = 'exact', 'differs', 'failed', 'unchecked'
TRACK_RESULTS = (TRACK_DIFFERS, TRACK_EXACT, TRACK_FAILED, TRACK_UNCHECKED)   # display order
RUN_RUNNING, RUN_COMPLETE, RUN_CANCELLED, RUN_STOPPED, RUN_FAILED, RUN_INTERRUPTED = (
    'running', 'complete', 'cancelled', 'stopped', 'failed', 'interrupted')
MARK_SAME, MARK_DIFFERENT = 'same', 'different'     # "can't hear" / "hear a difference"
MARKS = (MARK_SAME, MARK_DIFFERENT)
TRACK_LABELS = {'A': 'A', 'B': 'B', 'monitor': 'As heard'}
RESULT_LABELS = {TRACK_EXACT: 'Exact match', TRACK_DIFFERS: 'Differs numerically',
                 TRACK_FAILED: 'Could not check', TRACK_UNCHECKED: 'Not checked'}
RUN_LABELS = {RUN_RUNNING: 'running', RUN_COMPLETE: 'complete', RUN_CANCELLED: 'cancelled',
              RUN_STOPPED: 'stopped (code changed)', RUN_FAILED: 'failed',
              RUN_INTERRUPTED: 'interrupted'}


class VerifyError(RuntimeError):
    pass


# -- comparison -------------------------------------------------------------------
def pcm_sha(pcm):
    """sha256 of the raw int16 PCM bytes (== the record's WAV data fingerprint)."""
    return hashlib.sha256(np.ascontiguousarray(pcm, dtype=np.int16).tobytes()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def compare_pcm(ref, new):
    """Classify a comparable pair of int16 (n, ch) arrays.
    -> dict(result 'exact' | 'differs', n_ref, n_new, length_mismatch,
            frac_diff, max_abs_diff) -- the diagnostics are computed over the
    common prefix (the whole track when the lengths agree); the difference is
    taken in int32 (no int16 overflow: -32768 vs 32767 -> 65535).  A length
    mismatch is 'differs' by itself: the tail is never cut to declare a match."""
    ref = np.asarray(ref, dtype=np.int16)
    new = np.asarray(new, dtype=np.int16)
    n_ref, n_new = int(len(ref)), int(len(new))
    n = min(n_ref, n_new)
    length_mismatch = n_ref != n_new
    if n > 0:
        d = np.abs(ref[:n].astype(np.int32) - new[:n].astype(np.int32))
        diff_frames = int(np.count_nonzero(d.any(axis=1)))
        frac = diff_frames / n
        max_abs = int(d.max())
    else:
        frac, max_abs = 0.0, 0
    result = TRACK_DIFFERS if (length_mismatch or max_abs > 0) else TRACK_EXACT
    return dict(result=result, n_ref=n_ref, n_new=n_new, length_mismatch=length_mismatch,
                frac_diff=float(frac), max_abs_diff=max_abs)


def _track(result, reason='', **extra):
    d = dict(result=result, reason=reason, n_ref=None, n_new=None, length_mismatch=None,
             frac_diff=None, max_abs_diff=None, ref_sha256=None, new_sha256=None, new_file=None)
    d.update(extra)
    return d


def record_status(tracks):
    """Record-level grouping from the three track results: any difference ->
    'differs'; else any failure -> 'failed'; else anything unchecked ->
    'unchecked'; else 'exact'.  (A partial failure never hides the results the
    other tracks did get -- they stay in the track dicts.)"""
    rs = [tracks[o]['result'] for o in OUTPUTS if o in tracks]
    if TRACK_DIFFERS in rs:
        return TRACK_DIFFERS
    if TRACK_FAILED in rs:
        return TRACK_FAILED
    if TRACK_UNCHECKED in rs or len(rs) < len(OUTPUTS):
        return TRACK_UNCHECKED
    return TRACK_EXACT


def track_text(t):
    """One line for a track result (numbers only when they exist)."""
    r = t['result']
    if r == TRACK_EXACT:
        return "exact match"
    if r == TRACK_DIFFERS:
        parts = []
        if t.get('length_mismatch'):
            parts.append(f"length {t['n_ref']} -> {t['n_new']} samples")
        if t.get('frac_diff') is not None:
            parts.append(f"{100.0 * t['frac_diff']:.1f}% samples differ, max |d| {t['max_abs_diff']}")
        return "differs: " + "; ".join(parts)
    if r == TRACK_FAILED:
        return "could not check: " + (t.get('reason') or '?')
    return "not checked" + (f": {t['reason']}" if t.get('reason') else '')


# -- report ------------------------------------------------------------------------
def _write_json(path, doc):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _read_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def target_label(target):
    """'Pinned <sha7>' / 'Local: <reason>' for the run's target provenance."""
    if not target:
        return "version unknown"
    st, why = prov.status_of(target)
    if st == 'pinned':
        return f"Pinned {prov.short(target.get('commit'))}"
    return f"Local: {why}"


class Run:
    """A loaded report (read-only view + marks)."""

    def __init__(self, run_dir, doc, marks, active=False):
        self.dir, self.doc, self.marks = run_dir, doc, marks
        self.active = active                  # being executed by THIS bench right now

    @property
    def id(self):
        return self.doc['id']

    @property
    def status(self):
        s = self.doc.get('status')
        if s == RUN_RUNNING and not self.active:
            return RUN_INTERRUPTED             # the process that ran it is gone
        return s

    @property
    def status_label(self):
        lab = RUN_LABELS.get(self.status, self.status)
        why = self.doc.get('reason')
        return f"{lab}: {why}" if why and self.status != RUN_COMPLETE else lab

    @property
    def target(self):
        return self.doc.get('target') or {}

    @property
    def target_label(self):
        return target_label(self.target)

    @property
    def started(self):
        return self.doc.get('started', '')

    @property
    def record_ids(self):
        return list(self.doc.get('records') or [])

    def result(self, rid):
        """The record's entry (pending ones read as all-unchecked)."""
        r = (self.doc.get('results') or {}).get(rid)
        if r is None:
            why = ("in progress" if self.doc.get('current') == rid and self.active
                   else self.doc.get('reason') or "not reached")
            return dict(id=rid, title=rid, status=TRACK_UNCHECKED, reason=why, origin=None,
                        tracks={o: _track(TRACK_UNCHECKED, why) for o in OUTPUTS})
        return r

    def groups(self):
        """{'differs': [ids], 'exact': [...], 'failed': [...], 'unchecked': [...]}
        in catalog order."""
        g = {k: [] for k in TRACK_RESULTS}
        for rid in self.record_ids:
            g[self.result(rid)['status']].append(rid)
        return g

    def counts(self):
        return {k: len(v) for k, v in self.groups().items()}

    def mark(self, rid, output):
        return ((self.marks.get(rid) or {}).get(output)) or None

    def default_track(self, rid):
        """The track to open first: the changed monitor, else the first changed
        A/B, else monitor."""
        tr = self.result(rid)['tracks']
        if tr.get('monitor', {}).get('result') == TRACK_DIFFERS:
            return 'monitor'
        for o in ('A', 'B'):
            if tr.get(o, {}).get('result') == TRACK_DIFFERS:
                return o
        return 'monitor'

    def summary(self):
        c = self.counts()
        return (f"differs {c[TRACK_DIFFERS]}  exact {c[TRACK_EXACT]}  "
                f"failed {c[TRACK_FAILED]}  unchecked {c[TRACK_UNCHECKED]}")


def finalize_run(run_dir, status, reason=''):
    """Close a report: the record in progress and every pending one become
    'unchecked' with `reason`; results already stored are kept.  A report that
    is already closed is left as it is (returns False)."""
    path = os.path.join(run_dir, 'run.json')
    doc = _read_json(path)
    if doc.get('status') != RUN_RUNNING:
        return False
    results = doc.setdefault('results', {})
    for rid in doc.get('records') or []:
        if rid not in results:
            results[rid] = dict(id=rid, title=rid, status=TRACK_UNCHECKED, reason=reason,
                                origin=None, inputs=None, seconds=None,
                                tracks={o: _track(TRACK_UNCHECKED, reason) for o in OUTPUTS})
    doc['current'] = None
    doc['status'] = status
    doc['reason'] = reason
    doc['finished'] = _dt.datetime.now().isoformat(timespec='seconds')
    _write_json(path, doc)
    return True


def _runtime_digest_default():
    return prov.digest(prov.manifest_of_tree(prov.PROJECT_ROOT))


class Verifier:
    def __init__(self, catalog):
        self.catalog = catalog
        self.root = os.path.join(catalog.root, RUNS_DIRNAME)

    # -- reports ------------------------------------------------------------------
    def run_ids(self):
        if not os.path.isdir(self.root):
            return []
        out = [n for n in os.listdir(self.root)
               if os.path.isfile(os.path.join(self.root, n, 'run.json'))]
        return sorted(out, reverse=True)

    def run_dir(self, run_id):
        return os.path.join(self.root, run_id)

    def load_run(self, run_id, active=False):
        d = self.run_dir(run_id)
        try:
            doc = _read_json(os.path.join(d, 'run.json'))
        except (OSError, json.JSONDecodeError) as e:
            raise VerifyError(f"{run_id}: unreadable report ({e})")
        if doc.get('format') != RUN_FORMAT:
            raise VerifyError(f"{run_id}: report format {doc.get('format')!r} not supported")
        marks = {}
        mp = os.path.join(d, 'marks.json')
        if os.path.isfile(mp):
            try:
                marks = _read_json(mp)
            except (OSError, json.JSONDecodeError):
                marks = {}
        return Run(d, doc, marks if isinstance(marks, dict) else {}, active=active)

    def latest(self):
        ids = self.run_ids()
        return self.load_run(ids[0]) if ids else None

    # -- listening marks -----------------------------------------------------------
    def set_mark(self, run_id, rid, output, mark):
        """mark = 'same' | 'different' | None (back to not rated).  Belongs to
        THIS report's pair of that track only."""
        if mark not in MARKS and mark is not None:
            raise VerifyError(f"unknown mark {mark!r}")
        if output not in OUTPUTS:
            raise VerifyError(f"unknown track {output!r}")
        run = self.load_run(run_id)
        t = run.result(rid)['tracks'].get(output) or {}
        if t.get('result') != TRACK_DIFFERS:
            raise VerifyError(f"{rid} {output}: a listening mark needs a recomputed pair "
                              f"({RESULT_LABELS.get(t.get('result'), 'not in this report')})")
        marks = run.marks
        per = marks.setdefault(rid, {})
        if mark is None:
            per.pop(output, None)
            if not per:
                marks.pop(rid, None)
        else:
            per[output] = mark
        _write_json(os.path.join(run.dir, 'marks.json'), marks)
        return marks

    # -- pair for listening ------------------------------------------------------------
    def pair(self, run_id, rid, output):
        """(saved_pcm, recomputed_pcm) of a differing track, both re-read and
        checked against the fingerprints stored at check time -- a replaced or
        lost file invalidates the pair (VerifyError) instead of playing
        something else under the report's verdict."""
        run = self.load_run(run_id)
        t = run.result(rid)['tracks'].get(output)
        if t is None or t['result'] != TRACK_DIFFERS:
            raise VerifyError(f"{rid} {output}: no recomputed pair in this report "
                              f"({RESULT_LABELS.get(t['result'], '?') if t else '?'})")
        try:
            rec = self.catalog.load(rid)
            ref = rec.pcm(output)
        except CatalogError as e:
            raise VerifyError(f"original WAV unavailable: {e}")
        if pcm_sha(ref) != t.get('ref_sha256'):
            raise VerifyError(f"{rid} {output}: the original WAV changed since the check")
        if not t.get('new_file'):
            raise VerifyError(f"{rid} {output}: recomputed WAV not stored")
        try:
            new = read_wav(os.path.join(run.dir, rid, t['new_file']))
        except CatalogError as e:
            raise VerifyError(f"recomputed WAV unavailable: {e}")
        if pcm_sha(new) != t.get('new_sha256'):
            raise VerifyError(f"{rid} {output}: the recomputed WAV changed since the check")
        return ref, new

    # -- the run (in-process) --------------------------------------------------------------
    def _new_run_doc(self, run_id, target, ids):
        return dict(format=RUN_FORMAT, id=run_id,
                    started=_dt.datetime.now().isoformat(timespec='seconds'), finished=None,
                    status=RUN_RUNNING, reason='',
                    target={k: target.get(k) for k in
                            ('schema', 'commit', 'digest', 'match', 'reason', 'root', 'repo_id',
                             'dirty', 'environment', 'audio')},
                    catalog_root=os.path.abspath(self.catalog.root),
                    records=list(ids), current=None, results={})

    def run(self, progress=None, cancel=None, on_record=None, provenance=None,
            runtime_digest=None, yield_cpu=True, on_result=None):
        """Check every record listed NOW, in catalog order (newest first).
        progress(frac) within a record, on_record(index, total, rid) before
        each, on_result(rid, status) after each, cancel() -> True stops.
        Returns the Run (its report is on disk after every record)."""
        target = provenance if provenance is not None else prov.current(
            audio=dict(sr=SR, block=BLOCK, channels=CHANNELS))
        runtime_digest = runtime_digest or _runtime_digest_default
        ids = self.catalog.ids()
        # microseconds keep same-second runs in creation order (ids sort by name)
        run_id = _dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f-') + uuid.uuid4().hex[:6]
        run_dir = self.run_dir(run_id)
        os.makedirs(run_dir, exist_ok=False)
        doc = self._new_run_doc(run_id, target, ids)
        path = os.path.join(run_dir, 'run.json')
        _write_json(path, doc)
        if not ids:
            doc.update(status=RUN_COMPLETE, reason="catalog is empty",
                       finished=_dt.datetime.now().isoformat(timespec='seconds'))
            _write_json(path, doc)
            return self.load_run(run_id)
        digest0 = runtime_digest()
        status, reason = RUN_COMPLETE, ''
        for i, rid in enumerate(ids):
            if cancel is not None and cancel():
                status, reason = RUN_CANCELLED, "cancelled"
                break
            if runtime_digest() != digest0:
                status, reason = RUN_STOPPED, "runtime set changed during the run"
                break
            doc['current'] = rid
            _write_json(path, doc)
            if on_record is not None:
                on_record(i, len(ids), rid)
            res = self._verify_one(rid, run_dir, progress, cancel, yield_cpu)
            if res is None:
                status, reason = RUN_CANCELLED, "cancelled"
                break
            if runtime_digest() != digest0:
                # the code may have changed while this one was computed:
                # its result is not attributable to one version -> unchecked
                status, reason = RUN_STOPPED, "runtime set changed during the run"
                break
            doc['results'][rid] = res
            doc['current'] = None
            _write_json(path, doc)
            if on_result is not None:
                on_result(rid, res['status'])
        finalize_run(run_dir, status, reason)
        return self.load_run(run_id)

    def _verify_one(self, rid, run_dir, progress, cancel, yield_cpu):
        """One record -> its report entry (None when cancelled)."""
        t0 = time.time()
        entry = dict(id=rid, title=rid, created='', origin=None, inputs=None, seconds=None,
                     status=TRACK_FAILED, reason='', tracks={})
        try:
            rec = self.catalog.load(rid)
        except CatalogError as e:
            entry['reason'] = str(e)
            entry['tracks'] = {o: _track(TRACK_FAILED, str(e)) for o in OUTPUTS}
            entry['seconds'] = time.time() - t0
            return entry
        entry['title'] = rec.title
        entry['created'] = rec.created
        entry['origin'] = dict(commit=rec.commit, digest=rec.digest, status=rec.status,
                               status_reason=rec.status_reason,
                               format=rec.meta.get('format'))
        # fingerprints of what the check used (unknown pieces stay None)
        inputs = dict(record_json=file_sha(os.path.join(rec.dir, 'record.json')), wav={},
                      snapshot={})
        for o in OUTPUTS:
            try:
                inputs['wav'][o] = file_sha(rec.wav_path(o))
            except (KeyError, TypeError):
                inputs['wav'][o] = None
        snap = rec.meta.get('snapshot') or {}
        for key in ('origin', 'end'):
            files = snap.get(key) or {}
            for name in (files.get('json'), files.get('npz')):
                if name:
                    inputs['snapshot'][name] = file_sha(os.path.join(rec.dir, name))
        entry['inputs'] = inputs
        # originals first (a bad WAV fails only its own track)
        ref, ref_err = {}, {}
        for o in OUTPUTS:
            try:
                arr, info = read_wav(rec.wav_path(o), with_info=True)
            except (CatalogError, KeyError, TypeError) as e:
                ref_err[o] = str(e)
                continue
            if info['sr'] != SR or info['channels'] != CHANNELS:
                ref_err[o] = (f"stored WAV format {info['sr']} Hz x {info['channels']} ch "
                              f"!= {SR} Hz x {CHANNELS} ch")
                continue
            ref[o] = arr
        rc = self.catalog.recompute(rid, progress=progress, cancel=cancel, yield_cpu=yield_cpu)
        if rc.status == 'cancelled':
            return None
        for o in OUTPUTS:
            if o in ref_err:
                entry['tracks'][o] = _track(TRACK_FAILED, ref_err[o])
            elif rc.status != 'ok':
                entry['tracks'][o] = _track(TRACK_FAILED, rc.reason)
            else:
                cmp = compare_pcm(ref[o], rc.pcm[o])
                verdict = cmp.pop('result')
                t = _track(verdict, '', ref_sha256=pcm_sha(ref[o]), **cmp)
                if verdict == TRACK_DIFFERS:
                    d = os.path.join(run_dir, rid)
                    try:
                        os.makedirs(d, exist_ok=True)
                        _n, sha = _pcm_to_wav(rc.pcm[o], os.path.join(d, f"{o}.wav"))
                        t['new_file'], t['new_sha256'] = f"{o}.wav", sha
                    except (OSError, CatalogError) as e:
                        t = _track(TRACK_FAILED, f"cannot store the recomputed WAV: {e}",
                                   ref_sha256=t['ref_sha256'])
                entry['tracks'][o] = t
        entry['status'] = record_status(entry['tracks'])
        if entry['status'] == TRACK_FAILED and rc.status != 'ok':
            entry['reason'] = rc.reason
        elif entry['status'] == TRACK_FAILED:
            entry['reason'] = "; ".join(f"{o}: {entry['tracks'][o]['reason']}"
                                        for o in OUTPUTS
                                        if entry['tracks'][o]['result'] == TRACK_FAILED)
        entry['seconds'] = time.time() - t0
        return entry

    # -- the run in a worker process (live UI) ----------------------------------------------
    def run_in_subprocess(self, progress=None, cancel=None, on_record=None, on_result=None):
        """run() executed by a child interpreter (its own provenance = the
        target).  JSON lines on stdout drive the callbacks; cancel() -> True
        terminates the child and closes the report as cancelled.  Returns the
        Run (or raises VerifyError when no report could be started)."""
        cmd = [sys.executable, '-X', 'utf8', '-m', 'casynth_lab.verify', 'run', self.catalog.root]
        root = prov.PROJECT_ROOT
        try:
            proc = subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, encoding='utf-8')
        except OSError as e:
            raise VerifyError(f"cannot start the check process: {e}")
        state = {'run_id': None, 'final': None}
        lines = []

        def reader():
            for line in proc.stdout:
                lines.append(line)
        th = threading.Thread(target=reader, daemon=True)
        th.start()
        seen = 0
        cancelled = False
        while True:
            th.join(timeout=0.05)
            while seen < len(lines):
                line = lines[seen].strip()
                seen += 1
                if not line.startswith('{'):
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'run_id' in msg and state['run_id'] is None:
                    state['run_id'] = msg['run_id']
                if 'progress' in msg and progress is not None:
                    progress(float(msg['progress']))
                elif 'record' in msg and on_record is not None:
                    on_record(int(msg['index']), int(msg['total']), msg['record'])
                elif 'done' in msg and on_result is not None:
                    on_result(msg['done'], msg.get('status'))
                elif 'final' in msg:
                    state['final'] = msg['final']
            if not th.is_alive() and proc.poll() is not None:
                break
            if not cancelled and cancel is not None and cancel():
                cancelled = True
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        th.join(timeout=1.0)
        run_id = state['run_id']
        err = proc.stderr.read().strip().splitlines()
        if run_id is None:
            raise VerifyError("check process failed before starting a report: "
                              + (err[-1] if err else f"exit code {proc.returncode}"))
        if cancelled:
            finalize_run(self.run_dir(run_id), RUN_CANCELLED, "cancelled")
        elif state['final'] is None:
            finalize_run(self.run_dir(run_id), RUN_FAILED,
                         "check process failed: " + (err[-1] if err else f"exit {proc.returncode}"))
        return self.load_run(run_id)


def _cli(argv):
    """python -m casynth_lab.verify run ROOT      (JSON lines on stdout)"""
    if len(argv) != 2 or argv[0] != 'run':
        print("usage: python -m casynth_lab.verify run CATALOG_ROOT")
        return 2
    cat = Catalog(argv[1])
    ver = Verifier(cat)

    def emit(**kw):
        print(json.dumps(kw, ensure_ascii=False), flush=True)
    doc = prov.current(audio=dict(sr=SR, block=BLOCK, channels=CHANNELS))
    emit(provenance={k: doc.get(k) for k in ('commit', 'digest', 'match', 'reason', 'root')})
    real_new = ver._new_run_doc

    def announcing(run_id, target, ids):
        emit(run_id=run_id, total=len(ids), target=target_label(target))
        return real_new(run_id, target, ids)
    ver._new_run_doc = announcing
    run = ver.run(progress=lambda f: emit(progress=f),
                  on_record=lambda i, n, rid: emit(record=rid, index=i, total=n),
                  on_result=lambda rid, st: emit(done=rid, status=st),
                  provenance=doc, yield_cpu=False)
    emit(final=run.status, run_id=run.id, counts=run.counts())
    return 0


if __name__ == '__main__':
    sys.exit(_cli(sys.argv[1:]))
