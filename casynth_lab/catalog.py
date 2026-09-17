"""Local experiment catalog (S4): save a Cut as a record, list / load records,
read their WAVs, and REPLAY a record from the start with a byte-exact check.

Layout (default root lab_catalog/local/, user data, gitignored):
    <root>/<id>/record.json      metadata + embedded scene + applied journal
    <root>/<id>/A.wav B.wav monitor.wav
    <root>/<id>/state_end.json + .npz     (S5) runner snapshot at the cut end
    <root>/<id>/state_origin.json + .npz  (S5) the snapshot a branch started from
    <root>/<id>/replay/...       last replay output (originals never touched)
    <root>/.tmp/<session>/       streaming raw PCM of live sessions

Integrity: a record is assembled in <root>/<id>.partial/ and renamed to
<root>/<id>/ only when complete; an error leaves no partial record behind.
Listing tolerates corrupted entries (they are reported, not fatal).
No window, no device: everything here is testable headless.
"""
import copy
import datetime as _dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
import wave

import numpy as np

from casynth_config import SR, VOL_DEFAULT
from . import registry
from .runner import BLOCK, CHANNELS, OUTPUTS, DemoRunner, RUNNER_STATE_VERSION
from .scene import SceneError, scene_from_doc
from .snapshot import SnapshotError, save_state, load_state
from . import provenance as prov
from . import versions

RECORD_FORMAT = 3                         # 3 = S6 (provenance, pinned/local status)
RECORD_FORMATS_READ = (1, 2, 3)           # S4/S5 records open without migration
STATUS_PINNED, STATUS_LOCAL = 'pinned', 'local'
STATE_END, STATE_ORIGIN = 'state_end', 'state_origin'
DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'lab_catalog', 'local')
REPLAY_COMMANDS_SKIP = ('step',)          # journal entries computed by the CA, not input
_WAV_CHUNK_FRAMES = SR                    # 1 s per streaming chunk


class CatalogError(RuntimeError):
    pass


def _versions():
    v = {'python': platform.python_version()}
    for mod in ('numpy', 'scipy', 'sounddevice', 'pygame'):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:                      # noqa: BLE001
            v[mod] = 'n/a'
    return v


def read_wav(path, with_info=False):
    """int16 (n, channels) array of a record WAV (CatalogError if unreadable).
    with_info=True -> (array, {'sr', 'channels', 'sampwidth'}) -- the format
    as stored, so a caller can refuse an incomparable pair explicitly."""
    try:
        with wave.open(path, 'rb') as w:
            if w.getsampwidth() != 2:
                raise CatalogError(f"{path}: not 16-bit")
            n, ch, sr = w.getnframes(), w.getnchannels(), w.getframerate()
            data = w.readframes(n)
    except (wave.Error, EOFError, OSError) as e:
        raise CatalogError(f"{path}: unreadable WAV ({e})")
    arr = np.frombuffer(data, np.int16)
    if len(arr) != n * ch:
        raise CatalogError(f"{path}: truncated WAV")
    arr = arr.reshape(-1, ch)
    if with_info:
        return arr, {'sr': sr, 'channels': ch, 'sampwidth': 2}
    return arr


def _pcm_to_wav(pcm, wav_path):
    """Write an int16 (n, 2) array to a WAV in 1 s chunks; returns (frames, sha256)."""
    pcm = np.ascontiguousarray(pcm, dtype=np.int16)
    if pcm.ndim != 2 or pcm.shape[1] != CHANNELS:
        raise CatalogError(f"{wav_path}: bad PCM shape {pcm.shape}")
    h = hashlib.sha256()
    with wave.open(wav_path, 'wb') as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(2)
        w.setframerate(SR)
        for i in range(0, len(pcm), _WAV_CHUNK_FRAMES):
            chunk = pcm[i:i + _WAV_CHUNK_FRAMES].tobytes()
            w.writeframes(chunk)
            h.update(chunk)
    return len(pcm), h.hexdigest()


def _effective_scene(scene_doc, state):
    """The record's conditions: the demo scene with the field, both sides and
    the selected side as they were at the recording origin."""
    d = copy.deepcopy(scene_doc)
    d['format'] = 2
    d.pop('engine_id', None)
    d.pop('engine_params', None)
    d['cells'] = [list(c) for c in state['cells']]
    d['variants'] = {side: {'engine_id': v['engine_id'],
                            'engine_params': dict(v['engine_params'])}
                     for side, v in state['settings'].items()}
    d['initial_side'] = state['selected']
    if state.get('factory_variants'):
        d['factory_variants'] = copy.deepcopy(state['factory_variants'])
    if state.get('param_memory'):
        d['param_memory'] = copy.deepcopy(state['param_memory'])
    if state.get('param_ranges'):
        d['param_ranges'] = copy.deepcopy(state['param_ranges'])
    d.setdefault('listen', '')
    return d


def replace_retry(src, dst, attempts=20):
    """os.replace (file or directory); Windows may refuse transiently (indexer /
    antivirus holding a handle) -> retry briefly before giving up."""
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.05 * (i + 1))


_replace_dir = replace_retry


class Record:
    """A loaded record.json (WAVs are read on demand)."""

    def __init__(self, root, rid, meta):
        self.root, self.id, self.meta = root, rid, meta

    @property
    def dir(self):
        return os.path.join(self.root, self.id)

    def wav_path(self, output):
        return os.path.join(self.dir, self.meta['audio'][output]['file'])

    def pcm(self, output):
        return read_wav(self.wav_path(output))

    @property
    def title(self):
        return self.meta.get('title', self.id)

    @property
    def note(self):
        """The description typed at Save (record.json 'note'); never edited later."""
        return self.meta.get('note', '')

    # -- listening notes (2026-09-14) ----------------------------------------------
    NOTES_FILE = 'notes.md'

    @property
    def notes_path(self):
        return os.path.join(self.dir, self.NOTES_FILE)

    @property
    def notes(self):
        """Free-text notes attached to the record (<record>/notes.md, UTF-8):
        the user's listening feedback, written in the bench (Notes button) and
        read back by the agents (`python -m casynth_lab.catalog notes ROOT`).
        Separate from the Save description; WAVs / snapshots / record.json are
        never touched by it."""
        try:
            with open(self.notes_path, encoding='utf-8') as f:
                return f.read()
        except OSError:
            return ''

    @property
    def has_notes(self):
        return bool(self.notes.strip())

    @property
    def created(self):
        return self.meta.get('created', '')

    @property
    def seconds(self):
        return (self.meta['end_sample'] - self.meta['audio_start_sample']) / SR

    @property
    def engines(self):
        e = self.meta.get('engines', {})
        return e.get('A', '?'), e.get('B', '?')

    def engine_labels(self):
        out = []
        for eid in self.engines:
            try:
                out.append(registry.label(eid))
            except KeyError:
                out.append(f"{eid} (missing)")
        return tuple(out)

    # -- S6 --------------------------------------------------------------------
    @property
    def provenance(self):
        p = self.meta.get('provenance')
        return p if isinstance(p, dict) else None

    @property
    def status(self):
        """'pinned' | 'local'.  Records before S6 carry status 'local' with no
        provenance; nothing is attributed to them after the fact."""
        p = self.provenance
        if self.meta.get('status') == STATUS_PINNED and p and p.get('commit')                 and p.get('digest') and (self.meta.get('pin') or {}).get('commit') == p['commit']:
            return STATUS_PINNED
        return STATUS_LOCAL

    @property
    def status_reason(self):
        if self.status == STATUS_PINNED:
            return ''
        if not isinstance(self.provenance, dict) or not self.provenance:
            return "code version not recorded"
        return self.meta.get('status_reason') or prov.status_of(self.provenance)[1]

    @property
    def commit(self):
        p = self.provenance or {}
        return p.get('commit')

    @property
    def digest(self):
        p = self.provenance or {}
        return p.get('digest')

    def version_label(self):
        if self.status == STATUS_PINNED:
            return f"Pinned {prov.short(self.commit)}"
        return f"Local: {self.status_reason}"

    # -- S5 --------------------------------------------------------------------
    @property
    def parent_id(self):
        return self.meta.get('parent_record_id')

    @property
    def origin_kind(self):
        return self.meta.get('origin_kind') or 'fresh'

    def can_continue(self):
        """(ok, reason) -- whether "Continue" is offered.  A light check
        (files, versions, engines); the full validation happens on load."""
        for side, eid in self.meta.get('engines', {}).items():
            if eid not in registry.REGISTRY:
                return False, f"engine {eid!r} ({side}) is not registered"
        snap = self.meta.get('snapshot') or {}
        end = snap.get('end')
        if self.meta.get('format', 1) < 2 or not end:
            why = snap.get('end_reason') or "record has no end snapshot (S4 record)"
            return False, why
        if snap.get('runner_state_version') != RUNNER_STATE_VERSION:
            return False, (f"snapshot version {snap.get('runner_state_version')} "
                           f"!= {RUNNER_STATE_VERSION}")
        for name in (end.get('json'), end.get('npz')):
            if not name or not os.path.isfile(os.path.join(self.dir, name)):
                return False, f"snapshot file missing: {name}"
        return True, ""


def end_state_by_replay(meta, progress=None):
    """Run the record's journal (no audio comparison) and return the runner's
    end state as stored by newer records: cells, gen, vol."""
    try:
        scene = scene_from_doc(meta['scene'])
        runner = DemoRunner(scene, vol=meta.get('runner', {}).get('vol_initial', VOL_DEFAULT))
    except (SceneError, KeyError, TypeError, ValueError) as e:
        raise CatalogError(f"cannot rebuild the end state: {e}")
    origin = meta.get('origin_sample', 0)
    end = meta['end_sample'] - origin
    cmds = [dict(j, out_sample=j['out_sample'] - origin) for j in meta['journal']
            if j['kind'] not in REPLAY_COMMANDS_SKIP and j['out_sample'] >= origin]
    cmds.sort(key=lambda j: (j['out_sample'], j['seq']))
    i = 0
    try:
        while runner.out_samples < end:
            while i < len(cmds) and cmds[i]['out_sample'] <= runner.out_samples:
                c = cmds[i]
                runner.post(c['kind'], at=None, **c['args'])
                i += 1
            before = runner.out_samples
            runner.next_block()
            if progress is not None and (before // BLOCK) % 50 == 0:
                progress(min(before / max(end, 1), 1.0))
    except Exception as e:                     # noqa: BLE001
        raise CatalogError(f"cannot rebuild the end state: {e}")
    if progress is not None:
        progress(1.0)
    return dict(cells=[[int(a), int(b)] for a, b in np.argwhere(runner.grid > 0)],
                gen=int(runner.gen), vol=float(runner.vol))


def bench_scene(rec, state=None):
    """Scene document + volume that put the bench into the record's END state
    (field, engines/params of both sides, selected side).  The user starts it
    manually.  CatalogError for records without state_at_end."""
    meta = rec.meta
    st = state or meta.get('state_at_end')
    if not st:
        # older record (before state_at_end): recompute the end state from
        # the embedded conditions + journal (slow: a UI should rather call
        # Catalog.end_state_in_subprocess and pass the result as `state`)
        st = end_state_by_replay(meta)
    if 'settings_at_end' not in meta:
        raise CatalogError(f"{rec.id}: record has no end settings to open")
    d = copy.deepcopy(meta['scene'])
    d['format'] = 2
    d.pop('engine_id', None)
    d.pop('engine_params', None)
    d['id'] = f"{d.get('id', 'demo')}@{rec.id}"
    d['title'] = rec.title
    d['cells'] = [list(c) for c in st['cells']]
    d['variants'] = {side: {'engine_id': v['engine_id'],
                            'engine_params': dict(v['engine_params'])}
                     for side, v in meta['settings_at_end'].items()}
    d['initial_side'] = meta.get('selected_at_end', 'A')
    fv = st.get('factory_variants') or (meta.get('scene') or {}).get('factory_variants')
    if fv:
        d['factory_variants'] = copy.deepcopy(fv)
    if st.get('param_memory'):
        d['param_memory'] = copy.deepcopy(st['param_memory'])
    if st.get('param_ranges'):
        d['param_ranges'] = copy.deepcopy(st['param_ranges'])
    d.setdefault('listen', '')
    try:
        scene = scene_from_doc(d)
    except SceneError as e:
        raise CatalogError(f"{rec.id}: cannot open in bench: {e}")
    return scene, float(st.get('vol', VOL_DEFAULT))


class RecomputeResult:
    """Output of Catalog.recompute(): the three tracks of the saved window as
    the current code renders them (no comparison, no files)."""

    def __init__(self, status, reason='', pcm=None):
        self.status = status              # 'ok' | 'unavailable' | 'cancelled'
        self.reason = reason
        self.pcm = pcm or {}              # output -> int16 (n, 2)


class ReplayResult:
    def __init__(self, status, reason='', outputs=None, replay_dir=None):
        self.status = status              # 'match' | 'mismatch' | 'unavailable' | 'cancelled'
        self.reason = reason
        self.outputs = outputs or {}      # output -> bool (equal)
        self.replay_dir = replay_dir

    @property
    def text(self):
        return {'match': "Replay matched",
                'mismatch': "Replay differs: " + ", ".join(
                    o for o, ok in self.outputs.items() if not ok),
                'unavailable': "Replay unavailable: " + self.reason,
                'cancelled': "Replay cancelled"}[self.status]


class Catalog:
    def __init__(self, root=DEFAULT_ROOT, repo_root=prov.PROJECT_ROOT):
        self.root = root
        self.tmp_root = os.path.join(root, '.tmp')
        self.repo_root = repo_root          # the Git repository pins refer to (None = never pin)

    # -- listing / loading ----------------------------------------------------
    def ids(self):
        if not os.path.isdir(self.root):
            return []
        out = []
        for name in os.listdir(self.root):
            if name.startswith('.') or name.endswith('.partial'):
                continue
            if os.path.isfile(os.path.join(self.root, name, 'record.json')):
                out.append(name)
        return sorted(out, reverse=True)      # id starts with a timestamp -> newest first

    def load(self, rid):
        path = os.path.join(self.root, rid, 'record.json')
        try:
            with open(path, encoding='utf-8') as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise CatalogError(f"{rid}: unreadable record.json ({e})")
        for key in ('format', 'scene', 'journal', 'audio', 'audio_start_sample', 'end_sample'):
            if key not in meta:
                raise CatalogError(f"{rid}: record.json missing '{key}'")
        if meta['format'] not in RECORD_FORMATS_READ:
            raise CatalogError(f"{rid}: record format {meta['format']} not supported")
        return Record(self.root, rid, meta)

    def parent_of(self, rec):
        """(title, present) of the record's parent; title = id when the
        parent is gone (a missing parent never breaks the catalog)."""
        pid = rec.parent_id
        if not pid:
            return None, False
        try:
            return self.load(pid).title, True
        except CatalogError:
            return pid, False

    # -- provenance / pinning (S6) ----------------------------------------------------
    def _pin_status(self, doc):
        """(status, reason, pin) for a provenance doc: pinned only when the
        sound set matched a commit AND the holding ref exists (created here;
        any failure -> local with the reason, never a false pin)."""
        status, reason = prov.status_of(doc)
        if status != STATUS_PINNED:
            return STATUS_LOCAL, reason, None
        if self.repo_root is None:
            return STATUS_LOCAL, "pinning disabled for this catalog", None
        commit = doc['commit']
        try:
            prov.hold_commit(self.repo_root, commit)
        except prov.ProvenanceError as e:
            return STATUS_LOCAL, f"could not hold commit {prov.short(commit)}: {e}", None
        return STATUS_PINNED, '', {'commit': commit, 'ref': prov.pin_ref(commit)}

    def write_notes(self, rid, text):
        """Replace the record's notes.md atomically (empty text removes it).
        CatalogError when the record does not exist or cannot be written."""
        rec = self.load(rid)
        path = rec.notes_path
        tmp = path + '.tmp'
        try:
            if not (text or '').strip():
                if os.path.exists(path):
                    os.remove(path)
                return
            with open(tmp, 'w', encoding='utf-8', newline='\n') as f:
                f.write(text if text.endswith('\n') else text + '\n')
            replace_retry(tmp, path)
        except OSError as e:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise CatalogError(f"{rid}: cannot write notes: {e}")

    def notes_report(self, with_description=False):
        """[(Record, notes)] newest first for every record that has notes
        (with_description: also records whose Save description is non-empty)."""
        out = []
        for rec, _err in self.list():
            if rec is None:
                continue
            if rec.has_notes or (with_description and rec.note.strip()):
                out.append((rec, rec.notes))
        return out

    def _rewrite_meta(self, rec, meta):
        """Replace record.json atomically (WAVs, snapshots, id untouched)."""
        path = os.path.join(rec.dir, 'record.json')
        tmp = path + '.tmp'
        try:
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=1)
            replace_retry(tmp, path)
        except OSError as e:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise CatalogError(f"{rec.id}: cannot update record.json: {e}")

    def find_commit_for_digest(self, target, limit=200):
        """A commit (HEAD first, then recent history) whose sound set has
        exactly the digest `target`; None if none."""
        try:
            commits = prov.git(self.repo_root, 'rev-list', f'-n{limit}', 'HEAD').split()
        except prov.ProvenanceError:
            return None
        for c in commits:
            try:
                if prov.digest_at_commit(self.repo_root, c) == target:
                    return c
            except prov.ProvenanceError:
                continue
        return None

    def pin(self, rid, commit=None):
        """Pin a local record to a commit whose sound set has EXACTLY the
        fingerprint the record was made with.  Only provenance/status fields
        change (id, WAVs, conditions, journal, snapshots stay).  CatalogError
        (and no change) when nothing matches or Git/metadata fail."""
        rec = self.load(rid)
        if rec.status == STATUS_PINNED:
            return rec.commit
        if self.repo_root is None:
            raise CatalogError(f"{rid}: pinning disabled for this catalog")
        doc = rec.provenance
        if not doc or not doc.get('digest') or not doc.get('manifest'):
            raise CatalogError(f"{rid}: code fingerprint not recorded -- cannot pin")
        if commit is None:
            commit = self.find_commit_for_digest(doc['digest'])
            if commit is None:
                raise CatalogError(f"{rid}: no commit has this code fingerprint "
                                   f"({doc['digest'][:12]}); commit the exact runtime "
                                   f"files first or save a new experiment")
        else:
            try:
                same = prov.digest_at_commit(self.repo_root, commit) == doc['digest']
            except prov.ProvenanceError as e:
                raise CatalogError(f"{rid}: cannot read commit {prov.short(commit)}: {e}")
            if not same:
                raise CatalogError(f"{rid}: commit {prov.short(commit)} has different "
                                   f"runtime files -- matching audio does not prove the code")
        rid_repo, _head = prov.repo_identity(self.repo_root)
        if doc.get('repo_id') and rid_repo and doc['repo_id'] != rid_repo:
            raise CatalogError(f"{rid}: record comes from another repository")
        try:
            prov.hold_commit(self.repo_root, commit)
        except prov.ProvenanceError as e:
            raise CatalogError(f"{rid}: cannot hold commit {prov.short(commit)}: {e}")
        meta = copy.deepcopy(rec.meta)
        p = dict(doc)
        p.update(commit=commit, match='clean', dirty=[], reason='', repo_id=rid_repo)
        meta['provenance'] = p
        meta['status'] = STATUS_PINNED
        meta['status_reason'] = ''
        meta['pin'] = {'commit': commit, 'ref': prov.pin_ref(commit),
                       'pinned_at': _dt.datetime.now().isoformat(timespec='seconds')}
        self._rewrite_meta(rec, meta)
        return commit

    def source_plan(self, rec, current=None):
        """How "Continue" would run this record's code.
        -> dict(mode='in-process'|'worktree'|None, label, reason, commit).
        Pinned: same clean version here -> in-process; else a child bench of the
        held commit (needs the commit and a compatible environment).  Local:
        current code by the S5 rules, labelled as such."""
        current = current or prov.current()
        if rec.status == STATUS_PINNED:
            commit = rec.commit
            if current.get('match') == 'clean' and current.get('digest') == rec.digest:
                # identical sound set (the commit may differ: code is the truth)
                return dict(mode='in-process', commit=commit, reason='', differs=[],
                            label=f"same sound code as version {prov.short(commit)}")
            if self.repo_root is None:
                return dict(mode=None, commit=commit, label='',
                            reason="other version: this catalog has no repository")
            # the stored digest may come from an older set definition or an
            # unrelated commit: compare the SOUND CODE at the record's commit
            # with the sound code running here (the content is the truth)
            differs = self._sound_difference(commit, current)
            if differs is not None and not differs and current.get('match') == 'clean':
                return dict(mode='in-process', commit=commit, reason='', differs=[],
                            label=f"same sound code as version {prov.short(commit)}")
            ok, why = prov.environment_compatible((rec.provenance or {}).get('environment'))
            if not ok:
                return dict(mode=None, commit=commit, label='', differs=differs or [],
                            reason=f"no compatible environment: {why}")
            if not prov.commit_exists(self.repo_root, commit):
                return dict(mode=None, commit=commit, label='',
                            reason=f"commit {prov.short(commit)} is not available")
            what = (", ".join(differs[:4]) + (" ..." if len(differs) > 4 else "")) if differs \
                else "this checkout is not a clean commit"
            return dict(mode='worktree', commit=commit, differs=differs or [],
                        reason=f"sound code differs: {what}",
                        label=f"separate bench of version {prov.short(commit)}")
        return dict(mode='in-process', commit=rec.commit, reason='',
                    label="local record / current code")

    def _sound_difference(self, commit, current):
        """Sound-set paths whose content differs between `commit` and the
        running code (None when the commit cannot be read); cached per commit."""
        cache = self.__dict__.setdefault('_manifest_cache', {})
        if commit not in cache:
            try:
                cache[commit] = prov.manifest_at_commit(self.repo_root, commit)
            except prov.ProvenanceError:
                cache[commit] = None
        ref = cache[commit]
        if ref is None:
            return None
        return prov.manifest_difference(ref, current.get('manifest') or {})

    def worktree_for(self, rec):
        """Verified checkout of the record's pinned commit (VersionError)."""
        return versions.ensure_worktree(self.repo_root, self.root, rec.commit)

    def run_version(self, rec, action, extra=(), on_line=None):
        """Start a child bench of the record's version (see versions.py)."""
        wt = self.worktree_for(rec)
        return versions.ChildBench(wt, self.root, rec.id, action, extra, on_line).start()

    # -- continuation (S5) ------------------------------------------------------------
    def end_state(self, rid):
        """The END snapshot state tree of a record (CatalogError if absent /
        unreadable)."""
        rec = self.load(rid)
        ok, why = rec.can_continue()
        if not ok:
            raise CatalogError(f"{rid}: cannot continue: {why}")
        try:
            return load_state(rec.dir, STATE_END)
        except SnapshotError as e:
            raise CatalogError(f"{rid}: snapshot unreadable: {e}")

    def continue_runner(self, rid):
        """(runner, pristine_state, record): a DemoRunner restored from the
        record's end snapshot, fully validated -- nothing else is touched, so
        the caller's current session stays intact on failure (CatalogError).
        The clocks of the restored runner do not advance until it is driven."""
        rec = self.load(rid)
        state = self.end_state(rid)
        try:
            runner = DemoRunner.from_state(state)
        except (ValueError, KeyError, TypeError) as e:
            raise CatalogError(f"{rid}: cannot continue: {e}")
        return runner, state, rec

    def list(self):
        """[(Record or None, error or None)] newest first; a broken record
        does not break the list."""
        out = []
        for rid in self.ids():
            try:
                out.append((self.load(rid), None))
            except CatalogError as e:
                out.append((None, f"{rid}: {e}"))
        return out

    # -- saving ---------------------------------------------------------------
    @staticmethod
    def default_title(scene_title, when=None):
        when = when or _dt.datetime.now()
        return f"{scene_title} {when.strftime('%Y-%m-%d %H:%M:%S')}"

    def save(self, cut, title=None, note=''):
        """Assemble a record from a Cut.  Returns the new id.  Any failure
        raises CatalogError and leaves no partial record."""
        if cut is None:
            raise CatalogError("nothing to save yet: release Pause CA first")
        if cut.record_error:
            raise CatalogError(f"recording failed: {cut.record_error}")
        if cut.n_frames <= 0:
            raise CatalogError("nothing recorded yet")
        now = _dt.datetime.now()
        rid = now.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]
        scene_title = cut.scene_doc.get('title', 'demo')
        title = (title or '').strip() or self.default_title(scene_title, now)
        partial = os.path.join(self.root, rid + '.partial')
        final = os.path.join(self.root, rid)
        try:
            os.makedirs(partial, exist_ok=False)
            audio = {}
            for o in OUTPUTS:
                frames, sha = _pcm_to_wav(cut.pcm[o], os.path.join(partial, f"{o}.wav"))
                if frames != cut.n_frames:
                    raise CatalogError(f"{o}: wrote {frames} frames, expected {cut.n_frames}")
                audio[o] = {'file': f"{o}.wav", 'samples': frames, 'sha256': sha}
            snapshot = {'runner_state_version': RUNNER_STATE_VERSION,
                        'end': None, 'end_reason': '', 'origin': None, 'origin_seq': None}
            origin_kind = getattr(cut, 'origin_kind', None) or 'fresh'
            if getattr(cut, 'end_snapshot', None) is not None:
                j, a = save_state(partial, STATE_END, cut.end_snapshot)
                snapshot['end'] = {'json': j, 'npz': a}
                snapshot['engine_state_versions'] = {
                    side: s.get('engine_state_version')
                    for side, s in cut.end_snapshot['sides'].items()}
            else:
                snapshot['end_reason'] = (getattr(cut, 'end_snapshot_reason', '')
                                          or "engine without snapshot support")
            if origin_kind == 'snapshot':
                if getattr(cut, 'origin_snapshot', None) is None:
                    raise CatalogError("snapshot origin without a snapshot")
                j, a = save_state(partial, STATE_ORIGIN, cut.origin_snapshot)
                snapshot['origin'] = {'json': j, 'npz': a}
                snapshot['origin_seq'] = int(cut.origin_snapshot['seq'])
            doc = getattr(cut, 'provenance', None)
            status, status_reason, pin = self._pin_status(doc) if doc else \
                (STATUS_LOCAL, "code version not recorded", None)
            meta = {
                'format': RECORD_FORMAT,
                'id': rid,
                'provenance': doc,
                'status_reason': status_reason,
                'pin': pin,
                'origin_kind': origin_kind,
                'parent_record_id': getattr(cut, 'parent_record_id', None),
                'snapshot': snapshot,
                'created': now.isoformat(timespec='seconds'),
                'title': title,
                'note': note or '',
                'status': status,
                'scene': _effective_scene(cut.scene_doc, cut.origin_state),
                'demo_scene': cut.scene_doc,
                'origin_sample': cut.origin_sample,
                'origin_state': cut.origin_state,
                'window_seconds': cut.window_seconds,
                'runner': dict(cut.runner_settings, vol_initial=cut.vol_initial,
                               master_gain_level=cut.scene_doc.get('audio', {}).get('level', 1.0)),
                'versions': _versions(),
                'journal': [{'out_sample': t, 'seq': s, 'kind': k, 'args': a}
                            for (t, s, k, a) in cut.journal],
                'audio_start_sample': cut.audio_start_sample,
                'end_sample': cut.end_sample,
                'audio': audio,
                'engines': {side: eid for side, (eid, _p) in cut.side_settings.items()},
                'settings_at_end': {side: {'engine_id': eid, 'engine_params': p}
                                    for side, (eid, p) in cut.side_settings.items()},
                'selected_at_end': cut.selected,
                'state_at_end': cut.state_at_end,
                'diagnostics': cut.diagnostics,
            }
            tmp_json = os.path.join(partial, 'record.json')
            with open(tmp_json, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=1)
            _replace_dir(partial, final)
        except Exception as e:                 # noqa: BLE001
            shutil.rmtree(partial, ignore_errors=True)
            if isinstance(e, CatalogError):
                raise
            raise CatalogError(f"save failed: {e}")
        return rid

    # -- recompute / replay -------------------------------------------------------
    def recompute(self, rid, progress=None, cancel=None, yield_cpu=True):
        """THE journal replay: rebuild the record's start conditions (embedded
        scene, or the origin snapshot of a branch), apply the recorded commands
        with the current code and render exactly the saved window
        [audio_start_sample, end_sample) of A, B and monitor.  Nothing is
        written.  -> RecomputeResult(status 'ok' | 'unavailable' | 'cancelled',
        reason, pcm={output: int16 (n, 2)}).  Both the single check (replay)
        and the catalog check (verify) use this one path."""
        try:
            rec = self.load(rid)
        except CatalogError as e:
            return RecomputeResult('unavailable', str(e))
        meta = rec.meta
        rs = meta.get('runner', {})
        if (rs.get('sr', SR) != SR or rs.get('block', BLOCK) != BLOCK
                or rs.get('channels', CHANNELS) != CHANNELS):
            return RecomputeResult('unavailable',
                                   f"render settings differ (SR/BLOCK/channels): {rs}")
        origin = meta.get('origin_sample', 0)
        if (meta.get('origin_kind') or 'fresh') == 'snapshot':
            # branch: restore the ORIGIN snapshot (clocks included -> absolute
            # times); commands the snapshot already had queued (seq <= origin
            # seq) are re-applied by the runner itself, not from the journal
            snap = meta.get('snapshot') or {}
            if snap.get('runner_state_version') != RUNNER_STATE_VERSION or not snap.get('origin'):
                return RecomputeResult('unavailable',
                                       "origin snapshot missing or of another version")
            try:
                runner = DemoRunner.from_state(load_state(rec.dir, STATE_ORIGIN))
            except (SnapshotError, ValueError, KeyError, TypeError) as e:
                return RecomputeResult('unavailable', f"origin snapshot cannot be restored: {e}")
            if runner.out_samples != origin:
                return RecomputeResult('unavailable', "origin snapshot clock != origin_sample")
            origin_seq = snap.get('origin_seq') or 0
            shift = 0
            keep = lambda j: j['seq'] > origin_seq          # noqa: E731
        else:
            try:
                scene = scene_from_doc(meta['scene'])
            except (SceneError, KeyError, TypeError, ValueError) as e:
                return RecomputeResult('unavailable', f"scene cannot be rebuilt: {e}")
            try:
                runner = DemoRunner(scene, vol=rs.get('vol_initial', VOL_DEFAULT))
            except (ValueError, KeyError) as e:
                return RecomputeResult('unavailable', f"engine unavailable: {e}")
            shift = origin
            keep = lambda j: True                            # noqa: E731
        start, end = meta['audio_start_sample'] - shift, meta['end_sample'] - shift
        try:
            cmds = [dict(j, out_sample=j['out_sample'] - shift) for j in meta['journal']
                    if j['kind'] not in REPLAY_COMMANDS_SKIP and j['out_sample'] >= origin
                    and keep(j)]
            cmds.sort(key=lambda j: (j['out_sample'], j['seq']))
        except (KeyError, TypeError) as e:
            return RecomputeResult('unavailable', f"journal unreadable: {e}")
        got = {o: [] for o in OUTPUTS}
        i = 0
        total = max(end - runner.out_samples, 1)
        first = runner.out_samples
        try:
            while runner.out_samples < end:
                # feed the commands of THIS boundary just before the block, in
                # journal order (stop/reset clear only what is queued later)
                while i < len(cmds) and cmds[i]['out_sample'] <= runner.out_samples:
                    c = cmds[i]
                    runner.post(c['kind'], at=None, **c['args'])
                    i += 1
                before = runner.out_samples
                blk = runner.next_block()
                if before >= start:
                    for o in OUTPUTS:
                        got[o].append(blk.get(o))
                if cancel is not None and cancel():
                    return RecomputeResult('cancelled')
                if progress is not None and (before // BLOCK) % 50 == 0:
                    progress(min((before - first) / total, 1.0))
                if yield_cpu and (before // BLOCK) % 4 == 0:
                    time.sleep(0.002)     # leave the GIL to a live render thread
        except Exception as e:                 # noqa: BLE001
            return RecomputeResult('unavailable', f"replay failed: {e}")
        if i < len(cmds):
            return RecomputeResult('unavailable', f"{len(cmds) - i} journal commands "
                                                  f"beyond the recorded end")
        n = end - start
        pcm = {o: (np.concatenate(got[o], axis=0)[:n] if got[o]
                   else np.zeros((0, CHANNELS), np.int16)) for o in OUTPUTS}
        if progress is not None:
            progress(1.0)
        return RecomputeResult('ok', pcm=pcm)

    def replay(self, rid, progress=None, cancel=None, yield_cpu=True):
        """Single check (S4): recompute() + byte-exact comparison with the
        stored WAVs.  Writes <id>/replay/*.wav (the last single check's
        output; the originals are never modified)."""
        rc = self.recompute(rid, progress=progress, cancel=cancel, yield_cpu=yield_cpu)
        if rc.status != 'ok':
            return ReplayResult(rc.status, rc.reason)
        rec = self.load(rid)
        outputs, replay_dir = {}, os.path.join(rec.dir, 'replay')
        try:
            os.makedirs(replay_dir, exist_ok=True)
            for o in OUTPUTS:
                pcm = rc.pcm[o]
                with wave.open(os.path.join(replay_dir, f"{o}.wav"), 'wb') as w:
                    w.setnchannels(CHANNELS)
                    w.setsampwidth(2)
                    w.setframerate(SR)
                    w.writeframes(np.ascontiguousarray(pcm).tobytes())
                try:
                    ref = rec.pcm(o)
                except CatalogError as e:
                    return ReplayResult('unavailable', str(e), replay_dir=replay_dir)
                outputs[o] = bool(ref.shape == pcm.shape and np.array_equal(ref, pcm))
        except OSError as e:
            return ReplayResult('unavailable', f"cannot write replay output: {e}")
        status = 'match' if all(outputs.values()) else 'mismatch'
        return ReplayResult(status, outputs=outputs, replay_dir=replay_dir)


    # -- replay in a separate process (live UI) ------------------------------
    def end_state_in_subprocess(self, rid, progress=None, cancel=None):
        """end_state_by_replay() in a child interpreter (see replay_in_subprocess).
        Returns the state dict, or raises CatalogError."""
        res = self._run_child('end_state', rid, progress, cancel)
        if res.status == 'cancelled':
            raise CatalogError("cancelled")
        if res.status != 'match':
            raise CatalogError(res.reason)
        return res.outputs            # the state dict travels in 'outputs'

    def replay_in_subprocess(self, rid, progress=None, cancel=None):
        """Same as replay() but computed by a child interpreter, so the live
        audio callback / render thread never compete with it for the GIL.
        Progress arrives as JSON lines on the child's stdout; cancel() -> True
        terminates the child."""
        return self._run_child('replay', rid, progress, cancel)

    def _run_child(self, verb, rid, progress=None, cancel=None):
        cmd = [sys.executable, '-X', 'utf8', '-m', 'casynth_lab.catalog', verb,
               self.root, rid]
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        try:
            proc = subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, encoding='utf-8')
        except OSError as e:
            return ReplayResult('unavailable', f"cannot start replay process: {e}")
        result = None
        lines = []

        def reader():
            for line in proc.stdout:
                lines.append(line)
        import threading
        th = threading.Thread(target=reader, daemon=True)
        th.start()
        seen = 0
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
                if 'progress' in msg and progress is not None:
                    progress(float(msg['progress']))
                elif 'status' in msg:
                    result = ReplayResult(msg['status'], msg.get('reason', ''),
                                          msg.get('outputs') or {}, msg.get('replay_dir'))
            if not th.is_alive() and proc.poll() is not None:
                break
            if cancel is not None and cancel():
                proc.terminate()
                proc.wait(timeout=5)
                return ReplayResult('cancelled')
        if result is None:
            err = proc.stderr.read().strip().splitlines()
            return ReplayResult('unavailable',
                                f"replay process failed: {err[-1] if err else proc.returncode}")
        return result


def _cli(argv):
    """python -m casynth_lab.catalog replay ROOT ID      (JSON lines on stdout)
       python -m casynth_lab.catalog notes ROOT [--all]  (listening notes, Markdown)"""
    if len(argv) >= 2 and argv[0] == 'notes':
        cat = Catalog(argv[1])
        rows = cat.notes_report(with_description='--all' in argv[2:])
        if not rows:
            print(f"(no notes in {cat.root})")
            return 0
        for rec, notes in rows:
            la, lb = rec.engine_labels()
            print(f"## {rec.title}\n")
            print(f"- record `{rec.id}`, {rec.created.replace('T', ' ')}, {rec.seconds:.1f} s, "
                  f"A: {la} / B: {lb}, {rec.version_label()}")
            if rec.note.strip():
                print(f"- description: {rec.note.strip()}")
            if notes.strip():
                print()
                print(notes.rstrip())
            print()
        return 0
    if len(argv) != 3 or argv[0] not in ('replay', 'end_state'):
        print("usage: python -m casynth_lab.catalog replay|end_state ROOT ID  |  notes ROOT [--all]")
        return 2
    cat = Catalog(argv[1])

    def prog(f):
        print(json.dumps({'progress': f}), flush=True)
    if argv[0] == 'end_state':
        try:
            st = end_state_by_replay(cat.load(argv[2]).meta, progress=prog)
            print(json.dumps(dict(status='match', outputs=st), ensure_ascii=False), flush=True)
        except CatalogError as e:
            print(json.dumps(dict(status='unavailable', reason=str(e)), ensure_ascii=False),
                  flush=True)
        return 0
    res = cat.replay(argv[2], progress=prog, yield_cpu=False)
    print(json.dumps(dict(status=res.status, reason=res.reason, outputs=res.outputs,
                          replay_dir=res.replay_dir), ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(_cli(sys.argv[1:]))
