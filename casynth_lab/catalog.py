"""Local experiment catalog (S4): save a Cut as a record, list / load records,
read their WAVs, and REPLAY a record from the start with a byte-exact check.

Layout (default root lab_catalog/local/, user data, gitignored):
    <root>/<id>/record.json      metadata + embedded scene + applied journal
    <root>/<id>/A.wav B.wav monitor.wav
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
from .runner import BLOCK, CHANNELS, OUTPUTS, DemoRunner
from .scene import SceneError, scene_from_doc

RECORD_FORMAT = 1
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


def read_wav(path):
    """int16 (n, channels) array of a record WAV (CatalogError if unreadable)."""
    try:
        with wave.open(path, 'rb') as w:
            if w.getsampwidth() != 2:
                raise CatalogError(f"{path}: not 16-bit")
            n, ch = w.getnframes(), w.getnchannels()
            data = w.readframes(n)
    except (wave.Error, EOFError, OSError) as e:
        raise CatalogError(f"{path}: unreadable WAV ({e})")
    arr = np.frombuffer(data, np.int16)
    if len(arr) != n * ch:
        raise CatalogError(f"{path}: truncated WAV")
    return arr.reshape(-1, ch)


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
    d.setdefault('listen', '')
    return d


def _replace_dir(src, dst, attempts=20):
    """os.replace for a directory; Windows may refuse transiently (indexer /
    antivirus holding a handle) -> retry briefly before giving up."""
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.05 * (i + 1))


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
        return self.meta.get('note', '')

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
    d.setdefault('listen', '')
    try:
        scene = scene_from_doc(d)
    except SceneError as e:
        raise CatalogError(f"{rec.id}: cannot open in bench: {e}")
    return scene, float(st.get('vol', VOL_DEFAULT))


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
    def __init__(self, root=DEFAULT_ROOT):
        self.root = root
        self.tmp_root = os.path.join(root, '.tmp')

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
        if meta['format'] != RECORD_FORMAT:
            raise CatalogError(f"{rid}: record format {meta['format']} not supported")
        return Record(self.root, rid, meta)

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
            raise CatalogError("nothing to save yet: press Start first")
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
            meta = {
                'format': RECORD_FORMAT,
                'id': rid,
                'created': now.isoformat(timespec='seconds'),
                'title': title,
                'note': note or '',
                'status': 'local',
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

    # -- replay ---------------------------------------------------------------
    def replay(self, rid, progress=None, cancel=None, yield_cpu=True):
        """Recompute a record from its embedded conditions + journal and compare
        with the stored WAVs byte-exact.  progress(frac) is called during the
        render; cancel() -> True aborts.  Writes <id>/replay/*.wav; the
        originals are never modified."""
        try:
            rec = self.load(rid)
        except CatalogError as e:
            return ReplayResult('unavailable', str(e))
        meta = rec.meta
        try:
            scene = scene_from_doc(meta['scene'])
        except (SceneError, KeyError, TypeError, ValueError) as e:
            return ReplayResult('unavailable', f"scene cannot be rebuilt: {e}")
        rs = meta.get('runner', {})
        if (rs.get('sr', SR) != SR or rs.get('block', BLOCK) != BLOCK
                or rs.get('channels', CHANNELS) != CHANNELS):
            return ReplayResult('unavailable',
                                f"render settings differ (SR/BLOCK/channels): {rs}")
        try:
            runner = DemoRunner(scene, vol=rs.get('vol_initial', VOL_DEFAULT))
        except (ValueError, KeyError) as e:
            return ReplayResult('unavailable', f"engine unavailable: {e}")
        origin = meta.get('origin_sample', 0)
        start, end = meta['audio_start_sample'] - origin, meta['end_sample'] - origin
        cmds = [dict(j, out_sample=j['out_sample'] - origin) for j in meta['journal']
                if j['kind'] not in REPLAY_COMMANDS_SKIP and j['out_sample'] >= origin]
        cmds.sort(key=lambda j: (j['out_sample'], j['seq']))
        got = {o: [] for o in OUTPUTS}
        i = 0
        total = max(end, 1)
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
                    return ReplayResult('cancelled')
                if progress is not None and (before // BLOCK) % 50 == 0:
                    progress(min(before / total, 1.0))
                if yield_cpu and (before // BLOCK) % 4 == 0:
                    time.sleep(0.002)     # leave the GIL to a live render thread
        except Exception as e:                 # noqa: BLE001
            return ReplayResult('unavailable', f"replay failed: {e}")
        if i < len(cmds):
            return ReplayResult('unavailable', f"{len(cmds) - i} journal commands "
                                               f"beyond the recorded end")
        n = end - start
        outputs, replay_dir = {}, os.path.join(rec.dir, 'replay')
        try:
            os.makedirs(replay_dir, exist_ok=True)
            for o in OUTPUTS:
                pcm = (np.concatenate(got[o], axis=0)[:n] if got[o]
                       else np.zeros((0, CHANNELS), np.int16))
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
        if progress is not None:
            progress(1.0)
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
    """python -m casynth_lab.catalog replay ROOT ID   (JSON lines on stdout)"""
    if len(argv) != 3 or argv[0] not in ('replay', 'end_state'):
        print("usage: python -m casynth_lab.catalog replay|end_state ROOT ID")
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
