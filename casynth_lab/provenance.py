"""Execution provenance (S6): WHICH code produced a record, and does it match
a Git revision.

A provenance document (schema 1) describes:
  - the SOUND SET actually executed (`sound_set` = 3 since 2026-09-21, when the
    engines moved out into casynth_engines/): the files that can change the PCM
    of an experiment given its embedded conditions -- the shared DSP modules
    (casynth_core / casynth_engine / casynth_config), the casynth_engines
    package (engine_api, registry, legacy_engine, the engine modules) and the
    casynth_lab modules that turn a scene + journal + snapshot into blocks
    (runner, scene, snapshot).  The shims casynth_lab keeps on the moved names
    are in the set too: a shim decides which module a bench name resolves to.  NOT in the set: the bench UI
    (demo_bench.py), device / thread plumbing (audio_out), catalog storage
    and replay orchestration (catalog, recorder), checking and versioning
    (verify, versions, provenance), requirements.txt (the environment block
    carries the exact package versions) and demos/*.json (a record embeds its
    scene).  An edit of those must not re-pin experiments -- the 2026-09-14
    incident: a UI change made every demo record open "in version <old>".
    Set 1 (S6 as delivered) fingerprinted all of that; records made with it
    are compared by CONTENT AT THEIR COMMIT (see catalog.source_plan), never
    by their stored digest alone.
    Per file a content fingerprint (sha256 of the LF-normalised bytes, so a
    CRLF checkout on Windows equals the LF blob Git stores) and one overall
    digest;
  - the Git revision (repo id = root commit, full HEAD commit) and whether the
    sound set CONTENT equals that commit's blobs (`match` = 'clean' /
    'dirty' / 'unknown'), with the dirty files listed;
  - the environment, separately from the code: Python, exact package versions,
    OS / architecture, audio context.

The document is captured ONCE per process, as early as possible (first call of
current()), i.e. for the code that was imported -- not the HEAD at Save time.
Subprocesses capture their own.  A `loaded_from` check records whether the
imported casynth modules really come from that root (a child bench started in
a cached checkout must not silently import the main checkout).

Status derived for a record: 'pinned' when match == 'clean' and the commit is
known (and the pin ref was created), else 'local' with a reason.  Pinned means
ESTABLISHED ORIGIN; whether the version can run here and whether the PCM
reproduces are separate checks.

Diagnosis (why does Continue want another version?):
    python -m casynth_lab.provenance                      # this checkout: set, digest, match
    python -m casynth_lab.provenance why CATALOG [ID...]  # per pinned record: same sound
                                                          # code or which files differ
"""
import hashlib
import os
import platform
import shutil
import subprocess
import sys

PROVENANCE_SCHEMA = 1
SOUND_SET_VERSION = 3                       # 1 = S6 runtime set (bench + lab + demos + reqs),
                                            # 2 = the sound-only set, 3 = + casynth_engines/
PIN_REF_PREFIX = 'refs/casynth/pins/'      # one permanent ref per pinned commit
# the SOUND set (fingerprinted): explicit, no import analysis -- see the module doc
SOUND_FILES = ('casynth_core.py', 'casynth_engine.py', 'casynth_config.py')
SOUND_DIRS = (('casynth_lab', '.py'), ('casynth_engines', '.py'))
SOUND_EXCLUDE = frozenset(('casynth_lab/__init__.py', 'casynth_engines/__init__.py',
                           'casynth_lab/audio_out.py',
                           'casynth_lab/catalog.py', 'casynth_lab/recorder.py',
                           'casynth_lab/verify.py', 'casynth_lab/versions.py',
                           'casynth_lab/provenance.py', 'casynth_lab/textedit.py',
                           'casynth_lab/notes_window.py'))
# the files a working checkout of the bench needs (isolated copies in tests,
# the S7 demo repo): the sound set plus the bench, its resources and deps
RUNTIME_FILES = ('demo_bench.py', 'patterns.py', 'casynth_core.py', 'casynth_engine.py',
                 'casynth_config.py', 'requirements.txt')
RUNTIME_DIRS = (('casynth_lab', '.py'), ('casynth_engines', '.py'), ('demos', '.json'))
RUNTIME_MODULES = ('casynth_core', 'casynth_engine', 'casynth_config', 'casynth_lab',
                   'casynth_engines')
PACKAGES = ('numpy', 'scipy', 'pygame', 'sounddevice')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ProvenanceError(RuntimeError):
    pass


def _norm(data):
    """LF-normalised bytes (Git autocrlf on Windows checks files out as CRLF)."""
    return data.replace(b'\r\n', b'\n')


def _sha(data):
    return hashlib.sha256(_norm(data)).hexdigest()


def _paths(root, files, dirs, exclude=()):
    out = []
    for f in files:
        if os.path.isfile(os.path.join(root, f)):
            out.append(f)
    for d, ext in dirs:
        dd = os.path.join(root, d)
        if os.path.isdir(dd):
            for name in os.listdir(dd):
                if name.endswith(ext) and os.path.isfile(os.path.join(dd, name)):
                    out.append(f"{d}/{name}")
    return sorted(p for p in out if p not in exclude)


def runtime_paths(root):
    """Sorted repo-relative paths (forward slashes) of everything a working
    checkout of the bench needs (sound set + bench + resources + deps).  NOT
    the fingerprint -- see sound_paths()."""
    return _paths(root, RUNTIME_FILES, RUNTIME_DIRS)


def sound_paths(root):
    """Sorted repo-relative paths of the SOUND set present at `root` (the
    fingerprinted files)."""
    return _paths(root, SOUND_FILES, SOUND_DIRS, SOUND_EXCLUDE)


def is_sound_path(rel):
    """Does a repo-relative path (forward slashes) belong to the sound set?"""
    if rel in SOUND_EXCLUDE:
        return False
    if rel in SOUND_FILES:
        return True
    for d, ext in SOUND_DIRS:
        if rel.startswith(d + '/') and rel.endswith(ext) and rel.count('/') == 1:
            return True
    return False


def manifest_of_tree(root):
    """{relpath: sha256(LF-normalised content)} of the sound set in the
    working tree."""
    m = {}
    for rel in sound_paths(root):
        with open(os.path.join(root, rel), 'rb') as f:
            m[rel] = _sha(f.read())
    return m


def digest(manifest):
    h = hashlib.sha256()
    for k in sorted(manifest):
        h.update(f"{k} {manifest[k]}\n".encode())
    return h.hexdigest()


# -- git ------------------------------------------------------------------------
def git_exe():
    g = shutil.which('git')
    if g:
        return g
    for cand in (r"C:\Program Files\Git\cmd\git.exe", r"C:\Program Files\Git\bin\git.exe",
                 r"C:\Program Files\Git\mingw64\bin\git.exe"):
        if os.path.isfile(cand):
            return cand
    return None


def git(root, *args, check=True, input=None):
    """Run git in `root`; returns stdout (str).  ProvenanceError on failure."""
    exe = git_exe()
    if exe is None:
        raise ProvenanceError("git executable not found")
    try:
        r = subprocess.run([exe, *args], cwd=root, capture_output=True, input=input,
                           timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise ProvenanceError(f"git {' '.join(args)}: {e}")
    if check and r.returncode != 0:
        raise ProvenanceError(f"git {' '.join(args)}: {r.stderr.decode('utf-8', 'replace').strip()}")
    return r.stdout.decode('utf-8', 'replace')


def git_root(path):
    """Top-level directory of the repository containing `path` (None if none)."""
    try:
        out = git(path, 'rev-parse', '--show-toplevel').strip()
    except ProvenanceError:
        return None
    return os.path.normcase(os.path.abspath(out)) if out else None


def repo_identity(root):
    """(repo_id = root commit, head commit) -- None where unknown."""
    try:
        head = git(root, 'rev-parse', 'HEAD').strip() or None
    except ProvenanceError:
        return None, None
    try:
        roots = git(root, 'rev-list', '--max-parents=0', 'HEAD').split()
        rid = sorted(roots)[0] if roots else None
    except ProvenanceError:
        rid = None
    return rid, head


def manifest_at_commit(root, commit):
    """{relpath: sha256(LF-normalised blob)} of the SOUND set AT a commit
    (only paths matching the sound-set patterns, by the CURRENT definition)."""
    listing = git(root, 'ls-tree', '-r', '--name-only', commit)
    paths = [p for p in listing.splitlines() if is_sound_path(p.strip())]
    if not paths:
        return {}
    spec = ''.join(f"{commit}:{p}\n" for p in paths).encode()
    exe = git_exe()
    r = subprocess.run([exe, 'cat-file', '--batch'], cwd=root, input=spec,
                       capture_output=True, timeout=120)
    if r.returncode != 0:
        raise ProvenanceError(f"git cat-file: {r.stderr.decode('utf-8', 'replace').strip()}")
    data = r.stdout
    out, pos = {}, 0
    for p in paths:
        nl = data.index(b'\n', pos)
        header = data[pos:nl].decode()
        pos = nl + 1
        parts = header.split()
        if len(parts) < 3:                       # "<sha> missing"
            raise ProvenanceError(f"git cat-file: {header}")
        size = int(parts[2])
        blob = data[pos:pos + size]
        pos += size + 1                          # trailing newline
        out[p] = _sha(blob)
    return out


def digest_at_commit(root, commit):
    return digest(manifest_at_commit(root, commit))


def compare_to_commit(root, commit, manifest):
    """'clean' when the working runtime set equals the commit's, else 'dirty'
    with the differing / new / missing paths."""
    ref = manifest_at_commit(root, commit)
    dirty = manifest_difference(ref, manifest)
    return ('clean' if not dirty else 'dirty'), dirty


def manifest_difference(a, b):
    """Sorted paths whose fingerprint differs between two manifests
    (missing on one side counts)."""
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def pin_ref(commit):
    return PIN_REF_PREFIX + commit


def hold_commit(root, commit):
    """Create the permanent ref that keeps `commit` reachable (idempotent).
    ProvenanceError on failure -- callers must not report a pin then."""
    git(root, 'update-ref', pin_ref(commit), commit)
    got = git(root, 'rev-parse', '--verify', '--quiet', pin_ref(commit)).strip()
    if got != commit:
        raise ProvenanceError(f"pin ref for {commit[:7]} not verified")


def is_held(root, commit):
    try:
        return git(root, 'rev-parse', '--verify', '--quiet', pin_ref(commit)).strip() == commit
    except ProvenanceError:
        return False


def commit_exists(root, commit):
    exe = git_exe()
    if exe is None or not commit:
        return False
    try:
        r = subprocess.run([exe, 'cat-file', '-e', f"{commit}^{{commit}}"], cwd=root,
                           capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


# -- environment ----------------------------------------------------------------
def environment():
    pk = {}
    for mod in PACKAGES:
        try:
            pk[mod] = __import__(mod).__version__
        except Exception:                      # noqa: BLE001
            pk[mod] = None
    return dict(python=platform.python_version(), executable=sys.executable,
                packages=pk, os=platform.system(), os_version=platform.version(),
                machine=platform.machine())


def environment_compatible(rec_env, cur_env=None):
    """(ok, reason): the current environment can run what `rec_env` describes.
    S6 supports only the available verified interpreter: same Python
    major.minor and exactly the same package versions."""
    cur_env = cur_env or environment()
    if not rec_env:
        return False, "environment not recorded"
    a, b = str(rec_env.get('python', '')).split('.')[:2], cur_env['python'].split('.')[:2]
    if a != b:
        return False, f"Python {rec_env.get('python')} != {cur_env['python']}"
    for mod, ver in (rec_env.get('packages') or {}).items():
        if ver and cur_env['packages'].get(mod) != ver:
            return False, f"{mod} {ver} != {cur_env['packages'].get(mod)}"
    if rec_env.get('machine') and rec_env['machine'] != cur_env['machine']:
        return False, f"architecture {rec_env['machine']} != {cur_env['machine']}"
    return True, ""


# -- capture --------------------------------------------------------------------
def loaded_from(root):
    """Which of the runtime modules already imported live under `root`
    ({module: True/False}); modules not imported yet are omitted."""
    root_n = os.path.normcase(os.path.abspath(root)) + os.sep
    out = {}
    for name in RUNTIME_MODULES:
        m = sys.modules.get(name)
        f = getattr(m, '__file__', None) if m is not None else None
        if f:
            out[name] = os.path.normcase(os.path.abspath(f)).startswith(root_n)
    return out


def capture(root=PROJECT_ROOT, audio=None, check_loaded=True):
    """Provenance document for the runtime set at `root` (see module doc).
    check_loaded=False skips the imported-modules check (tests that describe
    an isolated repository from the main process)."""
    root = os.path.abspath(root)
    manifest = manifest_of_tree(root)
    doc = dict(schema=PROVENANCE_SCHEMA, sound_set=SOUND_SET_VERSION, root=root,
               manifest=manifest,
               digest=digest(manifest), repo_id=None, commit=None, match='unknown',
               dirty=[], reason='', environment=environment(), audio=dict(audio or {}),
               loaded_from=(loaded_from(root) if check_loaded else {}))
    if not manifest:
        doc['reason'] = "sound set not found"
        return doc
    if git_exe() is None:
        doc['reason'] = "git executable not found"
        return doc
    top = git_root(root)
    if top is None or top != os.path.normcase(root):
        doc['reason'] = "not the top level of a git repository"
        return doc
    rid, head = repo_identity(root)
    doc['repo_id'], doc['commit'] = rid, head
    if head is None:
        doc['reason'] = "no commit (empty repository)"
        return doc
    try:
        match, dirty = compare_to_commit(root, head, manifest)
    except ProvenanceError as e:
        doc['reason'] = str(e)
        return doc
    doc['match'], doc['dirty'] = match, dirty
    if dirty:
        doc['reason'] = "sound files differ from HEAD: " + ", ".join(dirty[:6]) + \
            (" ..." if len(dirty) > 6 else "")
    if any(v is False for v in doc['loaded_from'].values()):
        doc['match'] = 'unknown'
        doc['reason'] = "imported modules do not come from this checkout"
    return doc


_CURRENT = {}


def current(audio=None):
    """The process-wide provenance, captured on first use (the code that was
    imported)."""
    if 'doc' not in _CURRENT:
        _CURRENT['doc'] = capture(PROJECT_ROOT, audio=audio)
    return _CURRENT['doc']


def reset_current():
    _CURRENT.clear()


def status_of(doc):
    """('pinned' | 'local', reason) as derived from a provenance document
    alone (pinned additionally needs the held ref -- see catalog)."""
    if not doc:
        return 'local', "code version not recorded"
    if doc.get('match') == 'clean' and doc.get('commit'):
        return 'pinned', ''
    return 'local', doc.get('reason') or f"sound set does not match commit ({doc.get('match')})"


def short(commit):
    return (commit or '')[:7]


# -- diagnosis CLI ---------------------------------------------------------------
def _cli(argv):
    import json
    args = list(argv)
    if args and args[0] == 'why':
        from .catalog import Catalog
        if len(args) < 2:
            print("usage: python -m casynth_lab.provenance why CATALOG_ROOT [RECORD_ID ...]")
            return 2
        cat = Catalog(os.path.abspath(args[1]))
        ids = args[2:] or cat.ids()
        cur = current()
        print(f"this checkout: {short(cur.get('commit')) or '?'} {cur.get('match')} "
              f"sound digest {cur.get('digest', '')[:12]}  {cur.get('reason', '')}".rstrip())
        for rid in ids:
            try:
                rec = cat.load(rid)
            except Exception as e:                       # noqa: BLE001
                print(f"{rid}: cannot load: {e}")
                continue
            plan = cat.source_plan(rec, current=cur)
            line = (f"{rid}  {rec.title[:40]!r}  {rec.version_label()}  -> "
                    f"{plan['mode'] or 'unavailable'}")
            if plan.get('differs'):
                line += "  sound files differ: " + ", ".join(plan['differs'])
            elif plan.get('reason'):
                line += "  " + plan['reason']
            elif plan['mode'] == 'in-process' and rec.status == 'pinned':
                line += "  same sound code"
            stored = (rec.provenance or {}).get('sound_set', 1)
            if stored != SOUND_SET_VERSION:
                line += f"  (digest stored with set {stored}: compared by content at its commit)"
            print(line)
        return 0
    doc = current()
    print(json.dumps({k: doc.get(k) for k in ('commit', 'match', 'reason', 'digest',
                                               'sound_set', 'root')}, indent=1))
    print("sound set:")
    for k, v in sorted(doc.get('manifest', {}).items()):
        flag = " (differs from HEAD)" if k in doc.get('dirty', []) else ""
        print(f"  {k}  {v[:12]}{flag}")
    for k in doc.get('dirty', []):
        if k not in doc.get('manifest', {}):
            print(f"  {k}  (missing here, present at HEAD)")
    return 0


if __name__ == '__main__':
    sys.exit(_cli(sys.argv[1:]))
