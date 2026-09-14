"""Running a record's ORIGINAL code version (S6).

Pinned records name a commit that is held by a permanent ref
(refs/casynth/pins/<commit>).  To execute that version we keep a cache of
detached worktrees, one per commit, under <catalog>/../worktrees/<commit>/
(user data, gitignored with lab_catalog/).  The main checkout, its branch,
index and uncommitted changes are never touched: `git worktree add --detach`
only writes the new directory (+ .git/worktrees bookkeeping).

A child bench is started FROM that worktree with the minimal launch contract
    python demo_bench.py --catalog <abs catalog root> --record <id> --action <a>
so it imports the worktree's modules (its own directory is sys.path[0]; the
parent's PYTHONPATH is stripped), sees the SAME catalog (absolute root) and
reports its own provenance as the first JSON line on stdout.  A cache entry
is verified before use (HEAD == commit and the sound set's digest equals
the commit's); a broken one is removed and recreated from the held commit.
"""
import json
import os
import shutil
import subprocess
import sys
import threading

from . import provenance as prov

WORKTREES_DIRNAME = 'worktrees'


class VersionError(RuntimeError):
    pass


def worktrees_root(catalog_root):
    return os.path.join(os.path.dirname(os.path.abspath(catalog_root)), WORKTREES_DIRNAME)


def worktree_path(catalog_root, commit):
    return os.path.join(worktrees_root(catalog_root), commit)


def _verify(repo_root, path, commit):
    """True when `path` is a checkout of `commit` with an intact sound set."""
    if not os.path.isdir(path):
        return False
    try:
        head = prov.git(path, 'rev-parse', 'HEAD').strip()
        if head != commit:
            return False
        return prov.digest(prov.manifest_of_tree(path)) == prov.digest_at_commit(repo_root, commit)
    except prov.ProvenanceError:
        return False


def _rmtree(path):
    import stat

    def onexc(fn, p, exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            fn(p)
        except OSError:
            pass
    if os.path.exists(path):
        shutil.rmtree(path, onexc=onexc)


def remove_worktree(repo_root, path):
    try:
        prov.git(repo_root, 'worktree', 'remove', '--force', path, check=False)
    except prov.ProvenanceError:
        pass
    _rmtree(path)
    try:
        prov.git(repo_root, 'worktree', 'prune', check=False)
    except prov.ProvenanceError:
        pass


def ensure_worktree(repo_root, catalog_root, commit):
    """Path of a verified detached checkout of `commit` (created or repaired
    from the held commit).  VersionError when the commit is not available."""
    if not commit:
        raise VersionError("record has no commit")
    if not prov.commit_exists(repo_root, commit):
        raise VersionError(f"commit {prov.short(commit)} is not in the repository")
    path = worktree_path(catalog_root, commit)
    if _verify(repo_root, path, commit):
        return path
    remove_worktree(repo_root, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        prov.git(repo_root, 'worktree', 'add', '--detach', path, commit)
    except prov.ProvenanceError as e:
        raise VersionError(f"cannot check out {prov.short(commit)}: {e}")
    if not _verify(repo_root, path, commit):
        remove_worktree(repo_root, path)
        raise VersionError(f"checkout of {prov.short(commit)} failed verification")
    return path


def child_command(worktree, catalog_root, rid, action, extra=()):
    return [sys.executable, '-X', 'utf8', os.path.join(worktree, 'demo_bench.py'),
            '--catalog', os.path.abspath(catalog_root), '--record', rid,
            '--action', action, *extra]


def child_env():
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)        # never reach the parent's checkout
    env['PYTHONUTF8'] = '1'
    return env


class ChildBench:
    """A bench / worker process of another version.  JSON lines on its stdout
    are collected: `provenance` (first) and `status` (last)."""

    def __init__(self, worktree, catalog_root, rid, action, extra=(), on_line=None):
        self.cmd = child_command(worktree, catalog_root, rid, action, extra)
        self.worktree = worktree
        self.proc = None
        self.provenance = None
        self.result = None
        self.lines = []
        self.error = None
        self._on_line = on_line
        self._thread = None

    def start(self):
        try:
            self.proc = subprocess.Popen(self.cmd, cwd=self.worktree, env=child_env(),
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                         text=True, encoding='utf-8')
        except OSError as e:
            raise VersionError(f"cannot start the version's bench: {e}")
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return self

    def _reader(self):
        for line in self.proc.stdout:
            line = line.strip()
            self.lines.append(line)
            if line.startswith('{'):
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'provenance' in msg:
                    self.provenance = msg['provenance']
                elif 'status' in msg:
                    self.result = msg
                if self._on_line is not None:
                    self._on_line(msg)
        self.proc.wait()
        if self.proc.returncode != 0 and self.result is None:
            err = self.proc.stderr.read().strip().splitlines()
            self.error = err[-1] if err else f"exit code {self.proc.returncode}"

    @property
    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def wait(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)
        return not self.alive

    def terminate(self):
        if self.alive:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
