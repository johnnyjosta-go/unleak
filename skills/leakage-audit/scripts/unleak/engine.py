"""Static data-leakage analysis for Python ML code.

The analyzer walks each scope (module, function) in program order and keeps an
abstract value (`Val`) for every variable:

* ``roots``  - which loaded datasets the rows come from
* ``tags``   - which side of a split the rows are on (train / test / val)
* ``fits``   - which fitted preprocessing steps shaped the values
* ``frame``  - which columns a DataFrame still contains

Leakage is then a question about these values, e.g. "did a scaler fitted on
rows *without* a train tag shape data that carries a test tag?".
"""

from __future__ import annotations

import ast
import builtins
import itertools
import os
from dataclasses import dataclass, field, replace

from . import knowledge as K

EMPTY: frozenset = frozenset()
HELDOUT = ("test", "val")
# Picking a winner among scores. max/min/sorted only count with key= or as a
# running best inside the loop: max(scores) alone is often just a plot limit.
STRONG_SELECTION = {"argmax", "argmin", "nanargmax", "nanargmin", "idxmax", "idxmin",
                    "nlargest", "nsmallest", "sort_values", "argsort", "most_common", "index"}
WEAK_SELECTION = {"max", "min", "sorted"}
SCALAR_BUILTINS = {
    "len", "int", "float", "round", "str", "bool", "isinstance", "type", "id",
    "hash", "print", "range", "repr", "format", "callable", "hasattr",
    "issubclass", "ord", "chr", "input", "open_",
}
SCALAR_ATTRS = {"shape", "size", "ndim", "dtypes", "dtype", "empty", "name", "nbytes"}
VALUE_ATTRS = {"values", "T", "array", "str", "dt", "cat", "index"}
WRITE_METHODS = {"to_csv", "to_parquet", "to_pickle", "to_feather", "to_json", "to_excel", "to_hdf",
                 "to_orc", "to_stata", "write_csv", "write_parquet"}
WRITE_FUNCS = {"save": ("numpy", "np", "torch"), "savez": ("numpy", "np"), "savez_compressed": ("numpy", "np"),
               "savetxt": ("numpy", "np"), "dump": ("joblib", "pickle")}
# sklearn Bunch fields: dataset.data / dataset.target are arrays, not columns
BUNCH_ATTRS = {"data", "target", "images", "feature_names", "target_names", "frame",
               "DESCR", "filename", "data_module", "cov_train", "cov_test"}


# --- abstract values -------------------------------------------------------


@dataclass(frozen=True)
class Tag:
    """Rows on one side of one split."""

    origin: object
    role: str  # train | test | val
    parent: frozenset = EMPTY  # tags of the data that was split


@dataclass(frozen=True)
class ColumnSet:
    """All columns of a frame except some: df.columns.drop('y')."""

    excluded: frozenset = EMPTY


@dataclass(frozen=True)
class Frame:
    family: frozenset  # roots of the frame the columns belong to
    select: frozenset | None = None  # None: all columns except `dropped`
    dropped: frozenset = EMPTY
    unknown: bool = False
    site: ast.AST | None = None  # statement that chose these columns
    index: int = 0  # statement counter when the columns were chosen
    added: frozenset = EMPTY  # ids of ColAssigns written into this frame later


@dataclass(frozen=True)
class Est:
    cls: str
    kind: str
    fitted_on: Val | None = None
    fit_node: ast.AST | None = None
    early_stop: bool = False
    shuffle: bool = False
    grouped: bool = False
    temporal: bool = False
    init_data: Val | None = None
    cv: Val | None = None
    selects_columns: bool = False  # a ColumnTransformer picks the features itself


@dataclass(frozen=True)
class Val:
    roots: frozenset = EMPTY
    tags: frozenset = EMPTY
    fits: frozenset = EMPTY
    traits: frozenset = EMPTY
    est: Est | None = None
    mod: str | None = None
    frame: Frame | None = None
    col: tuple | None = None  # (family, column name)
    seq: tuple | None = None
    elem: Val | None = None
    const: object = None
    target: bool = False

    def roles(self) -> set:
        return {t.role for t in self.tags}

    def heldout(self) -> set:
        return {t for t in self.tags if t.role in HELDOUT}

    def data_only(self) -> Val:
        return Val(self.roots, self.tags, self.fits, self.traits, frame=self.frame,
                   col=self.col, target=self.target)


NOTHING = Val()
# Traits that describe a value itself, not what is derived from it.
TRANSIENT = frozenset({"prediction", "window_obj", "stat_value", "metric_value"})


def union(*vals: Val | None) -> Val:
    vals = [v for v in vals if v is not None and v is not NOTHING]
    if not vals:
        return NOTHING
    if len(vals) == 1:
        v = vals[0]
        return Val(v.roots, v.tags, v.fits, v.traits - TRANSIENT, frame=v.frame, col=v.col,
                   target=v.target)
    data = [v for v in vals if v.roots]
    return Val(
        roots=frozenset().union(*(v.roots for v in vals)),
        tags=frozenset().union(*(v.tags for v in vals)),
        fits=frozenset().union(*(v.fits for v in vals)),
        traits=frozenset().union(*(v.traits for v in vals)) - TRANSIENT,
        frame=data[0].frame if len(data) == 1 else None,
        col=data[0].col if len(data) == 1 else None,
        target=any(v.target for v in data),
    )


# --- events ----------------------------------------------------------------


@dataclass
class FitEvent:
    id: int
    kind: str
    cls: str
    method: str
    node: ast.AST
    data: Val
    path: str | None = None  # set when the fit happened in another file


@dataclass
class Written:
    """Data saved to disk: df.to_csv('clean.csv'), np.save('X.npy', X)..."""

    path: str
    events: list  # FitEvents that shaped the data
    traits: frozenset
    source: str | None = None  # file that wrote it


@dataclass
class Obs:
    """Held-out rows (or rows about to be cross-validated) seen somewhere."""

    val: Val
    tags: frozenset
    node: ast.AST
    what: str  # split | cv | use


@dataclass
class ColAssign:
    id: int
    family: frozenset
    col: str
    node: ast.AST
    index: int
    refs: list
    roles: frozenset


@dataclass
class Ref:
    col: str
    family: frozenset
    roles: frozenset
    shift: int  # >0 lag, <0 lead (future), 0 none
    centered: bool


@dataclass
class Loop:
    id: int
    node: ast.AST
    test_eval: bool = False
    evidence: bool = False


@dataclass
class RawFinding:
    rule: str
    node: ast.AST
    message: str
    related: list = field(default_factory=list)  # [(line, text, path or None)]
    path: str | None = None  # the file the node belongs to, if not the analysed one


# --- helpers ---------------------------------------------------------------


def dotted(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def const_int(node: ast.AST | None) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = const_int(node.operand)
        return -v if v is not None else None
    return None


def kwarg(call: ast.Call, name: str) -> ast.AST | None:
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def is_true(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def is_false(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is False


def src(node: ast.AST, limit: int = 60) -> str:
    try:
        text = ast.unparse(node)
    except Exception:  # pragma: no cover - unparse is total on parsed trees
        return "<expr>"
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def add_parents(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._ul_parent = node  # type: ignore[attr-defined]


def shift_sign(call: ast.Call) -> int:
    arg = call.args[0] if call.args else kwarg(call, "periods")
    if arg is None:
        return 1
    v = const_int(arg)
    if v is None:
        return 1  # unknown amount: assume an ordinary lag
    return (v > 0) - (v < 0)


def lambda_shift(call: ast.Call) -> int:
    """Sign of a shift applied inside .transform(lambda s: s.shift(1)...)."""
    for arg in list(call.args) + [k.value for k in call.keywords]:
        for sub in ast.walk(arg):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "shift":
                return shift_sign(sub)
    return 0


def shift_context(node: ast.AST, stop: ast.AST) -> tuple[int, bool]:
    """How a column reference is shifted on its way to the assigned value."""
    sign, centered = 0, False
    cur = node
    while cur is not stop:
        parent = getattr(cur, "_ul_parent", None)
        if parent is None:
            break
        if isinstance(parent, ast.Attribute) and parent.value is cur:
            call = getattr(parent, "_ul_parent", None)
            if isinstance(call, ast.Call) and call.func is parent:
                m = parent.attr
                if m == "shift" and sign == 0:
                    sign = shift_sign(call)
                elif m == "rolling" and is_true(kwarg(call, "center")):
                    centered = True
                elif m in ("transform", "apply", "agg", "aggregate", "pipe") and sign == 0:
                    sign = lambda_shift(call)
        cur = parent
    return sign, centered


# --- the analyzer ----------------------------------------------------------


class Analyzer:
    def __init__(self, tree: ast.Module, path: str | None = None, external: dict | None = None):
        self.tree = tree
        self.path = path
        # normalised path or basename -> Written records from other files
        self.external = external or {}
        self.writes: list[Written] = []
        self.ids = itertools.count(1)
        self.imports: dict[str, str] = {}
        self.function_names: set[str] = set()
        self.fit_events: dict[int, FitEvent] = {}
        self.obs: list[Obs] = []
        self.es_candidates: list[tuple[ast.AST, Val]] = []
        self.evals: list[Val] = []
        self.loops: dict[int, Loop] = {}
        self.findings: list[RawFinding] = []
        self._reported: set = set()
        self.pair_seen: set = set()

    # -- entry point --
    def run(self) -> list[RawFinding]:
        add_parents(self.tree)
        self._collect_names()
        module = Scope(self, self.tree.body)
        module.run()
        for fn in ast.walk(self.tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                Scope(self, fn.body, parent=module, args=fn.args).run()
        self._finalize()
        return self.findings

    def _collect_names(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    self.imports[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for a in node.names:
                    self.imports[a.asname or a.name] = f"{mod}.{a.name}" if mod else a.name
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.function_names.add(node.name)

    def report(self, rule: str, node: ast.AST, message: str, related: list | None = None,
               path: str | None = None) -> None:
        key = (rule, getattr(node, "lineno", 0), path)
        if key in self._reported:
            return
        self._reported.add(key)
        self.findings.append(RawFinding(rule, node, message, related or [], path))

    # -- cross-statement rules --
    def _finalize(self) -> None:
        rule_for = {
            K.TRANSFORMER: "UL101", K.UNKNOWN_TRANSFORMER: "UL101",
            K.SELECTOR: "UL102", K.OVERSAMPLER: "UL103",
            K.UNDERSAMPLER: "UL104", K.TARGET_ENCODER: "UL105", K.STAT: "UL106",
        }
        for ev in self.fit_events.values():
            if ev.kind not in rule_for or not ev.data.roots or ev.data.target:
                continue
            fit_tags = ev.data.tags
            has_train = any(t.role == "train" for t in fit_tags)
            has_held = any(t.role in HELDOUT for t in fit_tags)
            if has_held and not has_train:
                continue  # fitted on held-out data only: UL201, reported at the call
            for ob in self.obs:
                if ev.id not in ob.val.fits or ob.val.target:
                    continue
                mixed = any(t in fit_tags for t in ob.tags)
                if not mixed and not any(
                    fit_tags <= t.parent and (ev.data.roots & ob.val.roots) for t in ob.tags
                ):
                    continue
                rule = rule_for[ev.kind]
                if mixed and has_train and ev.kind in (K.TRANSFORMER, K.UNKNOWN_TRANSFORMER, K.STAT, K.UNDERSAMPLER):
                    rule = "UL107"
                message = self._fit_message(rule, ev)
                if ev.path:
                    message += f"; the result is saved to disk and split in {self.path or 'another file'}"
                self.report(rule, ev.node, message, [self._where(ob)], path=ev.path)
                break
        for node, evalv in self.es_candidates:
            test_tags = {t for t in evalv.tags if t.role == "test"}
            for ev in self.evals:
                if ev.tags & test_tags:
                    self.report(
                        "UL203", node,
                        "early stopping / best-iteration selection is driven by the test set, "
                        "and the same test set is used to report the final score",
                    )
                    break
        for loop in self.loops.values():
            if loop.test_eval and loop.evidence:
                self.report(
                    "UL204", loop.node,
                    "this loop scores candidates on the test set and keeps the best one, "
                    "so the test set is being used for model selection",
                )

    def _where(self, ob: Obs) -> tuple[int, str, str | None]:
        line = getattr(ob.node, "lineno", 0)
        text = {"split": "rows are split into train/test here",
                "cv": "data is cross-validated here"}.get(ob.what, "held-out rows are used here")
        return (line, text, self.path)

    def lookup_written(self, path: str) -> list[Written]:
        if self.path:
            full = os.path.normpath(os.path.join(os.path.dirname(self.path), path))
            if full in self.external:
                return self.external[full]
        hits = self.external.get(os.path.basename(path), [])
        return hits if len({w.path for w in hits}) == 1 else []

    @staticmethod
    def _fit_message(rule: str, ev: FitEvent) -> str:
        if ev.kind == K.STAT or ev.cls in K.SUPERVISED_STAT_FUNCS or ev.cls in ("corr", "corrwith"):
            call = f"`{src(ev.node, 50)}`"
        elif ev.method == "concat":  # hand-rolled resampling
            call = ev.cls[0].upper() + ev.cls[1:]
            if rule == "UL103":
                return f"{call}: duplicated minority rows end up on both sides of the split"
        else:
            call = f"{ev.cls}.{ev.method}()"
        return {
            "UL101": f"{call} is fitted on rows that are later used for evaluation",
            "UL102": f"{call} selects features using the labels of rows that are later used for evaluation",
            "UL103": f"{call} creates synthetic samples from rows that end up in the evaluation data",
            "UL104": f"{call} resamples before the split, so the evaluation data no longer has the real class balance",
            "UL105": f"{call} learns target statistics from rows that are later used for evaluation",
            "UL106": f"{call} is computed over all rows, including rows that later become test data, "
                     "and is used to transform the features before the split",
            "UL107": f"{call} is fitted on train and test data combined",
        }[rule]


class Scope:
    def __init__(self, an: Analyzer, body: list, parent: Scope | None = None,
                 args: ast.arguments | None = None):
        self.an = an
        self.body = body
        self.env: dict[str, Val] = {}
        self.index = 0
        self.stmt_node: ast.AST | None = None
        self.loop_stack: list[Loop] = []
        self.colassigns: list[ColAssign] = []
        self.collector: list | None = None
        self.targets: set[str] = set()
        self.entity_cols: list[tuple[frozenset, str]] = []
        self.ungrouped_reported = False
        if parent is not None:
            for k, v in parent.env.items():
                if v.const is not None:
                    self.env[k] = v
        if args is not None:
            for a in args.posonlyargs + args.args + args.kwonlyargs:
                if a.arg not in ("self", "cls"):
                    self.env[a.arg] = self.new_root(a.arg)

    # -- driver --
    def run(self) -> None:
        self._prepass()
        for st in self.body:
            try:
                self.stmt(st)
            except RecursionError:  # pragma: no cover - pathological nesting
                continue

    def _prepass(self) -> None:
        for node in self._walk_scope(self.body):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                t = node.targets[0]
                if isinstance(t, ast.Name) and K.TARGET_NAME_RE.match(t.id):
                    c = self._target_col(node.value)
                    if c:
                        self.targets.add(c)

    def _walk_scope(self, body: list):
        stack = list(body)
        while stack:
            node = stack.pop()
            yield node
            for child in ast.iter_child_nodes(node):
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                    stack.append(child)

    def _target_col(self, node: ast.AST) -> str | None:
        while isinstance(node, (ast.Attribute, ast.Call)):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "pop" and node.args:
                    return self._const_str(node.args[0])
                node = node.func.value if isinstance(node.func, ast.Attribute) else (node.args[0] if node.args else None)
            else:
                node = node.value
        if isinstance(node, ast.Subscript):
            return self._const_str(node.slice)
        return None

    def _const_str(self, node: ast.AST | None) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            v = self.env.get(node.id)
            if v is not None and isinstance(v.const, str):
                return v.const
        return None

    # -- values --
    def new_root(self, name: str, role: str | None = None) -> Val:
        rid = next(self.an.ids)
        role = role or K.split_role(name)
        tags = frozenset({Tag(("name", role), role)}) if role else EMPTY
        fam = frozenset({rid})
        return Val(roots=fam, tags=tags, frame=Frame(family=fam, index=self.index))

    def load(self, name: str) -> Val:
        if name in self.env:
            return self.env[name]
        root = name.split(".", 1)[0]
        if root in self.an.imports and root not in self.env:
            return Val(mod=self.an.imports[root] + name[len(root):])
        if name in self.an.function_names:
            return Val(mod="func." + name)
        if hasattr(builtins, name):
            return Val(mod="builtins." + name)
        v = self.new_root(name)
        self.env[name] = v
        return v

    # -- statements --
    def stmt(self, st: ast.stmt) -> None:
        self.index += 1
        self.stmt_node = st
        handler = getattr(self, "st_" + type(st).__name__, None)
        if handler is not None:
            handler(st)

    def st_Assign(self, st: ast.Assign) -> None:
        colassign = any(self._assigned_col(t) for t in st.targets)
        refs: list = []
        if colassign:
            self.collector = refs
        try:
            v = self.expr(st.value)
        finally:
            self.collector = None
        if (isinstance(st.value, ast.Call) and not v.roots and v.est is None and v.mod is None
                and v.const is None and v.seq is None and v.elem is None):
            # rng.normal(...), make_data(): an unknown call result may be a dataset
            fresh = self.new_root("value")
            v = replace(fresh, fits=v.fits, traits=v.traits, tags=v.tags)
        if isinstance(st.value, (ast.Name, ast.Attribute)) and v.frame is not None:
            v = replace(v, frame=replace(v.frame, site=st))  # X = df: report at the alias
        for t in st.targets:
            self.bind(t, v, st, refs)

    def st_AnnAssign(self, st: ast.AnnAssign) -> None:
        if st.value is not None:
            self.st_Assign(ast.Assign(targets=[st.target], value=st.value, lineno=st.lineno, col_offset=st.col_offset))

    def st_AugAssign(self, st: ast.AugAssign) -> None:
        refs: list = []
        if self._assigned_col(st.target):
            self.collector = refs
        try:
            v = self.expr(st.value)
            old = self.expr(st.target.value) if isinstance(st.target, ast.Subscript) else (
                self.load(dotted(st.target)) if dotted(st.target) else NOTHING)
        finally:
            self.collector = None
        self.bind(st.target, union(old, v), st, refs)

    def st_Expr(self, st: ast.Expr) -> None:
        self.expr(st.value)

    def st_Return(self, st: ast.Return) -> None:
        self.expr(st.value)

    def st_For(self, st: ast.For) -> None:
        it = self.expr(st.iter)
        loop = self.push_loop(st)
        self.bind(st.target, self.iter_elem(it), st)
        for s in st.body:
            self.stmt(s)
        self.pop_loop(loop)
        for s in st.orelse:
            self.stmt(s)

    st_AsyncFor = st_For

    def st_While(self, st: ast.While) -> None:
        loop = self.push_loop(st)
        self.cond(st.test)
        for s in st.body:
            self.stmt(s)
        self.pop_loop(loop)
        for s in st.orelse:
            self.stmt(s)

    def st_If(self, st: ast.If) -> None:
        self.cond(st.test)
        for s in st.body:
            self.stmt(s)
        for s in st.orelse:
            self.stmt(s)

    def st_With(self, st: ast.With) -> None:
        for item in st.items:
            v = self.expr(item.context_expr)
            if item.optional_vars is not None:
                self.bind(item.optional_vars, v, st)
        for s in st.body:
            self.stmt(s)

    st_AsyncWith = st_With

    def st_Try(self, st: ast.Try) -> None:
        for s in st.body:
            self.stmt(s)
        for h in st.handlers:
            for s in h.body:
                self.stmt(s)
        for s in st.orelse + st.finalbody:
            self.stmt(s)

    st_TryStar = st_Try

    def st_Match(self, st) -> None:
        self.expr(st.subject)
        for case in st.cases:
            for s in case.body:
                self.stmt(s)

    def st_Delete(self, st: ast.Delete) -> None:
        for t in st.targets:
            if isinstance(t, ast.Subscript):
                name = dotted(t.value)
                c = self._const_str(t.slice)
                if name and c and name in self.env:
                    v = self.env[name]
                    if v.frame is not None:
                        self.env[name] = replace(v, frame=self._drop_cols(v.frame, {c}, keep_site=True))

    def st_FunctionDef(self, st) -> None:
        self.env[st.name] = Val(mod="func." + st.name)

    st_AsyncFunctionDef = st_FunctionDef
    st_ClassDef = st_FunctionDef

    def cond(self, test: ast.AST) -> None:
        self.selection_evidence(self.expr(test))

    # -- loops / selection --
    def push_loop(self, node: ast.AST) -> Loop:
        loop = Loop(next(self.an.ids), node)
        self.an.loops[loop.id] = loop
        self.loop_stack.append(loop)
        return loop

    def pop_loop(self, loop: Loop) -> None:
        if self.loop_stack and self.loop_stack[-1] is loop:
            self.loop_stack.pop()

    def selection_evidence(self, v: Val) -> None:
        for t in v.traits:
            if isinstance(t, str) and t.startswith("tm:"):
                loop = self.an.loops.get(int(t[3:]))
                if loop is not None:
                    loop.evidence = True

    def iter_elem(self, it: Val) -> Val:
        if it.elem is not None:
            return it.elem
        if it.seq is not None:
            return union(*it.seq)
        return it.data_only() if it.roots else NOTHING

    # -- binding --
    def _assigned_col(self, t: ast.AST) -> bool:
        return isinstance(t, ast.Subscript) and self._col_key(t) is not None

    def _col_key(self, t: ast.Subscript) -> str | None:
        sl = t.slice
        if isinstance(t.value, ast.Attribute) and t.value.attr in ("loc", "at"):
            if isinstance(sl, ast.Tuple) and len(sl.elts) == 2:
                return self._const_str(sl.elts[1])
            return None
        return self._const_str(sl)

    def bind(self, t: ast.AST, v: Val, st: ast.AST, refs: list | None = None) -> None:
        if isinstance(t, (ast.Name, ast.Attribute)) and dotted(t):
            name = dotted(t)
            v = self.label(name, v, st)
            if v.frame is not None and v.frame.site is None:
                v = replace(v, frame=replace(v.frame, site=st, index=self.index))
            self.env[name] = v
            self.observe(v, st, "use")
        elif isinstance(t, (ast.Tuple, ast.List)):
            elts = t.elts
            if v.seq is not None and len(v.seq) == len(elts) and not any(isinstance(e, ast.Starred) for e in elts):
                for e, sv in zip(elts, v.seq):
                    self.bind(e, sv, st)
            else:
                base = replace(v, seq=None) if v.seq is None else union(*v.seq)
                for e in elts:
                    self.bind(e.value if isinstance(e, ast.Starred) else e, base, st)
        elif isinstance(t, ast.Starred):
            self.bind(t.value, v, st, refs)
        elif isinstance(t, ast.Subscript):
            self.bind_subscript(t, v, st, refs or [])

    def label(self, name: str, v: Val, st: ast.AST) -> Val:
        short = name.rsplit(".", 1)[-1]
        if v.roots and K.TARGET_NAME_RE.match(short):
            v = replace(v, target=True)
        role = K.split_role(short)
        if role is None or not v.roots or v.est is not None or v.traits & {"stat_value", "metric_value"}:
            return v
        roles = v.roles()
        if role in roles:
            return v
        if v.tags and all(t.origin[0] != "name" for t in v.tags) and len(roles) == 1:
            # X_val / X_train_linear from a positional split: same rows, clearer role
            return replace(v, tags=frozenset(replace(t, role=role) for t in v.tags))
        tag = Tag(("name", role), role)
        v = replace(v, tags=frozenset({tag}))
        if role in HELDOUT and "random_subset" in v.traits and "timeseries" in v.traits:
            self.an.report("UL402", st, "rows with lag/rolling features are split with a random sample")
        return v

    def bind_subscript(self, t: ast.Subscript, v: Val, st: ast.AST, refs: list) -> None:
        base = t.value
        if isinstance(base, ast.Attribute) and base.attr in ("loc", "iloc", "at", "iat"):
            base = base.value
        name = dotted(base)
        if not name:
            self.expr(t.value)
            return
        old = self.load(name)
        if old.est is not None or old.mod is not None:
            return
        col = self._col_key(t)
        new = union(old, v)
        frame = old.frame
        if col is not None and old.roots:
            frame = frame or Frame(family=old.roots, index=self.index)
            ca = ColAssign(next(self.an.ids), frame.family, col, st, self.index, [], old.roles())
            stop = st.value if isinstance(st, (ast.Assign, ast.AugAssign)) else st
            for ref_col, fam, node, roles in refs:
                sign, centered = shift_context(node, stop)
                ca.refs.append(Ref(ref_col, fam, roles, sign, centered))
            self.colassigns.append(ca)
            frame = replace(frame, added=frame.added | {ca.id},
                            dropped=frame.dropped - {col},
                            select=(frame.select | {col}) if frame.select is not None else None)
        new = replace(new, frame=frame, col=old.col, target=old.target, tags=old.tags | v.tags if old.tags else v.tags)
        if old.tags:
            new = replace(new, tags=old.tags)
        self.env[name] = new
        self.observe(new, st, "use")

    def observe(self, v: Val, node: ast.AST, what: str) -> None:
        if not v.fits or not v.roots:
            return
        held = v.heldout()
        if held:
            self.an.obs.append(Obs(v, frozenset(held), node, what))

    def observe_cv(self, x: Val, node: ast.AST) -> None:
        if x.fits and x.roots and not x.target:
            tag = Tag(("cv", id(node)), "test", x.tags)
            self.an.obs.append(Obs(x, frozenset({tag}), node, "cv"))

    # -- expressions --
    def expr(self, n: ast.AST | None) -> Val:
        if n is None:
            return NOTHING
        handler = getattr(self, "ex_" + type(n).__name__, None)
        if handler is not None:
            return handler(n)
        return union(*(self.expr(c) for c in ast.iter_child_nodes(n) if isinstance(c, ast.expr)))

    def ex_Constant(self, n: ast.Constant) -> Val:
        return Val(const=n.value) if isinstance(n.value, str) else NOTHING

    def ex_Name(self, n: ast.Name) -> Val:
        return self.load(n.id)

    def ex_JoinedStr(self, n) -> Val:
        return NOTHING

    def ex_Lambda(self, n: ast.Lambda) -> Val:
        saved = {}
        for a in n.args.args + n.args.posonlyargs + n.args.kwonlyargs:
            saved[a.arg] = self.env.get(a.arg)
            self.env[a.arg] = NOTHING
        try:
            v = self.expr(n.body)
        finally:
            for k, old in saved.items():
                if old is None:
                    self.env.pop(k, None)
                else:
                    self.env[k] = old
        return v

    def ex_NamedExpr(self, n: ast.NamedExpr) -> Val:
        v = self.expr(n.value)
        self.bind(n.target, v, self.stmt_node or n)
        return v

    def _seq(self, n) -> Val:
        vals = [self.expr(e) for e in n.elts]
        consts = [v.const for v in vals]
        const = tuple(consts) if consts and all(isinstance(c, str) for c in consts) else None
        u = union(*vals)
        return replace(u, seq=tuple(vals), const=const)

    ex_Tuple = _seq
    ex_List = _seq
    ex_Set = _seq

    def ex_Compare(self, n: ast.Compare) -> Val:
        left = self.expr(n.left)
        v = union(left, *(self.expr(c) for c in n.comparators))
        traits = v.traits | {"mask"}
        if left.col is not None:
            traits |= {f"cmpcol:{left.col[1]}"}
        return replace(v, traits=traits, col=None)

    def ex_BoolOp(self, n: ast.BoolOp) -> Val:
        v = union(*(self.expr(x) for x in n.values))
        return v

    def ex_Attribute(self, n: ast.Attribute) -> Val:
        name = dotted(n)
        if name and name in self.env:
            return self.env[name]
        if name and (name.split(".", 1)[0] in ("self", "cls") or n.attr in BUNCH_ATTRS
                     or K.split_role(n.attr) is not None):
            return self.load(name)
        base = self.expr(n.value)
        if base.mod is not None and not base.roots:
            return Val(mod=f"{base.mod}.{n.attr}")
        if base.const is not None and isinstance(base.const, ColumnSet):
            return base
        if base.est is not None:
            if n.attr in ("best_estimator_", "estimator_", "final_estimator_"):
                return Val(est=replace(base.est, kind=K.MODEL), fits=base.fits)
            return Val(fits=base.fits)
        if not base.roots:
            return base
        if n.attr in SCALAR_ATTRS:
            return NOTHING
        if n.attr == "columns":
            f = base.frame
            if f is not None and not f.unknown:
                return Val(const=tuple(sorted(f.select)) if f.select is not None else ColumnSet(f.dropped))
            return NOTHING
        if n.attr in K.COLUMN_PRESERVING_ATTRS or n.attr in VALUE_ATTRS:
            if n.attr == "iloc":
                return replace(base, frame=replace(base.frame, unknown=True) if base.frame else None)
            return base
        parent = getattr(n, "_ul_parent", None)
        is_method = isinstance(parent, ast.Call) and parent.func is n
        if not is_method and base.frame is not None and n.attr not in K.DATAFRAME_ATTRS:
            return self._colref(base, n.attr, n)
        return base

    def _colref(self, base: Val, col: str, node: ast.AST) -> Val:
        fam = base.frame.family if base.frame is not None else base.roots
        if K.ENTITY_COLUMN_RE.match(col):
            self.entity_cols.append((fam, col))
        if self.collector is not None:
            self.collector.append((col, fam, node, frozenset(base.roles())))
        return Val(base.roots, base.tags, base.fits, base.traits, col=(fam, col), target=base.target)

    def ex_Subscript(self, n: ast.Subscript) -> Val:
        base = self.expr(n.value)
        if base.est is not None or base.mod is not None:
            return union(Val(fits=base.fits), self.expr(n.slice))
        if base.const is not None and not base.roots:
            return NOTHING
        sl_node = n.slice
        is_loc = isinstance(n.value, ast.Attribute) and n.value.attr == "loc"
        col_node = sl_node
        row_node = None
        if is_loc and isinstance(sl_node, ast.Tuple) and len(sl_node.elts) == 2:
            row_node, col_node = sl_node.elts
        elif is_loc:
            row_node, col_node = sl_node, None
        sl = self.expr(col_node) if col_node is not None else NOTHING
        rows = self.expr(row_node) if row_node is not None else NOTHING
        if "corrmatrix" in base.traits and isinstance(sl.const, str):
            if sl.const in self.targets:
                return self.stat_event(n, "corr", base, supervised=True)
            return replace(base, traits=base.traits - {"corrmatrix"})
        if base.roots:
            if isinstance(sl.const, str) and not sl.roots:
                return self._colref(base, sl.const, n)
            cols = self._colset(sl)
            if cols is not None and base.frame is not None:
                frame = self._select(base.frame, cols)
                return replace(base.data_only(), frame=frame, tags=base.tags | rows.tags)
        # Row selection keeps the columns: slices, masks, split indices. Anything
        # else (df[some_list]) may pick columns we cannot see.
        keeps_columns = (col_node is None or isinstance(col_node, ast.Slice) or "mask" in sl.traits
                         or (not sl.roots and sl.const is None))
        for idx in (sl, rows):
            if idx is NOTHING:
                continue
            if "mask" in idx.traits:
                cls = {t[7:] for t in idx.traits if isinstance(t, str) and t.startswith("cmpcol:")} & self.targets
                base = replace(base, tags=base.tags | idx.tags,
                               traits=base.traits | {f"classfilter:{c}" for c in cls})
            else:
                base = replace(base.data_only(), tags=base.tags | idx.tags, fits=base.fits | idx.fits,
                               traits=base.traits | (idx.traits - {"mask"}),
                               roots=base.roots | idx.roots)
        if not keeps_columns and base.frame is not None:
            base = replace(base, frame=replace(base.frame, unknown=True))
        if isinstance(sl_node, ast.Slice) or (is_loc and isinstance(row_node, ast.Slice)):
            base = self._downsized(base)
        return base.data_only() if base.roots else base

    def _downsized(self, v: Val, up: bool = False) -> Val:
        """A class-filtered subset that was cut down (or blown up) to a chosen size."""
        cls = [t[12:] for t in v.traits if isinstance(t, str) and t.startswith("classfilter:")]
        if not cls:
            return v
        mark = "upsampled" if up else "downsampled"
        return replace(v, traits=v.traits | {f"{mark}:{c}" for c in cls})

    def _rebalance_event(self, n: ast.Call, parts: list[Val]) -> Val:
        """pd.concat of class subsets where one was resized: manual over/undersampling."""
        joined = union(*parts)
        by_col: dict[str, int] = {}
        for p in parts:
            for t in p.traits:
                if isinstance(t, str) and t.startswith("classfilter:"):
                    by_col[t[12:]] = by_col.get(t[12:], 0) + 1
        for col, count in by_col.items():
            up = any(f"upsampled:{col}" in p.traits for p in parts)
            down = any(f"downsampled:{col}" in p.traits for p in parts)
            if count >= 2 and (up or down):
                kind = K.OVERSAMPLER if up else K.UNDERSAMPLER
                how = "upsampled with replacement" if up else "cut down"
                ev = FitEvent(next(self.an.ids), kind, f"manual resampling of '{col}' ({how})", "concat", n,
                              replace(joined, traits=EMPTY))
                self.an.fit_events[ev.id] = ev
                drop = {t for t in joined.traits if isinstance(t, str)
                        and t.split(":", 1)[0] in ("classfilter", "downsampled", "upsampled")}
                return replace(joined, fits=joined.fits | {ev.id}, traits=joined.traits - drop)
        return joined

    def _colset(self, v: Val):
        if isinstance(v.const, tuple):
            return frozenset(v.const)
        if isinstance(v.const, ColumnSet):
            return v.const
        return None

    def _select(self, frame: Frame, cols) -> Frame:
        if isinstance(cols, ColumnSet):
            if frame.select is not None:
                return replace(frame, select=frame.select - cols.excluded, site=None, index=self.index, added=EMPTY)
            return replace(frame, dropped=frame.dropped | cols.excluded, site=None, index=self.index, added=EMPTY)
        return replace(frame, select=frozenset(cols), site=None, index=self.index, added=EMPTY)

    def _drop_cols(self, frame: Frame, cols, keep_site: bool = False) -> Frame:
        site = frame.site if keep_site else None
        index = frame.index if keep_site else self.index
        added = frame.added if keep_site else EMPTY
        if frame.select is not None:
            return replace(frame, select=frame.select - set(cols), site=site, index=index, added=added)
        return replace(frame, dropped=frame.dropped | set(cols), site=site, index=index, added=added)

    def _comprehension(self, n, elt_nodes) -> Val:
        loop = self.push_loop(n)
        saved = {}
        try:
            # df.columns filtered by != / not in: a column set
            if (isinstance(n, ast.ListComp) and len(n.generators) == 1
                    and isinstance(n.generators[0].target, ast.Name)):
                gen = n.generators[0]
                itv = self.expr(gen.iter)
                if isinstance(itv.const, ColumnSet) or isinstance(itv.const, tuple):
                    excluded = self._excluded(gen.target.id, gen.ifs)
                    if excluded is not None and isinstance(n.elt, ast.Name) and n.elt.id == gen.target.id:
                        if isinstance(itv.const, ColumnSet):
                            return Val(const=ColumnSet(itv.const.excluded | excluded))
                        return Val(const=tuple(c for c in itv.const if c not in excluded))
            out = []
            for gen in n.generators:
                itv = self.expr(gen.iter)
                for name_node in ast.walk(gen.target):
                    if isinstance(name_node, ast.Name):
                        saved.setdefault(name_node.id, self.env.get(name_node.id))
                self.bind(gen.target, self.iter_elem(itv), self.stmt_node or n)
                for cond in gen.ifs:
                    self.cond(cond)
            for e in elt_nodes:
                out.append(self.expr(e))
            return union(*out)
        finally:
            for k, old in saved.items():
                if old is None:
                    self.env.pop(k, None)
                else:
                    self.env[k] = old
            self.pop_loop(loop)

    def _excluded(self, var: str, ifs: list) -> frozenset | None:
        excluded: set = set()
        conds = []
        for c in ifs:
            if isinstance(c, ast.BoolOp) and isinstance(c.op, ast.And):
                conds.extend(c.values)
            else:
                conds.append(c)
        for c in conds:
            if not (isinstance(c, ast.Compare) and isinstance(c.left, ast.Name) and c.left.id == var
                    and len(c.ops) == 1):
                return None
            op, right = c.ops[0], self.expr(c.comparators[0])
            if isinstance(op, ast.NotEq) and isinstance(right.const, str):
                excluded.add(right.const)
            elif isinstance(op, ast.NotIn) and isinstance(right.const, tuple):
                excluded.update(right.const)
            else:
                return None
        return frozenset(excluded)

    def ex_ListComp(self, n):
        return self._comprehension(n, [n.elt])

    ex_SetComp = ex_ListComp
    ex_GeneratorExp = ex_ListComp

    def ex_DictComp(self, n):
        return self._comprehension(n, [n.key, n.value])

    # -- calls --
    def eval_args(self, n: ast.Call) -> tuple[list[Val], dict[str, Val]]:
        pos = [self.expr(a.value if isinstance(a, ast.Starred) else a) for a in n.args]
        kw = {}
        for k in n.keywords:
            kw[k.arg or "**"] = self.expr(k.value)
        return pos, kw

    def ex_Call(self, n: ast.Call) -> Val:
        f = n.func
        if isinstance(f, ast.Attribute):
            objv = self.expr(f.value)
            if objv.mod is not None and not objv.roots:
                return self.call_func(n, f.attr, f"{objv.mod}.{f.attr}")
            return self.call_method(n, objv, f.attr, f.value)
        if isinstance(f, ast.Name):
            v = self.env.get(f.id)
            if v is not None and v.mod is None and (v.est is not None or v.roots):
                return self.call_method(n, v, "__call__", f)
            return self.call_func(n, f.id, self.an.imports.get(f.id, f.id))
        fv = self.expr(f)
        pos, kw = self.eval_args(n)
        if fv.est is not None:
            return self.call_method(n, fv, "__call__", f, pos, kw)
        return union(fv, *pos, *kw.values())

    def call_func(self, n: ast.Call, name: str, qual: str) -> Val:
        name = K.last(name)
        if name in SCALAR_BUILTINS:
            self.eval_args(n)
            return NOTHING
        if name in ("enumerate", "zip", "list", "tuple", "iter", "reversed", "next", "sorted"):
            pos, kw = self.eval_args(n)
            if name == "enumerate" and pos:
                return replace(pos[0].data_only(), elem=Val(seq=(NOTHING, self.iter_elem(pos[0]))))
            if name == "zip":
                return replace(union(*pos), elem=Val(seq=tuple(self.iter_elem(p) for p in pos)))
            if name == "next" and pos:
                return self.iter_elem(pos[0])
            if name == "sorted" and "key" in kw:
                self.selection_evidence(union(*pos, *kw.values()))
            return pos[0] if pos else NOTHING
        if name in K.PIPELINE_BUILDERS:
            return self.make_pipeline(n, name)
        kind = K.estimator_kind(name)
        if kind is not None and name not in K.DATA_CONSTRUCTORS:
            return self.make_estimator(n, name, kind)
        if name in K.SPLIT_FUNCS:
            return self.split_call(n, name)
        if name in K.CV_FUNCS:
            return self.cv_call(n)
        pos, kw = self.eval_args(n)
        allv = pos + list(kw.values())
        if name in K.METRIC_FUNCS:
            return self.eval_event(n, union(*allv))
        if name in K.SUPERVISED_STAT_FUNCS and pos:
            return self.stat_event(n, name, pos[0], supervised=True)
        if (name in K.NUMPY_STAT_FUNCS and qual.split(".", 1)[0] in ("numpy", "np", "torch", "jax", "cupy") and pos
                and "prediction" not in pos[0].traits):
            axis = kwarg(n, "axis") or kwarg(n, "dim") or (n.args[1] if len(n.args) > 1 else None)
            if self._per_row(axis):
                return union(*allv)
            return self.stat_event(n, name, pos[0], supervised=False)
        if name in K.EARLY_STOPPING_NAMES:
            if name == "ModelCheckpoint" and not is_true(kwarg(n, "save_best_only")):
                return NOTHING
            return Val(traits=frozenset({"early_stop"}))
        if name in STRONG_SELECTION or name in WEAK_SELECTION:
            v = union(*allv)
            if name in STRONG_SELECTION or "key" in kw or (self.loop_stack and len(n.args) >= 2):
                self.selection_evidence(v)
            return v
        if name == "train" and qual.split(".", 1)[0] in ("xgboost", "xgb", "lightgbm", "lgb"):
            return self.native_train(n, pos, kw)
        if name in ("DMatrix", "Dataset", "Pool", "QuantileDMatrix") and pos:
            label = kw.get("label") or (pos[1] if len(pos) > 1 and name != "DMatrix" else None)
            if label is not None:
                self.pair_site(n, pos[0], label)
        if name in K.LOADERS or name.startswith(K.LOADER_PREFIXES):
            path = next((v.const for v in pos if isinstance(v.const, str)), None)
            role = K.path_role(path) if path else None
            v = self.new_root(name, role)
            if path:
                v = self._import_written(v, path)
            return v
        if name in WRITE_FUNCS and qual.split(".", 1)[0] in WRITE_FUNCS[name]:
            self._record_write(pos, kw, name)
            return NOTHING
        if name == "concat" and pos and pos[0].seq:
            return self._rebalance_event(n, list(pos[0].seq))
        if name == "resample" and qual.startswith("sklearn") and pos:
            up = is_true(kwarg(n, "replace"))
            return self._downsized(union(pos[0]), up=up)
        if name in K.DATA_CONSTRUCTORS or name in ("concat", "merge", "where", "get_dummies"):
            return union(*allv)
        return union(*allv)

    def call_method(self, n: ast.Call, objv: Val, m: str, objnode: ast.AST,
                    pos: list | None = None, kw: dict | None = None) -> Val:
        objname = dotted(objnode)
        if m == "assign" and objv.roots:
            return self.frame_assign(n, objv)
        if pos is None:
            pos, kw = self.eval_args(n)
        allv = pos + list(kw.values())
        est = objv.est
        if est is None and m in K.FIT_METHODS and not (objv.roots and m == "fit" and objv.frame is None and False):
            est = Est(cls=objname or "estimator", kind=self._guess_kind(objname or "", m))
            objv = Val(est=est)
        if est is not None:
            if m in K.FIT_METHODS:
                return self.fit_call(n, objv, est, m, pos, kw, objname)
            if m == "split" and est.kind == K.SPLITTER:
                return self.cv_split(n, est, pos, kw)
            if m in K.APPLY_METHODS or m in K.SCORE_METHODS:
                data = union(*[v for v in pos if v.roots]) if pos else NOTHING
                res = replace(data, fits=data.fits | objv.fits) if data.roots else Val(fits=objv.fits)
                if est.selects_columns:
                    res = replace(res, frame=None, col=None)
                if m in K.PREDICT_METHODS or m in ("score_samples", "kneighbors"):
                    res = replace(res, traits=res.traits | {"prediction"}, frame=None, col=None)
                if m in K.PREDICT_METHODS or m in K.SCORE_METHODS:
                    self.check_predict(n, est, data)
                if m in K.SCORE_METHODS and len(pos) >= 2 and not est.selects_columns:
                    self.pair_site(n, pos[0], pos[1])
                    return self.eval_event(n, union(*pos, Val(fits=objv.fits)))
                self.observe(res, n, "use")
                return res
            if m in ("set_params", "set_output", "get_booster", "best_estimator_"):
                return objv
            return union(Val(fits=objv.fits), *allv)
        # -- data / unknown objects --
        if isinstance(objv.const, (ColumnSet, tuple)) and m in ("drop", "difference", "tolist", "to_list", "copy"):
            cols = self._colset(pos[0]) if pos else None
            if m in ("drop", "difference") and isinstance(cols, frozenset):
                if isinstance(objv.const, ColumnSet):
                    return Val(const=ColumnSet(objv.const.excluded | cols))
                return Val(const=tuple(c for c in objv.const if c not in cols))
            return objv
        if m in STRONG_SELECTION:
            self.selection_evidence(objv)
        res = union(objv, *allv)
        if "window_obj" in res.traits:
            res = replace(res, traits=res.traits - {"window_obj"})
        if objv.roots and m in WRITE_METHODS:
            target = pos[0] if pos else kw.get("path_or_buf", kw.get("path", NOTHING))
            if isinstance(target.const, str):
                self._save(target.const, objv)
            return NOTHING
        if objv.roots:
            if "window_obj" in objv.traits:
                # df.rolling(7).mean(), df.groupby('g')['x'].mean(): not a global statistic
                if m in K.TIMESERIES_METHODS:
                    res = replace(res, traits=res.traits | {"timeseries"})
                return replace(res, col=objv.col, frame=None)
            if m in ("rolling", "ewm", "expanding", "resample", "groupby", "window"):
                traits = res.traits | {"window_obj"} | ({"timeseries"} if m != "groupby" else set())
                if m == "rolling" and is_true(kwarg(n, "center")):
                    traits |= {"centered"}
                return replace(objv.data_only(), traits=traits, roots=res.roots, tags=res.tags, fits=res.fits)
            missing_rate = (isinstance(objnode, ast.Call) and isinstance(objnode.func, ast.Attribute)
                            and objnode.func.attr in ("isnull", "isna", "notnull", "notna", "duplicated"))
            if m in K.PANDAS_STAT_METHODS and "prediction" not in objv.traits and not missing_rate:
                if self._per_row(kwarg(n, "axis") or (n.args[0] if n.args and m != "quantile" else None)):
                    return replace(res, col=None, frame=None)
                return self.stat_event(n, m, objv, supervised=False, extra=allv)
            if m == "corrwith":
                return self.stat_event(n, m, union(objv, *allv), supervised=True)
            if m == "corr":
                return replace(res, traits=res.traits | {"corrmatrix"}, frame=None, col=None)
            if m in K.TIMESERIES_METHODS:
                res = replace(res, traits=res.traits | {"timeseries"}, col=objv.col)
            elif m in K.RANDOM_SUBSET_METHODS:
                res = replace(res, traits=res.traits | {"random_subset"}, frame=objv.frame, col=objv.col)
                frac = kwarg(n, "frac")
                if kwarg(n, "n") is not None or n.args or (isinstance(frac, ast.Constant) and frac.value != 1):
                    res = self._downsized(res, up=is_true(kwarg(n, "replace")))
            elif m == "head" and objv.traits:
                res = self._downsized(replace(res, frame=objv.frame))
            elif m == "pop":
                c = pos[0].const if pos else None
                if isinstance(c, str):
                    if objname and objv.frame is not None:
                        self.env[objname] = replace(objv, frame=self._drop_cols(objv.frame, {c}, keep_site=True))
                    return self._colref(objv, c, n)
            elif m == "drop":
                res = replace(res, frame=self._frame_drop(n, objv, pos, kw), col=None, tags=objv.tags | res.tags)
            elif m == "groupby":
                res = replace(objv.data_only(), traits=objv.traits | {"groupby"})
            elif m in K.COLUMN_PRESERVING_METHODS:
                res = replace(res, frame=objv.frame, col=objv.col, tags=res.tags)
            else:
                res = replace(res, frame=None, col=objv.col)
        if objname is None and is_true(kwarg(n, "inplace")) and isinstance(objnode, ast.Subscript):
            # df["col"].fillna(..., inplace=True) updates df
            base_name = dotted(objnode.value)
            if base_name and base_name in self.env:
                old = self.env[base_name]
                self.env[base_name] = replace(union(old, res), frame=old.frame, tags=old.tags, target=old.target)
                self.observe(self.env[base_name], n, "use")
            return NOTHING
        if objname and (is_true(kwarg(n, "inplace")) or m in K.MUTATING_METHODS):
            if objname in self.env or objv.roots:
                keep = self.env.get(objname, objv)
                newv = replace(res, elem=keep.elem, seq=None) if m in K.MUTATING_METHODS else res
                if m in K.MUTATING_METHODS:
                    newv = replace(union(keep, *allv), frame=keep.frame, target=keep.target)
                self.env[objname] = newv
                self.observe(newv, n, "use")
            if is_true(kwarg(n, "inplace")):
                return NOTHING
        return res

    # -- files written here and read elsewhere --
    def _save(self, path: str, data: Val) -> None:
        events = [self.an.fit_events[i] for i in sorted(data.fits) if i in self.an.fit_events]
        events = [replace(e, path=e.path or self.an.path) for e in events
                  if e.kind not in (K.MODEL, K.SEARCH, K.STATELESS, K.ENCODER)]
        if events or "timeseries" in data.traits:
            self.an.writes.append(Written(path, events, data.traits & {"timeseries"}, self.an.path))

    def _record_write(self, pos: list, kw: dict, name: str) -> None:
        if name in ("save", "savez", "savez_compressed", "savetxt") and pos and isinstance(pos[0].const, str):
            if len(pos) > 1:
                self._save(pos[0].const, union(*pos[1:]))
        elif name == "dump" and len(pos) > 1 and isinstance(pos[1].const, str):
            self._save(pos[1].const, pos[0])

    def _import_written(self, v: Val, path: str) -> Val:
        written = self.an.lookup_written(path)
        if not written:
            return v
        ids, traits = set(), set()
        data = Val(roots=v.roots)
        for w in written:
            traits |= w.traits
            for ev in w.events:
                new = FitEvent(next(self.an.ids), ev.kind, ev.cls, ev.method, ev.node, data, path=ev.path)
                self.an.fit_events[new.id] = new
                ids.add(new.id)
        return replace(v, fits=v.fits | ids, traits=v.traits | traits)

    def _guess_kind(self, name: str, method: str) -> str:
        low = name.lower()
        for word, kind in (("smote", K.OVERSAMPLER), ("oversampl", K.OVERSAMPLER), ("undersampl", K.UNDERSAMPLER),
                           ("selector", K.SELECTOR), ("select", K.SELECTOR), ("scaler", K.TRANSFORMER),
                           ("imputer", K.TRANSFORMER), ("pca", K.TRANSFORMER), ("vectori", K.TRANSFORMER),
                           ("tfidf", K.TRANSFORMER), ("tokenizer", K.TRANSFORMER), ("normaliz", K.TRANSFORMER),
                           ("target_enc", K.TARGET_ENCODER), ("te_", K.TARGET_ENCODER)):
            if word in low:
                return kind
        if method in ("fit_resample", "fit_sample"):
            return K.OVERSAMPLER
        if method in ("fit_transform", "adapt", "fit_on_texts", "fit_on_sequences"):
            return K.UNKNOWN_TRANSFORMER
        return K.MODEL

    def _per_row(self, axis: ast.AST | None) -> bool:
        if axis is None:
            return False
        v = const_int(axis)
        if v is not None:
            return v in (1, -1)
        return isinstance(axis, ast.Constant) and axis.value in ("columns", 1)

    def _frame_drop(self, n: ast.Call, objv: Val, pos: list, kw: dict) -> Frame | None:
        frame = objv.frame
        if frame is None:
            return None
        axis = kwarg(n, "axis")
        cols_v = kw.get("columns")
        if cols_v is None:
            is_cols = (const_int(axis) == 1) or (isinstance(axis, ast.Constant) and axis.value == "columns")
            if not is_cols:
                return frame  # dropping rows
            cols_v = pos[0] if pos else kw.get("labels")
        if cols_v is None:
            return replace(frame, unknown=True)
        if isinstance(cols_v.const, str):
            cols = {cols_v.const}
        elif isinstance(cols_v.const, tuple):
            cols = set(cols_v.const)
        else:
            return replace(frame, unknown=True, site=None, index=self.index)
        self.entity_cols.extend((frame.family, c) for c in cols if K.ENTITY_COLUMN_RE.match(c))
        return self._drop_cols(frame, cols)

    def frame_assign(self, n: ast.Call, objv: Val) -> Val:
        frame = objv.frame or Frame(family=objv.roots, index=self.index)
        parts = [objv]
        for k in n.keywords:
            refs: list = []
            self.collector = refs
            try:
                v = self.expr(k.value)
            finally:
                self.collector = None
            parts.append(v)
            if k.arg:
                ca = ColAssign(next(self.an.ids), frame.family, k.arg, self.stmt_node, self.index, [], objv.roles())
                for col, fam, node, roles in refs:
                    sign, centered = shift_context(node, k.value)
                    ca.refs.append(Ref(col, fam, roles, sign, centered))
                self.colassigns.append(ca)
                frame = replace(frame, added=frame.added | {ca.id}, dropped=frame.dropped - {k.arg})
        return replace(union(*parts), frame=frame, tags=objv.tags)

    # -- estimators --
    def _early_stop_kw(self, n: ast.Call, kw: dict) -> bool:
        if "early_stopping_rounds" in kw or "early_stopping_round" in kw:
            return True
        cb = kw.get("callbacks")
        return cb is not None and "early_stop" in cb.traits

    def make_estimator(self, n: ast.Call, name: str, kind: str) -> Val:
        pos, kw = self.eval_args(n)
        if kind == K.SPLITTER:
            shuffle_default, grouped, temporal = K.SPLITTERS[name]
            shuffle = shuffle_default or is_true(kwarg(n, "shuffle"))
            return Val(est=Est(name, kind, shuffle=shuffle, grouped=grouped, temporal=temporal))
        early = self._early_stop_kw(n, kw)
        if name.startswith("CatBoost") and not is_false(kwarg(n, "use_best_model")):
            early = True
        init = None
        if kind == K.MODEL and name in K.STATSMODELS_MODELS:
            data = [v for v in pos + list(kw.values()) if v.roots]
            init = union(*data) if data else None
        cv = kw.get("cv") if kind == K.SEARCH else None
        inner_fits = frozenset().union(*(v.fits for v in pos + list(kw.values())))
        return Val(est=Est(name, kind, early_stop=early, init_data=init, cv=cv), fits=inner_fits)

    def make_pipeline(self, n: ast.Call, name: str) -> Val:
        pos, kw = self.eval_args(n)
        steps: list[Est] = []

        def collect(v: Val):
            if v.est is not None:
                steps.append(v.est)
            if v.seq:
                for s in v.seq:
                    collect(s)

        for v in pos + [kw.get("steps", NOTHING), kw.get("transformers", NOTHING)]:
            collect(v)
        selects = name in ("ColumnTransformer", "make_column_transformer") or any(s.selects_columns for s in steps)
        if name in ("ColumnTransformer", "make_column_transformer", "FeatureUnion", "make_union"):
            kinds = [s.kind for s in steps]
            kind = next((k for k in (K.TARGET_ENCODER, K.SELECTOR, K.TRANSFORMER) if k in kinds), K.STATELESS)
            return Val(est=Est(name, kind, selects_columns=selects))
        if not steps:
            return Val(est=Est(name, K.MODEL, selects_columns=selects))
        last = steps[-1]
        kind = last.kind
        if kind in (K.ENCODER, K.STATELESS):
            others = [s.kind for s in steps]
            kind = next((k for k in (K.TARGET_ENCODER, K.SELECTOR, K.TRANSFORMER) if k in others), kind)
        return Val(est=Est(name, kind, early_stop=last.early_stop, selects_columns=selects))

    def fit_call(self, n: ast.Call, objv: Val, est: Est, m: str, pos: list, kw: dict,
                 objname: str | None) -> Val:
        X = pos[0] if pos else kw.get("X", kw.get("x", NOTHING))
        y = pos[1] if len(pos) > 1 else kw.get("y")
        if not X.roots and est.init_data is not None:
            X = est.init_data
        kind = est.kind
        if kind == K.MODEL and m == "fit_transform" and est.cls in K.SELECTORS:
            kind = K.SELECTOR
        ev = FitEvent(next(self.an.ids), kind, est.cls, m, n, X)
        self.an.fit_events[ev.id] = ev
        if X.roots and "test" in X.roles() and not ({"train", "val"} & X.roles()):
            if kind not in (K.STAT, K.STATELESS) and not (kind == K.ENCODER and X.target):
                self.an.report(
                    "UL201", n,
                    f"{est.cls}.{m}() is fitted on test data `{src(n.args[0]) if n.args else 'X'}`; "
                    "fit on the training split and only transform/predict the test split",
                )
        if kind in (K.MODEL, K.SEARCH):
            self.check_early_stopping(n, est, kw)
        if y is not None and not est.selects_columns:
            self.pair_site(n, X, y)
        if kind == K.SEARCH or est.cls == "RFECV":
            self.cv_event(n, X, est.cv)
        fitted = replace(est, fitted_on=X, fit_node=n, kind=kind)
        estv = Val(est=fitted, fits=objv.fits | {ev.id})
        if objname:
            self.env[objname] = estv
        if m in ("fit", "partial_fit", "adapt", "fit_on_texts", "fit_on_sequences"):
            return estv
        if m in ("fit_resample", "fit_sample"):
            xr = replace(X.data_only(), fits=X.fits | {ev.id})
            yr = replace(y.data_only(), fits=y.fits | {ev.id}) if y is not None else NOTHING
            self.observe(xr, n, "use")
            return replace(union(xr, yr), seq=(xr, yr))
        res = replace(X.data_only(), fits=X.fits | {ev.id})
        if est.selects_columns:
            res = replace(res, frame=None, col=None)
        self.observe(res, n, "use")
        return res

    def native_train(self, n: ast.Call, pos: list, kw: dict) -> Val:
        dtrain = pos[1] if len(pos) > 1 else kw.get("dtrain", kw.get("train_set", NOTHING))
        est = Est("Booster", K.MODEL)
        ev = FitEvent(next(self.an.ids), K.MODEL, "train", "train", n, dtrain)
        self.an.fit_events[ev.id] = ev
        if dtrain.roots and "test" in dtrain.roles() and "train" not in dtrain.roles():
            self.an.report("UL201", n, "the booster is trained on test data")
        evalv = union(kw.get("evals"), kw.get("valid_sets"))
        if evalv.roots and self._early_stop_kw(n, kw):
            self.an.es_candidates.append((n, evalv))
        return Val(est=replace(est, fitted_on=dtrain, fit_node=n), fits=frozenset({ev.id}))

    def check_early_stopping(self, n: ast.Call, est: Est, kw: dict) -> None:
        evalv = union(*(kw[k] for k in K.EVAL_SET_KWARGS if k in kw))
        if not evalv.roots:
            return
        early = est.early_stop or self._early_stop_kw(n, kw)
        if est.cls.startswith("CatBoost") and is_false(kwarg(n, "use_best_model")):
            early = False
        elif est.cls.startswith("CatBoost") and is_true(kwarg(n, "use_best_model")):
            early = True
        if early:
            self.an.es_candidates.append((n, evalv))

    def check_predict(self, n: ast.Call, est: Est, data: Val) -> None:
        if est.kind not in (K.MODEL, K.SEARCH) or est.fitted_on is None or not data.roots:
            return
        fit = est.fitted_on
        if fit.tags and not any(t.role == "train" for t in fit.tags):
            return  # fitted on held-out rows only: UL201 already
        for t in data.heldout():
            if fit.tags <= t.parent and (fit.roots & data.roots):
                self.an.report(
                    "UL202", est.fit_node,
                    f"{est.cls} is trained on `{src(est.fit_node.args[0]) if getattr(est.fit_node, 'args', None) else 'data'}`, "
                    "which still contains the rows it is evaluated on",
                    [(getattr(n, "lineno", 0), "evaluated on held-out rows here")],
                )
                return

    # -- splits --
    def split_call(self, n: ast.Call, name: str) -> Val:
        pos, kw = self.eval_args(n)
        sid = next(self.an.ids)
        shuffle = not is_false(kwarg(n, "shuffle"))
        if name == "random_split":
            arrays = pos[:1]
            lengths = n.args[1] if len(n.args) > 1 else kwarg(n, "lengths")
            parts = len(lengths.elts) if isinstance(lengths, (ast.List, ast.Tuple)) else 2
        else:
            arrays = pos
            parts = 2
        roles = ["train", "test"] if parts == 2 else ["train", "val", "test"] if parts == 3 else ["train"] + ["test"] * (parts - 1)
        outputs = []
        for a in arrays:
            for i, role in enumerate(roles):
                tag = Tag((sid, i), role, a.tags)
                out = replace(a.data_only(), tags=frozenset({tag}))
                outputs.append(out)
                if role in HELDOUT:
                    self.observe(out, n, "split")
        for a in arrays:
            if a.roots and "timeseries" in a.traits and shuffle and name != "temporal_train_test_split":
                self.an.report("UL402", n, "rows with lag/rolling/diff features are split randomly "
                               "(train_test_split shuffles by default)")
                break
        self.check_ungrouped(n, grouped=False, data=union(*arrays))
        return replace(union(*outputs), seq=tuple(outputs))

    def cv_split(self, n: ast.Call, est: Est, pos: list, kw: dict) -> Val:
        X = pos[0] if pos else NOTHING
        sid = next(self.an.ids)
        tr = Val(tags=frozenset({Tag((sid, 0), "train", X.tags)}))
        te = Val(tags=frozenset({Tag((sid, 1), "test", X.tags)}))
        if X.roots and "timeseries" in X.traits and est.shuffle:
            self.an.report("UL402", n, f"rows with lag/rolling/diff features are cross-validated with shuffled {est.cls}")
        if not est.grouped and "groups" not in kw and len(pos) < 3:
            self.check_ungrouped(n, grouped=False, data=X)
        self.observe_cv(X, n)
        return Val(elem=Val(seq=(tr, te)))

    def cv_call(self, n: ast.Call) -> Val:
        pos, kw = self.eval_args(n)
        est = pos[0] if pos else kw.get("estimator", NOTHING)
        X = pos[1] if len(pos) > 1 else kw.get("X", NOTHING)
        y = pos[2] if len(pos) > 2 else kw.get("y")
        self.cv_event(n, X, kw.get("cv"), grouped="groups" in kw)
        if y is not None and not (est.est is not None and est.est.selects_columns):
            self.pair_site(n, X, y)
        return Val(roots=X.roots, traits=frozenset({"cv_score"}))

    def cv_event(self, n: ast.Call, X: Val, cv: Val | None, grouped: bool = False) -> None:
        self.observe_cv(X, n)
        splitter = cv.est if cv is not None and cv.est is not None and cv.est.kind == K.SPLITTER else None
        if X.roots and "timeseries" in X.traits and splitter is not None and splitter.shuffle:
            self.an.report("UL402", n, f"rows with lag/rolling/diff features are cross-validated with shuffled {splitter.cls}")
        if not grouped and not (splitter is not None and (splitter.grouped or splitter.temporal)):
            self.check_ungrouped(n, grouped=False, data=X)

    def check_ungrouped(self, n: ast.AST, grouped: bool, data: Val = NOTHING) -> None:
        if grouped or self.ungrouped_reported:
            return
        fam = data.frame.family if data.frame is not None else data.roots
        cols = sorted(c for f, c in self.entity_cols if f & fam)
        if not cols:
            return
        self.ungrouped_reported = True
        col = cols[0]
        self.an.report("UL403", n, f"the data has an entity id column ('{col}') but this split is not grouped; "
                       "check with `unleak data --group` that no entity is on both sides")

    # -- evaluation / statistics --
    def eval_event(self, n: ast.Call, v: Val) -> Val:
        self.an.evals.append(v)
        traits = set(v.traits)
        if "test" in v.roles():
            for loop in self.loop_stack:
                loop.test_eval = True
                traits.add(f"tm:{loop.id}")
        self.observe(v, n, "use")
        traits.add("metric_value")
        return Val(roots=v.roots, tags=v.tags, fits=v.fits, traits=frozenset(traits))

    def stat_event(self, n: ast.AST, name: str, data: Val, supervised: bool, extra: list | None = None) -> Val:
        if not data.roots:
            return union(data, *(extra or []))
        kind = K.SELECTOR if supervised else K.STAT
        ev = FitEvent(next(self.an.ids), kind, name, name, n, data)
        self.an.fit_events[ev.id] = ev
        v = union(data, *(extra or []))
        return replace(v, fits=v.fits | {ev.id}, frame=None, col=None, traits=v.traits | {"stat_value"})

    # -- target leakage at (X, y) pairs --
    def pair_site(self, n: ast.AST, X: Val, y: Val) -> None:
        if X is None or y is None or X.frame is None or X.frame.unknown or y.col is None:
            return
        fam_y, target = y.col
        frame = X.frame
        if not (frame.family & fam_y):
            return
        site = frame.site or self.stmt_node or n
        key = (id(site), target)
        if key in self.an.pair_seen:
            return
        self.an.pair_seen.add(key)
        if self.contains(frame, target):
            node = site if frame.site is not None else n
            self.an.report("UL301", node, f"target column '{target}' is still in the feature matrix "
                           f"(it is used as y at line {getattr(n, 'lineno', '?')})")
        for ca in self.colassigns:
            if not (ca.family & frame.family):
                continue
            if not (ca.index <= frame.index or ca.id in frame.added):
                continue
            if ca.col == target:
                if any(r.shift < 0 for r in ca.refs):
                    continue  # forecasting target built from future values
                for r in ca.refs:
                    if r.col != target and self.contains(frame, r.col):
                        self.an.report("UL303", ca.node, f"target '{target}' is computed from '{r.col}', "
                                       f"which is still a feature")
                        break
                continue
            if not self.contains(frame, ca.col):
                continue
            if ca.roles != {"train"}:
                bad = [r for r in ca.refs if r.col == target and r.shift <= 0 and r.roles != {"train"}]
                if bad:
                    how = "a future value of" if bad[0].shift < 0 else "the current value of"
                    self.an.report("UL302", ca.node, f"feature '{ca.col}' is computed from {how} the target '{target}'")
                    continue
            if any(r.shift < 0 for r in ca.refs):
                self.an.report("UL401", ca.node, f"feature '{ca.col}' uses future values (negative shift)")
            elif any(r.centered for r in ca.refs):
                self.an.report("UL401", ca.node, f"feature '{ca.col}' uses a centred rolling window, "
                               "which includes future rows")

    def contains(self, frame: Frame, col: str) -> bool:
        if frame.select is not None:
            if col in frame.select:
                return True
            return any(ca.col == col and ca.id in frame.added for ca in self.colassigns)
        if col in frame.dropped:
            return False
        cas = [ca for ca in self.colassigns if ca.col == col and (ca.family & frame.family)]
        if not cas:
            return True
        return any(ca.index <= frame.index or ca.id in frame.added for ca in cas)


def analyze(tree: ast.Module, path: str | None = None, external: dict | None = None) -> Analyzer:
    an = Analyzer(tree, path, external)
    an.run()
    return an
