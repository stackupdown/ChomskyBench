from typing import Any, Tuple, List, Set, Dict, Union
import re
import json
def normalize_structure(obj):
    """Recursively normalize the data structure: convert lists/sets into sorted tuples 
    (if the object is iterable and its elements are sortable).
    """
    if isinstance(obj, (list, tuple)):
        # Recursively normalize each element, then sort them (if the elements are sortable)
        normalized_items = [normalize_structure(item) for item in obj]
        try:
            # Attempt to sort the elements (if they are comparable)
            return tuple(sorted(normalized_items))
        except TypeError:
            # If sorting is not possible (like containing elements of different types), convert directly to a tuple
            return tuple(normalized_items)
    elif isinstance(obj, set):
        # Convert sets into sorted tuples as well
        try:
            return tuple([normalize_structure(item) for item in obj])
        except TypeError:
            return tuple(normalize_structure(item) for item in obj)
    elif isinstance(obj, dict):
        # For dictionaries (which theoretically should not appear), 
        # convert to sorted tuples of key-value pairs for safety.
        sorted_items = sorted(obj.items(), key=lambda x: str(x[0]))
        return tuple((normalize_structure(k), normalize_structure(v)) for k, v in sorted_items)
    else:
        return obj

class ParserError(Exception):
    pass

class Parser:
    def __init__(self, s: str):
        self.s = s
        self.n = len(s)
        self.i = 0

    def parse(self):
        self._skip_ws()
        val = self._parse_value()
        self._skip_ws()
        if self.i != self.n:
            # raise ParserError(f"Extra characters at end: {self.s[self.i:self.i+20]!r}")
            return val
        return val

    # ========== low-level ==========

    def _peek(self, k=0):
        j = self.i + k
        return self.s[j] if j < self.n else ''

    def _get(self):
        ch = self._peek()
        if ch:
            self.i += 1
        return ch

    def _skip_ws(self):
        while self._peek() and self._peek().isspace():
            self.i += 1

    # ========== value parsers ==========

    def _parse_value(self):
        self._skip_ws()
        ch = self._peek()
        if not ch:
            raise ParserError("Unexpected end of input")
        if ch == '(':
            return self._parse_tuple()
        if ch == '[':
            return self._parse_list()
        if ch == '{':
            return self._parse_set()
        if ch in ('"', "'"):
            return self._parse_string()
        return self._parse_atom()

    def _parse_tuple(self):
        assert self._get() == '('
        self._skip_ws()
        items = []
        if self._peek() == ')':
            self._get()
            return tuple(items)
        while True:
            self._skip_ws()
            items.append(self._parse_value())
            self._skip_ws()
            ch = self._peek()
            if ch == ',':
                self._get()
                self._skip_ws()
                if self._peek() == ')':  # allow trailing comma
                    self._get()
                    break
                continue
            elif ch == ')':
                self._get()
                break
            else:
                raise ParserError(f"Expected ',' or ')', got {ch!r} at pos {self.i}")
        return tuple(items)

    def _parse_list(self):
        assert self._get() == '['
        self._skip_ws()
        items = []
        if self._peek() == ']':
            self._get()
            return items
        while True:
            items.append(self._parse_value())
            self._skip_ws()
            ch = self._peek()
            if ch == ',':
                self._get()
                self._skip_ws()
                if self._peek() == ']':  # trailing comma
                    self._get()
                    break
                continue
            elif ch == ']':
                self._get()
                break
            else:
                raise ParserError(f"Expected ',' or ']', got {ch!r} at pos {self.i}")
        return items

    def _parse_set(self):
        assert self._get() == '{'
        self._skip_ws()
        items = list()
        if self._peek() == '}':
            self._get()
            return tuple(items)
        while True:
            val = self._parse_value()
            try:
                hash(val)
            except TypeError:
                raise ParserError("Only hashable elements allowed in set")
            items.append(val)
            self._skip_ws()
            ch = self._peek()
            if ch == ',':
                self._get()
                self._skip_ws()
                if self._peek() == '}':  # trailing comma
                    self._get()
                    break
                continue
            elif ch == '}':
                self._get()
                break
            else:
                raise ParserError(f"Expected ',' or '}}', got {ch!r} at pos {self.i}")
        return tuple(items)

    def _parse_string(self):
        quote = self._get()  # ' or "
        buf = []
        while True:
            ch = self._get()
            if not ch:
                raise ParserError("Unterminated string literal")
            if ch == '\\':
                esc = self._get()
                if esc == 'n':
                    buf.append('\n')
                elif esc == 't':
                    buf.append('\t')
                elif esc == 'r':
                    buf.append('\r')
                elif esc == '\\':
                    buf.append('\\')
                elif esc == quote:
                    buf.append(quote)
                else:
                    # keep unknown escape
                    buf.append('\\' + (esc or ''))
            elif ch == quote:
                break
            else:
                buf.append(ch)
        return ''.join(buf)

    def _parse_atom(self):
        start = self.i
        while True:
            ch = self._peek()
            if not ch or ch.isspace() or ch in ',()[]{}':
                break
            self.i += 1
        raw = self.s[start:self.i]
        if raw == '':
            raise ParserError(f"Empty atom at pos {self.i}")

        # bool
        low = raw.lower()
        if low == 'true':
            return True
        if low == 'false':
            return False

        # integer
        if re.fullmatch(r'[+-]?\d+', raw):
            try:
                return int(raw)
            except ValueError:
                pass

        # treat others as strings（q0、qaccept、q0→q1→...）
        return raw

# ========== normalization & canonicalization ==========

def _sort_key(x):
    if isinstance(x, bool):
        return ('B', x)
    if isinstance(x, int):
        return ('I', x)
    if isinstance(x, str):
        return ('S', x)
    if isinstance(x, tuple):
        return ('T', tuple(_sort_key(e) for e in x))
    return ('R', repr(x))

def normalize(obj):
    # normalize recursively
    if isinstance(obj, set):
        # set -> tuple, normalize then sort
        elems = [normalize(e) for e in obj]
        elems.sort(key=_sort_key)
        return tuple(elems)
    if isinstance(obj, list):
        return tuple([normalize(e) for e in obj])
    if isinstance(obj, tuple):
        return tuple(normalize(e) for e in obj)
    # keep scalars
    return obj

def is_tuple_of_scalars(x):
    # Determine whether it is a "tuple of scalars," commonly used to identify "a tuple from a set."
    if not isinstance(x, tuple):
        return False
    return all(isinstance(e, (str, int, bool)) for e in x)

def canonicalize_answer1_structure(x):
    """
    If the top level is a tuple and it starts with >= 2 tuples of scalars (from a set),
    then pack this prefix into a list to obtain (list_of_state_sets, ...), aligning it with answer2.
    In all other cases, leave it unchanged.
    """
    if not isinstance(x, tuple):
        return x
    elems = list(x)
    if elems and isinstance(elems[0], list):
        return x  # already a list
    i = 0
    prefix = []
    while i < len(elems) and is_tuple_of_scalars(elems[i]):
        prefix.append(elems[i])
        i += 1
    if len(prefix) >= 2:
        new_first = list(prefix)
        rest = elems[i:]
        return (new_first, *rest)
    return x

def canonicalize(x, treat_as_answer1=False):
    x = normalize(x)
    return x

# ========== helper: high-level API ==========

def parse_and_canonicalize(s: str, treat_as_answer1=False):
    try:
        parsed = Parser(s).parse()
    except ParserError as e:
        # print(f"Parse error: {e} at pos {e.args[0]}")
        return s, s
    canon = canonicalize(parsed, treat_as_answer1=treat_as_answer1)
    return parsed, canon

def are_equivalent(s1, s2):
    # print("{}->{}".format(s1, obj1))
    # obj2 = parse_string_to_object(s2)
    _, obj1 = parse_and_canonicalize(s1)
    _, obj2 = parse_and_canonicalize(s2)

    norm1 = normalize_structure(obj1)
    norm2 = normalize_structure(obj2)

    correct_item = 0
    if type(norm1) == tuple and type(norm2) == tuple:
        for i in range(min(len(norm1), len(norm2))):
            if norm1[i] == norm2[i]:
                correct_item += 1
        pass_ratio = correct_item / len(norm1)
    else:
        pass_ratio = int(norm1 == norm2)
    return norm1, norm2, norm1 == norm2, pass_ratio

def load_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results

def load_jsonl(file):
    results = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if line.strip():
                problem = json.loads(line)
                results.append(problem)
    return results


def dump_jsonl(data, file, ensure_ascii=True):
    with open(file, 'w') as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=ensure_ascii) + '\n')

    print(f"dumped {len(data)} lines to {file}")

def get_answer_n(s1):
    _, obj1 = parse_and_canonicalize(s1)
    norm1 = normalize_structure(obj1)
    return len(norm1)

def get_order(s):
    res = -1
    mp = {'re_': 0, 'csg': 1, 'cfg': 2, 'dcfg': 3, 'regular_': 4}
    for k, v in mp.items():
        if s['id'].startswith(k):
            res = v
            break
    return res

def are_equivalent_of_ratios(s1, s2):
    _, obj1 = parse_and_canonicalize(s1)
    _, obj2 = parse_and_canonicalize(s2)

    norm1 = normalize_structure(obj1)
    norm2 = normalize_structure(obj2)

    correct_item = 0
    pass_ratios = []
    if type(norm1) == tuple and type(norm2) == tuple:
        for i in range(min(len(norm1), len(norm2))):
            if norm1[i] == norm2[i]:
                correct_item += 1
                pass_ratios.append(1)
            else:
                pass_ratios.append(0)
    else:
        pass_ratios = [int(norm1 == norm2)]
    return norm1, norm2, norm1 == norm2, pass_ratios


def box_pattern(max_depth: int) -> re.Pattern:
    """
    generate re expression for matching braces with max_depth
    depth=1:  \{[^{}]*\}
    depth=2:  \{(?:[^{}]|\{[^{}]*\})*\}
    depth=3:  \{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}
    """
    if max_depth < 1:
        raise ValueError("max_depth must be >= 1")

    # pattern of depth=1
    pat = r'\{[^{}]*\}'

    # Recursively construct higher layers: the pattern for depth=d takes depth=d-1 as inner "atomic unit"
    for _ in range(2, max_depth + 1):
        pat = r'\{(?:[^{}]|' + pat + r')*\}'

    pat = r'[\\]?box.*?' + pat
    return re.compile(pat)


def find_blocks(text: str, max_depth: int = 2):
    """
    extract \box{...} with limited max_depth,
    if max_depth is not enough, only extract max_depth level from inner
    """
    text = text.strip()
    text = '\n'.join(text.split('\n')[-5:])

    pat = box_pattern(max_depth)
    out = []
    for m in pat.finditer(text):
        s0 = m.group(0)
        open_rel = s0.index('{')
        out.append(s0[open_rel + 1:-1])
    return out

def remove_text(s):
    pattern = r'\\text\{([^}]*)\}'
    return re.sub(pattern, r'\1', s)

def extract_answer(completion, model):
    # match = re.search(r'The final answer is:.*?###(.*?)###', completion, re.DOTALL)
    groups = find_blocks(completion, 3)
    if groups:
        return remove_text(groups[0].strip())
    else:
        if model == "gemini-2.5-pro":
            match = re.findall(r'```box.*?```', completion, re.DOTALL)
            if match:
                return remove_text(match[-1].replace('box', '').strip('`').strip())
        return None
