from __future__ import annotations

import argparse
import ast
import json
import random
import re
import time
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = SimpleNamespace()

SYSTEM_PROMPT = """You are an expert logical solver governed by a strict reasoning protocol. For any problem, you must apply the following steps sequentially:

1. **RULES UNDERSTANDING:** Before attempting to solve, explicitly identify and list the rules or constraints that govern the problem.
2. **HINTS UNDERSTANDING:** If hints are present, explicitly identify and list them.
3. **STATE OBSERVATION:** identify and list the current resources, positions, or variables.
4. **ACTION PROPOSAL:** formulate a specific move or change to the state. Avoid proposing actions that immediately reverse the previous valid move unless no other legal options exist.
5. **CONSTRAINT OR RULES VERIFICATION:** validate the proposed action against the rules and the current state. Check for any conflicts or impossibilities. Do not proceed if the action violates the established boundaries.
6. **EXECUTION or CORRECTION:**
    - If a move is VALID: execute the move and update the state.
    - If a move is NOT VALID: state the violation, BACKTRACK, and propose an alternative.
    - ALWAYS explain the available resources, positions, or variables in the new state (inventory audit).
7. **GOAL CHECK:** after each valid move, check if the goal state is reached."""

# --- symbol registry (values), used to build a per-json env dynamically ---

names_female = ["Emily", "Sophie", "Mary", "Julia", "Sarah", "Olivia", "Emma"]
names_male = ["Liam", "Noah", "James", "Ethan", "Mason", "Logan", "Lucas"]


def sample(items: Sequence[Any], k: int = 1) -> Any:
    if k == 1:
        return random.choice(list(items))
    return random.sample(list(items), k)


def sample_sequential(items: Sequence[Any], k: int) -> List[Any]:
    items = list(items)
    if not items:
        raise ValueError("sample_sequential received empty sequence")
    if k <= len(items):
        start = random.randint(0, len(items) - k)
        return items[start : start + k]
    out: List[Any] = []
    start = random.randint(0, len(items) - 1)
    for i in range(k):
        out.append(items[(start + i) % len(items)])
    return out


def shuffle_list(items: Sequence[Any]) -> List[Any]:
    out = list(items)
    random.shuffle(out)
    return out


def numbers_within(low: int, high: int) -> int:
    return random.randint(low, high)


def frange(start: float, stop: float | None = None, step: float = 1.0) -> List[float]:
    if stop is None:
        start, stop = 0.0, float(start)
    vals: List[float] = []
    x = float(start)
    if step == 0:
        raise ValueError("step must be non-zero")
    if step > 0:
        while x <= float(stop) + 1e-12:
            vals.append(round(x, 10))
            x += step
    else:
        while x >= float(stop) - 1e-12:
            vals.append(round(x, 10))
            x += step
    return vals


def is_int(val: Any, tol: float = 1e-9) -> bool:
    if isinstance(val, int):
        return True
    try:
        f = float(val)
    except Exception:
        return False
    return abs(f - round(f)) <= tol


def divides(a: int, b: int) -> bool:
    return b != 0 and a % b == 0


def fix_floats(val: Any, ndigits: int = 10) -> float:
    return round(float(val), ndigits)


SAFE_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "int": int,
    "float": float,
    "round": round,
    "list": list,
    "tuple": tuple,
    "set": set,
    "sorted": sorted,
}

SYMBOLS: Dict[str, Any] = {
    # runtime modules/helpers
    "np": np,
    "Fraction": Fraction,
    "sample": sample,
    "sample_sequential": sample_sequential,
    "shuffle_list": shuffle_list,
    "numbers_within": numbers_within,
    "frange": frange,
    "is_int": is_int,
    "divides": divides,
    "fix_floats": fix_floats,
    "range": range,
    # data constants
    "names_female": names_female,
    "names_male": names_male,
    "names": names_female + names_male,
    "weekdays": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "colors": ["red", "blue", "green", "yellow", "orange", "purple", "black", "white"],
    "fruits": ["apple", "banana", "orange", "pear", "grape", "mango", "peach"],
    "sports": ["soccer", "basketball", "tennis", "baseball", "swimming", "running"],
    "cities": ["New York", "London", "Paris", "Tokyo", "Sydney", "Toronto"],
    "currencies_sym": ["$", "€", "£", "¥"],
    "multi_times": [2, 3, 4, 5, 6],
    "multiple_ice": [2, 3, 4],
    "multiple": ["double", "triple", "quadruple"],
    "fractions": [Fraction(1, 2), Fraction(1, 3), Fraction(2, 3), Fraction(1, 4), Fraction(3, 4)],
    "fraction_nums": [Fraction(1, 2), Fraction(1, 3), Fraction(2, 3), Fraction(1, 4), Fraction(3, 4), Fraction(1, 5)],
    "fraction_decimals": [0.25, 0.5, 0.75, 0.2, 0.4, 0.6],
    "fraction_alnum": [Fraction(1, 2), Fraction(1, 3), Fraction(2, 3), Fraction(1, 4), Fraction(3, 4)],
    "fraction_alph": ["half", "one-third", "two-thirds", "one-fourth", "three-fourths"],
    "weights_sm": [1, 2, 3, 4, 5],
    "weights_med": [10, 15, 20, 25, 30],
    "length_lg": [50, 75, 100, 125, 150, 200],
}


ALLOWED_BUILTIN_NAMES = {
    "True", "False", "None", "and", "or", "not", "in", "is",
    "abs", "min", "max", "sum", "len", "int", "float", "round", "list", "tuple", "set", "sorted",
}


def _extract_block(text: str, key: str) -> List[str]:
    match = re.search(rf"#{key}:\s*\n(.*?)(?=\n#\w+:|\Z)", text, flags=re.S)
    if not match:
        return []
    return [ln.strip()[2:].strip() for ln in match.group(1).splitlines() if ln.strip().startswith("- ")]


def _resolve_sample_space(value: Any) -> Any:
    if isinstance(value, range):
        return random.choice(list(value))
    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        return random.choice(seq) if seq else value
    return value


def _collect_required_symbols(text: str) -> set[str]:
    """Analyze ONLY this json text and return external symbol names it needs."""
    lines = _extract_block(text, "init") + _extract_block(text, "conditions")
    answer_match = re.search(r"#answer:\s*(.+)", text)
    if answer_match:
        lines.append(answer_match.group(1).strip())

    required: set[str] = set()
    assigned: set[str] = set()

    for raw in lines:
        stmt = raw.replace("$", "")
        try:
            tree = ast.parse(stmt)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                assigned.add(node.id)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in assigned and node.id not in ALLOWED_BUILTIN_NAMES:
                    required.add(node.id)
    return required


def build_env_for_template(text: str) -> Dict[str, Any]:
    required = _collect_required_symbols(text)
    env: Dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
    for name in sorted(required):
        if name in SYMBOLS:
            env[name] = SYMBOLS[name]
    # Always available helpers in case AST misses dynamic access
    env.setdefault("range", range)
    env.setdefault("sample", sample)
    env.setdefault("Fraction", Fraction)
    env.setdefault("np", np)
    return env


def execute_init_and_conditions(text: str, max_attempts: int = 4000) -> Tuple[Dict[str, Any], Any]:
    init_lines = _extract_block(text, "init")
    condition_lines = _extract_block(text, "conditions")
    answer_match = re.search(r"#answer:\s*(.+)", text)
    answer_expr = answer_match.group(1).strip() if answer_match else None
    template_env = build_env_for_template(text)

    for _ in range(max_attempts):
        local_env: Dict[str, Any] = {}
        try:
            for line in init_lines:
                if line.startswith("$"):
                    lhs, rhs = line[1:].split("=", 1)
                    local_env[lhs.strip()] = _resolve_sample_space(eval(rhs.strip(), template_env, local_env))
                else:
                    exec(line, template_env, local_env)

            if not all(bool(eval(c, template_env, local_env)) for c in condition_lines):
                continue

            answer_value = eval(answer_expr, template_env, local_env) if answer_expr else None
            return local_env, answer_value
        except Exception:
            continue
    raise RuntimeError("Failed to satisfy #conditions within max_attempts")


def _stringify(value: Any) -> str:
    if isinstance(value, Fraction):
        return str(value)
    return str(value)


def render_annotated_text(template: str, local_env: Dict[str, Any]) -> str:
    def repl(match: re.Match[str]) -> str:
        inside = match.group(1)
        if "," not in inside:
            return match.group(0)
        key, default = inside.split(",", 1)
        key = key.strip().lstrip("$")
        default = default.strip()
        if key in local_env:
            return _stringify(local_env[key])
        return default

    text = re.sub(r"\{([^{}]+)\}", repl, template)
    text = re.sub(r"\n\s*#init:.*", "", text, flags=re.S)
    return text.strip()


def format_assistant(answer_text: str) -> str:
    if "####" in answer_text:
        reasoning, final = answer_text.rsplit("####", 1)
        return f"<think>{reasoning.strip()}</think>\n#### {final.strip()}"
    return f"<think>{answer_text.strip()}</think>"


def load_templates(template_dirs: Sequence[Path]) -> List[Tuple[Path, Dict[str, Any]]]:
    records: List[Tuple[Path, Dict[str, Any]]] = []
    for d in template_dirs:
        for fp in sorted(d.glob("*.json")):
            with fp.open("r", encoding="utf-8") as f:
                records.append((fp, json.load(f)))
    return records


def generate_samples(records: List[Tuple[Path, Dict[str, Any]]], instances: int, backtracking_ratio: float, verbose: bool = False) -> List[Dict[str, str]]:
    if not 0 <= backtracking_ratio <= 1:
        raise ValueError("backtracking_ratio must be in [0,1]")

    start = time.time()
    samples: List[Dict[str, str]] = []
    total_templates = len(records)

    for idx, (fp, obj) in enumerate(records, 1):
        q_ann = obj.get("question_annotated") or obj.get("question")
        a_ann = obj.get("answer_annotated") or obj.get("answer")
        a_back = obj.get("answer_annotated_backtrack") or obj.get("answer_backtrack") or a_ann
        if not isinstance(q_ann, str) or not isinstance(a_ann, str):
            continue

        back_count = round(instances * backtracking_ratio)
        normal_count = instances - back_count
        modes = [False] * normal_count + [True] * back_count
        random.shuffle(modes)

        for j, use_backtrack in enumerate(modes, 1):
            try:
                env, _ = execute_init_and_conditions(q_ann)
                user_q = render_annotated_text(q_ann, env)
                answer_tpl = a_back if use_backtrack else a_ann
                assistant = format_assistant(render_annotated_text(answer_tpl, env))
                status = "ok"
            except Exception:
                user_q = (obj.get("question") or q_ann).strip()
                answer_src = obj.get("answer_backtrack") if use_backtrack else obj.get("answer")
                assistant = format_assistant((answer_src or a_ann).strip())
                status = "fallback"
            samples.append({"system": SYSTEM_PROMPT, "user": user_q, "assistant": assistant})

            if verbose:
                mode = "backtrack" if use_backtrack else "normal"
                print(f"[{idx}/{total_templates}] {fp.name} instance {j}/{instances} mode={mode} status={status}")

        if verbose and idx % 25 == 0:
            elapsed = time.time() - start
            print(f"-- progress: {idx}/{total_templates} templates, {len(samples)} samples, elapsed={elapsed:.1f}s")

    random.shuffle(samples)
    if verbose:
        print(f"Done: generated {len(samples)} samples in {time.time()-start:.1f}s")
    return samples


def write_jsonl(samples: List[Dict[str, str]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate combined SFT JSONL from GSM symbolic templates")
    parser.add_argument("--templates-root", type=Path, default=Path("templates"))
    parser.add_argument("--subdirs", nargs="+", default=["symbolic", "p1", "p2"])
    parser.add_argument("--instances", type=int, default=1, help="Instances per template")
    parser.add_argument("--backtracking-ratio", type=float, default=0.3, help="Fraction of backtracking samples")
    parser.add_argument("--output", type=Path, default=Path("generated_data/sft_combined.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true", help="Print progress while generating")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    template_dirs = [args.templates_root / s for s in args.subdirs]
    records = load_templates(template_dirs)
    if args.verbose:
        print(f"Loaded {len(records)} templates from {[str(d) for d in template_dirs]}")

    samples = generate_samples(
        records,
        instances=args.instances,
        backtracking_ratio=args.backtracking_ratio,
        verbose=args.verbose,
    )
    write_jsonl(samples, args.output)
    print(f"Wrote {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
