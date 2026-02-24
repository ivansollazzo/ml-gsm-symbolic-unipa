import argparse
import json
import random
import re
from fractions import Fraction
from pathlib import Path
from typing import List, Tuple


class NP:
    class random:
        @staticmethod
        def randint(low, high=None, size=None):
            if high is None:
                low, high = 0, low
            if size is None:
                return random.randint(int(low), int(high) - 1)
            return [random.randint(int(low), int(high) - 1) for _ in range(int(size))]

    @staticmethod
    def arange(start, stop=None, step=1):
        if stop is None:
            start, stop = 0, start
        out = []
        x = float(start)
        step = float(step)
        while x < float(stop) - 1e-9:
            out.append(round(x, 10))
            x += step
        return out


np = NP()

SYSTEM_PROMPT = """You are an expert logical solver governed by a strict reasoning protocol. For any problem, you must apply the following steps sequentially:

1. **RULES UNDERSTANDING:** Before attempting to solve, explicitly identify and list the rules or constraints that govern the problem.
2. **HINTS UNDERSTANDING:** If hints are present, explicitly identify and list them. 
3. **STATE OBSERVATION:** identify and list the current resources, positions, or variables.
4. **ACTION PROPOSAL:** formulate a specific move or change to the state. Avoid proposing actions that immediately reverse the previous valid move unless no other legal options exist.
5. **CONSTRAINT OR RULES VERIFICATION:** validate the proposed action against the rules and the current state. Check for any conflicts or impossibilities. Do not proceed if the action violates the established boundaries.
6. **EXECUTION or CORRECTION:
    - If a move is VALID: execute the move and update the state.
    - If a move is NOT VALID: state the violation, BACKTRACK, and propose an alternative.
    - ALWAYS explain the available resources, positions, or variables in the new state (inventory audit).
7. **GOAL CHECK:** after each valid move, check if the goal state is reached."""

SEED = 42
random.seed(SEED)

# helper pools used by templates
names_female = ["Alice", "Maria", "Giulia", "Sara", "Elena", "Anna", "Sofia", "Laura"]
names_male = ["Luca", "Marco", "Giovanni", "Paolo", "Matteo", "Davide", "Andrea", "Stefano"]
names = sorted(set(names_female + names_male))
cities = ["Rome", "Milan", "Turin", "Bologna", "Naples", "Florence"]
colors = ["red", "blue", "green", "yellow", "purple", "orange"]
currencies_sym = ["$", "€", "£"]
fruits = ["apple", "banana", "orange", "pear", "peach", "kiwi"]
sports = ["soccer", "tennis", "basketball", "volleyball", "swimming"]
weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
length_lg = ["kilometer", "meter", "yard", "foot"]
weights_sm = ["gram", "ounce"]
weights_med = ["kilogram", "pound"]
multi_times = [2, 3, 4]
multiple = [2, 3, 4]
multiple_ice = [2, 3]
fractions = [0.25, 1 / 3, 0.5, 2 / 3, 0.75]
fraction_nums = fractions
fraction_decimals = fractions
fraction_alph = [0.5, 1 / 3, 0.25, 2 / 3, 0.75]
fraction_alnum = sorted(set([i / 10 for i in range(1, 10)] + [0.25, 1 / 3, 0.5, 2 / 3, 0.75]))


def sample(values, k=None):
    if k is None:
        return random.choice(list(values))
    return random.sample(list(values), int(k))


def sample_sequential(values, k=2):
    vals = list(values)
    k = int(k)
    if k <= 0:
        return []
    start = random.randint(0, max(len(vals) - k, 0))
    return vals[start : start + k]


def frange(start, stop=None, step=1.0):
    if stop is None:
        start, stop = 0.0, start
    out = []
    x = float(start)
    while x <= float(stop) + 1e-9:
        out.append(round(x, 10))
        x += float(step)
    return out


def numbers_within(start, stop):
    return list(range(int(start), int(stop) + 1))


def fix_floats(x):
    if isinstance(x, (list, tuple)):
        return [fix_floats(v) for v in x]
    return int(x) if float(x).is_integer() else round(float(x), 2)


def shuffle_list(values):
    vals = list(values)
    random.shuffle(vals)
    return vals


def divides(a, b):
    return b != 0 and a % b == 0


def is_int(x):
    return float(x).is_integer()


SAFE_GLOBALS = {
    "__builtins__": {},
    "np": np,
    "int": int,
    "float": float,
    "round": round,
    "abs": abs,
    "min": min,
    "max": max,
    "len": len,
    "sum": sum,
    "Fraction": Fraction,
    "list": list,
    "range": lambda *a: list(range(*[int(v) for v in a])),
    "sample": sample,
    "sample_sequential": sample_sequential,
    "numbers_within": numbers_within,
    "fix_floats": fix_floats,
    "frange": frange,
    "shuffle_list": shuffle_list,
    "divides": divides,
    "is_int": is_int,
    "names": names,
    "names_female": names_female,
    "names_male": names_male,
    "cities": cities,
    "colors": colors,
    "currencies_sym": currencies_sym,
    "fruits": fruits,
    "sports": sports,
    "weekdays": weekdays,
    "length_lg": length_lg,
    "weights_sm": weights_sm,
    "weights_med": weights_med,
    "multi_times": multi_times,
    "multiple": multiple,
    "multiple_ice": multiple_ice,
    "fractions": fractions,
    "fraction_nums": fraction_nums,
    "fraction_decimals": fraction_decimals,
    "fraction_alph": fraction_alph,
    "fraction_alnum": fraction_alnum,
}

PLACEHOLDER = re.compile(r"\{([^{}]+)\}")


def parse_annotated(annotated: str) -> Tuple[str, str, str]:
    question = annotated.split("\n#init:", 1)[0].strip()
    init, cond = "", ""
    if "\n#init:" in annotated:
        rest = annotated.split("\n#init:", 1)[1]
        if "\n#conditions:" in rest:
            init, rest = rest.split("\n#conditions:", 1)
        else:
            init, rest = rest, ""
        if "\n#answer:" in rest:
            cond, _ = rest.split("\n#answer:", 1)
        else:
            cond = rest
    return question.strip(), init.strip(), cond.strip()


def eval_expr(expr: str, env: dict):
    return eval(expr, SAFE_GLOBALS, env)


def instantiate_vars(init_block: str, cond_block: str, max_trials: int = 250000):
    init_lines = [ln.strip() for ln in init_block.splitlines() if ln.strip().startswith("-")]
    cond_lines = [ln.strip()[1:].strip() for ln in cond_block.splitlines() if ln.strip().startswith("-")]

    for _ in range(max_trials):
        env = {}
        ok = True
        for line in init_lines:
            lhs, rhs = line[1:].strip().split("=", 1)
            lhs_raw = lhs.strip()
            lhs = lhs_raw.replace("$", "").strip()
            rhs = rhs.strip()
            try:
                value = eval_expr(rhs, env)
                if "$" in lhs_raw and isinstance(value, (list, tuple)):
                    value = random.choice(list(value))
            except Exception:
                ok = False
                break

            keys = [p.strip().replace("$", "") for p in lhs.split(",")]
            if len(keys) == 1:
                env[keys[0]] = value
            else:
                for key, val in zip(keys, value):
                    env[key] = val

        if not ok:
            continue

        try:
            if all(bool(eval_expr(c, env)) for c in cond_lines if c):
                return env
        except Exception:
            continue

    raise RuntimeError("Unable to satisfy template conditions.")


def render_text(text: str, env: dict):
    def repl(match):
        body = match.group(1).strip()
        if "," in body:
            left, right = body.split(",", 1)
            left = left.strip().replace("$", "")
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", left):
                if left in env:
                    val = env[left]
                    return str(fix_floats(val) if isinstance(val, (int, float)) else val)
                fallback = right.strip().replace("$", "")
                try:
                    val = eval_expr(fallback, env)
                except Exception:
                    return fallback
                return str(fix_floats(val) if isinstance(val, (int, float)) else val)

        expr = body.replace("$", "")
        try:
            val = eval_expr(expr, env)
            return str(fix_floats(val) if isinstance(val, (int, float)) else val)
        except Exception:
            return expr

    return PLACEHOLDER.sub(repl, text)


def extract_rules(question: str) -> List[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", question) if s.strip()]
    rules = []
    for s in sentences:
        low = s.lower()
        if low.startswith(("how ", "what ", "find ", "calculate ", "determine ")):
            continue
        if "?" in s:
            continue
        if any(k in low for k in ["each", "every", "per", "ratio", "%", "more", "less", "times", "total", "cost", "rate"]):
            rules.append(s)
        elif re.search(r"\d", s):
            rules.append(s)
    if not rules:
        rules = [s for s in sentences if "?" not in s][:2]
    return rules[:4]


def extract_hints(question: str, cond_rendered: List[str]) -> List[str]:
    nums = re.findall(r"-?\d+(?:\.\d+)?", question)
    uniq = []
    for n in nums:
        if n not in uniq:
            uniq.append(n)
    hints = []
    if uniq:
        hints.append("Key numeric values: " + ", ".join(uniq[:8]) + ".")
    if cond_rendered:
        hints.append("Useful constraints: " + "; ".join(cond_rendered[:3]) + ".")
    return hints


def rewrite_reasoning(reasoning: str, question: str, cond_rendered: List[str]) -> str:
    lines = reasoning.splitlines()
    # Drop existing RULES/HINTS blocks from templates, keep the rest of reasoning steps.
    kept = []
    skip = False
    for ln in lines:
        marker = ln.strip().upper()
        if marker.startswith("**RULES UNDERSTANDING:**") or marker.startswith("**HINTS UNDERSTANDING:**"):
            skip = True
            continue
        if skip and marker.startswith("**STATE OBSERVATION:**"):
            skip = False
            kept.append(ln)
            continue
        if not skip:
            kept.append(ln)

    rules = extract_rules(question)
    hints = extract_hints(question, cond_rendered)

    header = ["**RULES UNDERSTANDING:**"] + [f"- {r}" for r in rules]
    header += ["**HINTS UNDERSTANDING:**"] + ([f"- {h}" for h in hints] if hints else ["- No extra hints."])

    merged = "\n".join(header + [ln for ln in kept if ln.strip()])
    return merged


def build_assistant(answer_text: str, question_text: str, cond_rendered: List[str]) -> str:
    if "\n####" in answer_text:
        reasoning, final = answer_text.split("\n####", 1)
        reasoning = rewrite_reasoning(reasoning.strip(), question_text, cond_rendered)
        return f"<think>{reasoning}</think>\n{final.strip()}"
    reasoning = rewrite_reasoning(answer_text.strip(), question_text, cond_rendered)
    return f"<think>{reasoning}</think>"


def generate(instances: int, backtracking_ratio: float, output: str, verbose: bool):
    template_files = sorted(Path("templates").glob("*/*.json"))
    normal_pool = []
    backtracking_pool = []

    for idx, file in enumerate(template_files, start=1):
        obj = json.loads(file.read_text())
        question_raw, init_block, cond_block = parse_annotated(obj["question_annotated"])
        cond_exprs = [ln.strip()[1:].strip() for ln in cond_block.splitlines() if ln.strip().startswith("-")]

        for _ in range(instances):
            env_n = instantiate_vars(init_block, cond_block)
            q_n = render_text(question_raw, env_n)
            cond_n = [render_text(c, env_n) for c in cond_exprs]
            a_n = render_text(obj["answer_annotated"], env_n)
            normal_pool.append({
                "system": SYSTEM_PROMPT,
                "user": q_n,
                "assistant": build_assistant(a_n, q_n, cond_n),
            })

            env_b = instantiate_vars(init_block, cond_block)
            q_b = render_text(question_raw, env_b)
            cond_b = [render_text(c, env_b) for c in cond_exprs]
            a_b = render_text(obj["answer_annotated_backtrack"], env_b)
            backtracking_pool.append({
                "system": SYSTEM_PROMPT,
                "user": q_b,
                "assistant": build_assistant(a_b, q_b, cond_b),
            })

        if verbose and (idx % 25 == 0 or idx == len(template_files)):
            print(f"Processed {idx}/{len(template_files)} templates")

    total_final = len(template_files) * instances
    n_back = int(round(total_final * backtracking_ratio))
    n_norm = total_final - n_back

    random.shuffle(normal_pool)
    random.shuffle(backtracking_pool)
    final_samples = normal_pool[:n_norm] + backtracking_pool[:n_back]
    random.shuffle(final_samples)

    out_path = Path(output)
    if not out_path.is_absolute():
        out_path = Path("generated_data") / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for row in final_samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "templates_total": len(template_files),
        "instances_per_template": instances,
        "normal_pool": len(normal_pool),
        "backtracking_pool": len(backtracking_pool),
        "final_total": len(final_samples),
        "final_normal": n_norm,
        "final_backtracking": n_back,
        "backtracking_ratio": round(n_back / max(len(final_samples), 1), 4),
        "output": str(out_path),
    }
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Generate final dataset from templates.")
    parser.add_argument("--instances", type=int, default=2, help="Target final instances per template.")
    parser.add_argument("--backtracking-ratio", type=float, default=0.3, help="Final share of backtracking samples.")
    parser.add_argument("--output", default="dataset_final.jsonl", help="Output JSONL path (relative to generated_data/ if not absolute).")
    parser.add_argument("--verbose", action="store_true", help="Enable progress logging.")
    args = parser.parse_args()

    if args.instances <= 0:
        raise ValueError("--instances must be > 0")
    if not (0 <= args.backtracking_ratio <= 1):
        raise ValueError("--backtracking-ratio must be in [0,1]")

    generate(args.instances, args.backtracking_ratio, args.output, args.verbose)


if __name__ == "__main__":
    main()
