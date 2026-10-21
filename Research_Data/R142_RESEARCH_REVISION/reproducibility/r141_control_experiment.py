"""R141 prospective finite-candidate experiment.

Run one predeclared complete batch, not one candidate, per process.
Original responses, token sequences, margins, costs, choices and adapters are
emitted as independently hashed compressed artifacts. No outcome is invented.
"""
import os, sys, json, math, time, random, hashlib, gzip, base64, gc
import re
from pathlib import Path
from collections import defaultdict

COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
MODELS = {
    "qwen": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
    "smollm": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "31b70e2e869a7173562077fd711b654946d38674"),
}
GRID = [0.25, 0.50, 0.75, 1.0]
PILOT = 0.05
ALPHA = 0.10
CAP_ABS, CAP_GAIN = 0.25, 0.15
MAX_PROMPT, MAX_NEW = 2048, 512
MIN_MARGIN = 0.001

def canonical_family(gt):
    """Order-invariant multiset; preserves repeated calls and argument names."""
    parts = []
    for call in gt:
        name = next(iter(call))
        parts.append([name, sorted(call[name])])
    return json.dumps(sorted(parts, key=lambda x: json.dumps(x)), separators=(",", ":"))

def family_split(rows, targets, seed):
    import numpy as np
    groups = defaultdict(list)
    for row in rows:
        groups[row["family"]].append(row)
    names = sorted(groups)
    names = [names[j] for j in np.random.default_rng(seed).permutation(len(names))]
    out = {name: [] for name, _ in targets}
    pos = 0
    for family in names:
        while pos < len(targets)-1 and len(out[targets[pos][0]]) >= targets[pos][1]:
            pos += 1
        out[targets[pos][0]].extend(groups[family])
    seen = {}
    for role, arr in out.items():
        for row in arr:
            if row["family"] in seen:
                assert seen[row["family"]] == role
            seen[row["family"]] = role
    return out

def quantile_bound(scores, alpha=ALPHA):
    values = sorted(float(x) for x in scores)
    rank = math.ceil((len(values)+1)*(1-alpha))
    return (values[rank-1] if 1 <= rank <= len(values) else float("inf")), rank

def jsonable(x):
    if hasattr(x, "tolist"):
        return jsonable(x.tolist())
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return "Infinity" if x > 0 else "-Infinity"
    return x

def extract_calls(text):
    start = text.find("[")
    if start < 0:
        return None
    depth, quoted, escaped = 0, False, False
    for pos, char in enumerate(text[start:], start):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[start:pos+1])
                    return value if isinstance(value, list) else None
                except ValueError:
                    return None
    return None

def choose_value(values, schema):
    value = next((v for v in values if v != ""), None)
    if value is None:
        raise ValueError("No materializable allowed value")
    typ = schema.get("type")
    if typ == "dict" and isinstance(value, dict):
        return {k: choose_value(v, schema.get("properties", {}).get(k, {}))
                if isinstance(v, list) else v for k, v in value.items()}
    if typ in ("array", "tuple") and isinstance(value, list):
        item_schema = schema.get("items") or {}
        if item_schema.get("type") == "dict":
            return [choose_value([v], item_schema) if isinstance(v, dict) else v
                    for v in value]
    return value

def canonical_target(record, gt):
    functions = {f["name"]: f for f in record["function"]}
    calls = []
    for call in gt:
        name = next(iter(call))
        allowed = call[name]
        schema = functions[name]["parameters"]
        required = list(schema.get("required", []))
        required += [p for p, vs in allowed.items() if p not in required and "" not in vs]
        args = {p: choose_value(allowed[p], schema.get("properties", {}).get(p, {}))
                for p in required}
        calls.append({name: args})
    return calls

def run():
    import copy, urllib.request, platform
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, PeftModel
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    from bfcl_eval.constants.enums import Language
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING

    family = sys.argv[1] if len(sys.argv) > 1 else "qwen"
    block = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    assert family in MODELS and block in range(4)
    model_id, revision = MODELS[family]
    seed = 47100 + 100*block + (1000 if family == "smollm" else 0)
    run_id = f"{family}_batch{block+1}"
    root = Path("/tmp/r141")/run_id
    root.mkdir(parents=True, exist_ok=True)
    resume_report = []
    if len(sys.argv) > 3:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ["HF_TOKEN"])
        source_job = sys.argv[3]
        for _ in range(180):
            stage = api.inspect_job(job_id=source_job, namespace="jianlau7550").status.stage
            if stage in {"COMPLETED", "ERROR", "CANCELED", "DELETED"}:
                break
            time.sleep(10)
        recovered = {}
        def source_log_events():
            # The pinned Transformers stack requires an older hub client; use
            # the documented Jobs SSE endpoint for server-side full-log tail.
            import socket
            request = urllib.request.Request(
                "https://huggingface.co/api/jobs/jianlau7550/"+source_job+"/logs?tail=100000",
                headers={"Authorization": "Bearer "+os.environ["HF_TOKEN"],
                         "Accept": "text/event-stream"})
            try:
                with urllib.request.urlopen(request, timeout=15) as response:
                    for raw_line in response:
                        if raw_line.startswith(b"data: "):
                            event = json.loads(raw_line[6:])
                            yield event["data"]
                            if event["data"].startswith("===== Job finished"):
                                break
            except (TimeoutError, socket.timeout):
                pass
        for event in source_log_events():
            for line in event.splitlines():
                found = re.search(r'R141_ARTIFACT_(BEGIN|END) (\{.*\})', line)
                if found:
                    meta = json.loads(found[2])
                    assert meta["run"] == run_id
                    recovered.setdefault(meta["name"], {"chunks": {}})["meta"] = meta
                    continue
                found = re.search(r'R141_ARTIFACT_CHUNK (\S+) (\S+) (\d+)/(\d+) ([A-Za-z0-9+/=]+)', line)
                if found:
                    recovered.setdefault(found[2], {"chunks": {}})["chunks"][int(found[3])] = found[5]
        for name, item in recovered.items():
            meta = item.get("meta")
            if not meta or set(item["chunks"]) != set(range(meta["chunks"])):
                raise RuntimeError("Incomplete source artifact: "+name)
            raw = gzip.decompress(base64.b64decode("".join(item["chunks"][i] for i in range(meta["chunks"]))))
            assert hashlib.sha256(raw).hexdigest() == meta["sha256"], name
            dest = root/name
            dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(raw)
            resume_report.append(meta)
        assert (root/"adapters/source_adapter/adapter_model.safetensors").exists()
        # Resume the exact frozen weights after the generic interface failure.
        for name in ["source_adapter"]+[f"V{i+1}" for i in range(8)]:
            import shutil
            shutil.copytree(root/"adapters"/name, root/name, dirs_exist_ok=True)
        print("R141_RESUME_VERIFIED "+run_id+" "+str(len(resume_report)), flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "The scientific training run requires a GPU."
    # Qwen's long-schema source training overflows float16 before an endpoint
    # is produced. bfloat16 preserves exponent range; SmolLM retains its frozen
    # float16 recipe. This is a recorded family-specific numerical setting.
    dtype = torch.bfloat16 if family == "qwen" else torch.float16
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(4)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    events, artifact_index = [], []
    begin = time.time()

    def sync():
        torch.cuda.synchronize()

    def emit(name, value, binary=False):
        raw = bytes(value) if binary else json.dumps(jsonable(value), sort_keys=True,
                                                      ensure_ascii=False).encode()
        fp = root/name
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_bytes(raw)
        packed = gzip.compress(raw, compresslevel=6)
        encoded = base64.b64encode(packed).decode()
        digest = hashlib.sha256(raw).hexdigest()
        chunks = [encoded[i:i+12000] for i in range(0, len(encoded), 12000)]
        meta = {"run": run_id, "name": name, "bytes": len(raw), "sha256": digest,
                "chunks": len(chunks)}
        print("R141_ARTIFACT_BEGIN "+json.dumps(meta), flush=True)
        for i, chunk in enumerate(chunks):
            print(f"R141_ARTIFACT_CHUNK {run_id} {name} {i:05d}/{len(chunks):05d} {chunk}",
                  flush=True)
            if len(chunks) > 10:
                time.sleep(0.025)
        print("R141_ARTIFACT_END "+json.dumps(meta), flush=True)
        artifact_index.append(meta)

    alias = "R141-Checker-Alias"
    MODEL_CONFIG_MAPPING[alias] = copy.copy(MODEL_CONFIG_MAPPING["gorilla-openfunctions-v2"])
    MODEL_CONFIG_MAPPING[alias].underscore_to_dot = False
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    eos = tok.eos_token_id
    source_hashes, exclusions = {}, []
    base_url = (f"https://raw.githubusercontent.com/ShishirPatil/gorilla/{COMMIT}/"
                "berkeley-function-call-leaderboard/bfcl_eval/data/")

    def get_jsonl(name):
        raw = urllib.request.urlopen(base_url+name, timeout=90).read()
        source_hashes[name] = hashlib.sha256(raw).hexdigest()
        return [json.loads(line) for line in raw.decode().splitlines() if line.strip()]

    def make_rows(category):
        records = get_jsonl("BFCL_v4_"+category+".json")
        truth = {x["id"]: x["ground_truth"]
                 for x in get_jsonl("possible_answer/BFCL_v4_"+category+".json")}
        rows = []
        for r in records:
            gt = truth[r["id"]]
            calls = canonical_target(r, gt)
            check = ast_checker(r["function"], calls, gt, Language.PYTHON, category, alias)
            assert check["valid"], (r["id"], check)
            text = "\n".join(m["content"] for turn in r["question"] for m in turn
                             if isinstance(m.get("content"), str))
            messages = [
                {"role": "system", "content": 'Return only a JSON list of function-call objects. '
                 'Each object is {"function_name":{"argument":value}}. Include all calls '
                 'needed to satisfy the user. Use the complete tool schemas below.'},
                {"role": "user", "content": "Request:\n"+text+"\nTools:\n"+
                 json.dumps(r["function"], ensure_ascii=False, separators=(",", ":"))}
            ]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_ids = tok.encode(prompt, add_special_tokens=False)
            target = json.dumps(calls, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            target_ids = tok.encode(target, add_special_tokens=False)+[eos]
            if len(prompt_ids) > MAX_PROMPT or len(target_ids) > MAX_NEW:
                exclusions.append({"id": r["id"], "prompt_tokens": len(prompt_ids),
                                   "target_tokens": len(target_ids)})
                continue
            rows.append({"id": r["id"], "family": canonical_family(gt), "record": r,
                         "gt": gt, "category": category, "prompt": prompt,
                         "prompt_ids": prompt_ids, "target_ids": target_ids})
        return rows

    old, new = make_rows("simple_python"), make_rows("parallel")
    old_s = family_split(old, [("source_train", 160), ("calibration", 80),
                              ("selection", 60), ("fresh_old", 10**9)], seed+7)
    new_s = family_split(new, [("update_train", 100), ("cap_validation", 40),
                              ("cap_audit", 10**9)], seed+11)
    split = {**old_s, **new_s}
    assert all(split[k] for k in split), {k: len(v) for k, v in split.items()}
    manifest = {k: [{"id": x["id"], "family": x["family"]} for x in v]
                for k, v in split.items()}
    config = {"run": run_id, "seed": seed, "model": model_id, "revision": revision,
              "bfcl_commit": COMMIT, "source_hashes": source_hashes, "grid": GRID,
              "pilot": PILOT, "alpha": ALPHA, "cap_abs": CAP_ABS, "cap_gain": CAP_GAIN,
              "max_prompt_tokens": MAX_PROMPT, "max_new_tokens": MAX_NEW,
              "exclusions": exclusions, "split": manifest,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "torch": torch.__version__, "python": platform.python_version(),
              "dtype": str(dtype),
              "gpu": torch.cuda.get_device_name(0), "job_id": os.environ.get("JOB_ID")}
    emit("configuration.json", config)
    if resume_report:
        emit("resume_provenance.json", {"source_job": source_job,
            "artifacts": resume_report, "scientific_rule_changed": False,
            "fix": "Restrict LoRA scaling lookup to dictionaries; reuse exact prior artifacts"})
    emit("benchmark_rows.json", {k: v for k, v in split.items()})
    print("R141_STAGE "+run_id+" configuration_frozen", flush=True)

    def load_base(path, pinned=False):
        return AutoModelForCausalLM.from_pretrained(
            path, revision=revision if pinned else None, torch_dtype=dtype,
            attn_implementation="eager").to(device)

    def train(path, rows, training_seed, epochs, destination, pinned=False):
        if resume_report and (Path(destination)/"adapter_model.safetensors").exists():
            previous = json.loads((root/"training"/(Path(destination).name+".json")).read_text())
            events.append({"kind": "training", "name": Path(destination).name,
                "seconds": previous["seconds"], "optimizer_steps": previous["steps"],
                "records": len(rows), "epochs": epochs, "reused_source_job": source_job})
            return
        random.seed(training_seed); np.random.seed(training_seed)
        torch.manual_seed(training_seed); torch.cuda.manual_seed_all(training_seed)
        m = load_base(path, pinned)
        m.config.use_cache = False
        m = get_peft_model(m, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.,
              target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM"))
        m.train()
        optimizer = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-4)
        rng = random.Random(training_seed)
        losses, steps = [], 0
        sync(); started = time.time()
        for ep in range(epochs):
            indices = list(range(len(rows))); rng.shuffle(indices)
            optimizer.zero_grad(set_to_none=True)
            for j, i in enumerate(indices):
                row = rows[i]
                ids = row["prompt_ids"]+row["target_ids"]
                labels = [-100]*len(row["prompt_ids"])+row["target_ids"]
                output = m(input_ids=torch.tensor([ids], device=device),
                           labels=torch.tensor([labels], device=device), use_cache=False)
                loss = output.loss
                assert bool(torch.isfinite(loss)), "Nonfinite training loss"
                # The final partial block is normalized by its actual size.
                block_start = (j//8)*8
                accum = min(8, len(indices)-block_start)
                (loss/accum).backward()
                losses.append(float(loss.detach()))
                if (j+1)%8 == 0 or j+1 == len(indices):
                    torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True); steps += 1
        sync(); elapsed = time.time()-started
        m.save_pretrained(destination)
        events.append({"kind": "training", "name": Path(destination).name,
                       "seconds": elapsed, "optimizer_steps": steps,
                       "records": len(rows), "epochs": epochs})
        emit("training/"+Path(destination).name+".json",
             {"seed": training_seed, "losses": losses, "seconds": elapsed, "steps": steps})
        for fp in sorted(Path(destination).glob("*")):
            if fp.is_file():
                emit("adapters/"+Path(destination).name+"/"+fp.name, fp.read_bytes(), True)
        del m, optimizer
        gc.collect(); torch.cuda.empty_cache()

    def load_adapter(path):
        b = load_base(str(root/"M0"))
        m = PeftModel.from_pretrained(b, str(path)).to(device)
        m.eval()
        return m

    def scales(m):
        return [(mod, float(mod.scaling["default"])) for mod in m.modules()
                if isinstance(getattr(mod, "scaling", None), dict) and "default" in mod.scaling]

    def strength(sc, s):
        for mod, value in sc:
            mod.scaling["default"] = value*s

    @torch.no_grad()
    def evaluate(m, rows, tag):
        m.eval()
        cached = root/"rows"/(tag+".json")
        if resume_report and cached.exists():
            details = json.loads(cached.read_text())
            assert [x["id"] for x in details] == [x["id"] for x in rows]
            events.append({"kind": "endpoint", "tag": tag, "record_calls": len(rows),
                "seconds": None, "reused_source_job": source_job,
                "prompt_tokens": sum(x["prompt_tokens"] for x in details),
                "generated_tokens": sum(len(x["tokens"]) for x in details),
                "timing_status": "Shared preparation time unavailable; observation counts exact"})
            return details
        details = []
        sync(); started = time.time()
        generated_positions, prompt_positions, forward_batches = 0, 0, 0
        for start in range(0, len(rows), 4):
            batch = rows[start:start+4]
            width = max(len(x["prompt_ids"]) for x in batch)
            ids = [[tok.pad_token_id]*(width-len(x["prompt_ids"]))+x["prompt_ids"] for x in batch]
            mask = [[0]*(width-len(x["prompt_ids"]))+[1]*len(x["prompt_ids"]) for x in batch]
            out = m.generate(input_ids=torch.tensor(ids, device=device),
                attention_mask=torch.tensor(mask, device=device), do_sample=False,
                max_new_tokens=MAX_NEW, min_new_tokens=0, repetition_penalty=1.,
                no_repeat_ngram_size=0, renormalize_logits=False,
                pad_token_id=tok.pad_token_id, eos_token_id=eos, use_cache=True)
            forward_batches += out.shape[1]-width
            for i, row in enumerate(batch):
                tokens = out[i, width:].tolist()
                if eos in tokens:
                    tokens = tokens[:tokens.index(eos)+1]
                text = tok.decode(tokens, skip_special_tokens=True)
                parsed = extract_calls(text)
                try:
                    checked = ast_checker(row["record"]["function"], parsed, row["gt"],
                            Language.PYTHON, row["category"], alias) if parsed is not None else {}
                    valid = bool(checked.get("valid", False))
                    error = None if valid else checked.get("error_type", "json_parse")
                except Exception as exc:
                    valid, error = False, "checker_exception:"+type(exc).__name__
                details.append({"id": row["id"], "family": row["family"], "valid": valid,
                                "error": error, "text": text, "tokens": tokens,
                                "prompt_tokens": len(row["prompt_ids"])})
                generated_positions += len(tokens); prompt_positions += len(row["prompt_ids"])
        sync(); elapsed = time.time()-started
        events.append({"kind": "endpoint", "tag": tag, "record_calls": len(rows),
                       "batch_forward_calls": forward_batches, "seconds": elapsed,
                       "prompt_tokens": prompt_positions, "generated_tokens": generated_positions})
        emit("rows/"+tag+".json", details)
        return details

    @torch.no_grad()
    def margins(m, rows, references, tag):
        m.eval(); ans = []
        cached = root/"margins"/(tag+".json")
        if resume_report and cached.exists():
            previous = json.loads(cached.read_text())
            assert previous["ids"] == [x["id"] for x in rows]
            events.append({"kind": "margin", "tag": tag, "record_calls": len(rows),
                "seconds": None, "reused_source_job": source_job,
                "sequence_tokens": sum(len(x["prompt_ids"])+len(references[x["id"]]["tokens"]) for x in rows),
                "timing_status": "Shared preparation time unavailable; observation counts exact"})
            return np.array(previous["values"])
        sync(); started = time.time()
        positions = 0
        for row in rows:
            target = references[row["id"]]["tokens"]
            ids = row["prompt_ids"]+target
            out = m(input_ids=torch.tensor([ids], device=device), use_cache=False)
            z = out.logits[0, len(row["prompt_ids"])-1:len(ids)-1].float()
            y = torch.tensor(target, device=device)
            good = z.gather(1, y[:, None]).squeeze(1)
            top, idx = z.topk(2, dim=1)
            rival = torch.where(idx[:, 0] == y, top[:, 1], top[:, 0])
            ans.append(float((good-rival).min()) if len(target) else -1e9)
            positions += len(ids)
        sync(); elapsed = time.time()-started
        events.append({"kind": "margin", "tag": tag, "record_calls": len(rows),
                       "batch_forward_calls": len(rows), "seconds": elapsed,
                       "sequence_tokens": positions})
        emit("margins/"+tag+".json", {"ids": [x["id"] for x in rows], "values": ans})
        return np.array(ans)

    def group_values(rows, values, conjunction=False):
        groups = defaultdict(list)
        for row, value in zip(rows, values):
            groups[row["family"]].append(value)
        return {f: bool(all(v)) if conjunction else float(min(v)) for f, v in groups.items()}

    def safe_fraction(rows, values):
        grouped = group_values(rows, values)
        return sum(v > MIN_MARGIN for v in grouped.values())/len(grouped) if grouped else 0.

    def accuracy(details):
        return float(np.mean([x["valid"] for x in details])) if details else None

    # The incumbent is trained independently in every complete batch.
    source_path = root/"source_adapter"
    train(model_id, split["source_train"], seed, 3, str(source_path), pinned=True)
    b = load_base(model_id, pinned=True)
    m0 = PeftModel.from_pretrained(b, str(source_path)).merge_and_unload()
    m0.save_pretrained(root/"M0"); tok.save_pretrained(root/"M0"); m0.eval()
    base_details = {role: evaluate(m0, split[role], "M0_"+role)
                    for role in ["calibration", "selection", "fresh_old",
                                 "cap_validation", "cap_audit"]}
    refs = {x["id"]: x for arr in base_details.values() for x in arr}
    protected = {role: [x for x in split[role] if refs[x["id"]]["valid"]]
                 for role in ["calibration", "selection", "fresh_old"]}
    all_select_fams = sorted({x["family"] for x in protected["selection"]})
    random.Random(seed+29).shuffle(all_select_fams)
    selected_fams = set(all_select_fams[:20])
    sel = [x for x in protected["selection"] if x["family"] in selected_fams]
    cal, fresh = protected["calibration"], protected["fresh_old"]
    gcal = margins(m0, cal, refs, "M0_calibration")
    gsel = margins(m0, sel, refs, "M0_selection")
    # Fresh reference margins are audit measurements and never selection inputs.
    gfresh = margins(m0, fresh, refs, "M0_fresh_audit")
    base_cv, base_ca = accuracy(base_details["cap_validation"]), accuracy(base_details["cap_audit"])
    del b, m0; gc.collect(); torch.cuda.empty_cache()

    candidates, cap_cache = {}, {}
    replay = split["source_train"][:40]
    for i in range(8):
        name = f"V{i+1}"
        path = root/name
        training_seed = seed+40+i
        train(str(root/"M0"), split["update_train"]+replay, training_seed, 1, str(path))
        m = load_adapter(path)
        cv = evaluate(m, split["cap_validation"], name+"_capval_1.00")
        cap_cache[(name, 1.0)] = cv
        norm2 = 0.
        for mod in m.modules():
            if hasattr(mod, "lora_A") and "default" in mod.lora_A:
                a = mod.lora_A["default"].weight.detach().float()
                b = mod.lora_B["default"].weight.detach().float()
                norm2 += float(torch.trace((b.T@b)@(a@a.T)))*float(mod.scaling["default"])**2
        candidates[name] = {"seed": training_seed, "full_cap_validation": accuracy(cv),
                            "norm": math.sqrt(max(norm2, 0.))}
        del m; gc.collect(); torch.cuda.empty_cache()
        print("R141_STAGE "+run_id+" "+name+"_trained", flush=True)
    median = float(np.median([v["norm"] for v in candidates.values()]))
    shortlist = sorted(candidates, key=lambda n: (-candidates[n]["full_cap_validation"],
                       abs(candidates[n]["norm"]-median), n))[:4]
    emit("shortlist_frozen.json", {"candidates": candidates, "shortlist": shortlist})

    cap_grid = {}
    for name in shortlist:
        m = load_adapter(root/name); sc = scales(m)
        for s in GRID:
            if (name, s) not in cap_cache:
                strength(sc, s)
                cap_cache[name, s] = evaluate(m, split["cap_validation"], f"{name}_capval_{s:.2f}")
            cap_grid[f"{name}@{s:.2f}"] = accuracy(cap_cache[name, s])
        del m; gc.collect(); torch.cuda.empty_cache()
    admissible = [(name, s) for name in shortlist for s in GRID
                  if cap_grid[f"{name}@{s:.2f}"] >= CAP_ABS
                  and cap_grid[f"{name}@{s:.2f}"]-base_cv >= CAP_GAIN-1e-12]
    emit("capability_gate.json", {"incumbent": base_cv, "grid": cap_grid,
                                 "admissible": admissible})
    dcal, dsel, actual_cal = {}, {}, {}
    for name in shortlist:
        m = load_adapter(root/name); sc = scales(m); strength(sc, PILOT)
        dcal[name] = (margins(m, cal, refs, name+"_cal_pilot")-gcal)/PILOT
        dsel[name] = (margins(m, sel, refs, name+"_sel_pilot")-gsel)/PILOT
        for s in GRID:
            strength(sc, s)
            actual_cal[name, s] = margins(m, cal, refs, f"{name}_cal_{s:.2f}")
        del m; gc.collect(); torch.cuda.empty_cache()
    residual = np.zeros(len(cal))
    for name in shortlist:
        for s in GRID:
            residual = np.maximum(residual, np.abs(actual_cal[name, s]-gcal-s*dcal[name]))
    residual_families = defaultdict(list)
    for row, value in zip(cal, residual):
        residual_families[row["family"]].append(float(value))
    family_scores = {f: max(v) for f, v in residual_families.items()}
    q, rank = quantile_bound(family_scores.values())
    emit("calibration.json", {"scores": family_scores, "q": q, "rank": rank,
                              "n_family": len(family_scores), "n_record": len(cal)})

    def tie(pair):
        name, s = pair
        return (-cap_grid[f"{name}@{s:.2f}"], s*candidates[name]["norm"], name, s)

    choices = {m: None for m in ["full_update", "simple_scaling", "minimum_norm",
                                "local_prediction", "calibrated_prediction", "direct_validation"]}
    prediction_scores = {}
    if admissible:
        full = [p for p in admissible if p[1] == 1.]
        choices["full_update"] = min(full, key=tie) if full else None
        top = min(shortlist, key=lambda n: (-candidates[n]["full_cap_validation"], n))
        own = [p for p in admissible if p[0] == top]
        choices["simple_scaling"] = min(own, key=lambda p: (p[1], tie(p))) if own else None
        choices["minimum_norm"] = min(admissible, key=lambda p: (p[1]*candidates[p[0]]["norm"], tie(p)))
        for name, s in admissible:
            p = gsel+s*dsel[name]
            prediction_scores[f"{name}@{s:.2f}"] = {
                "unadjusted": safe_fraction(sel, p),
                "calibrated": safe_fraction(sel, p-q) if math.isfinite(q) else 0.}
        choices["local_prediction"] = min(admissible, key=lambda p:
            (-prediction_scores[f"{p[0]}@{p[1]:.2f}"]["unadjusted"], tie(p))) if sel else None
        eligible = [p for p in admissible if math.isfinite(q) and
                    prediction_scores[f"{p[0]}@{p[1]:.2f}"]["calibrated"] > 0]
        choices["calibrated_prediction"] = min(eligible, key=lambda p:
            (-prediction_scores[f"{p[0]}@{p[1]:.2f}"]["calibrated"], tie(p))) if eligible else None

    # Equal cap on candidate-record observations. Endpoint and margin calls are
    # both counted; their different GPU-time and token costs are also reported.
    # Incumbent preparation and capability screening are shared costs.
    budget = len(shortlist)*(len(GRID)+1)*len(cal)+len(shortlist)*len(sel)
    development = cal+sel
    fams = sorted({x["family"] for x in development})
    random.Random(seed+83).shuffle(fams)
    direct_rows, spent = [], 0
    for f in fams:
        rows = [x for x in development if x["family"] == f]
        need = len(rows)*len(admissible)
        if spent+need > budget:
            break
        direct_rows.extend(rows); spent += need
    direct_scores = {}
    if admissible and direct_rows:
        for name in shortlist:
            own = [p for p in admissible if p[0] == name]
            if not own:
                continue
            m = load_adapter(root/name); sc = scales(m)
            for _, s in own:
                strength(sc, s)
                det = evaluate(m, direct_rows, f"{name}_direct_{s:.2f}")
                good = group_values(direct_rows, [x["valid"] for x in det], True)
                direct_scores[f"{name}@{s:.2f}"] = sum(good.values())/len(good)
            del m; gc.collect(); torch.cuda.empty_cache()
        choices["direct_validation"] = min(admissible, key=lambda p:
            (-direct_scores[f"{p[0]}@{p[1]:.2f}"], tie(p)))
    frozen = {"choices": choices, "prediction_scores": prediction_scores,
              "direct_scores": direct_scores, "candidate_record_budget": budget,
              "direct_candidate_record_calls": spent,
              "direct_family_n": len({x["family"] for x in direct_rows}),
              "selection_family_n": len(selected_fams),
              "fresh_outcomes_opened": False}
    emit("choices_frozen.json", frozen)
    print("R141_STAGE "+run_id+" choices_frozen_before_fresh_candidates", flush=True)

    # Independent audit of the exact models and strengths selected above.
    fresh_results, certificates = {}, {}
    for name in shortlist:
        m = load_adapter(root/name); sc = scales(m); strength(sc, PILOT)
        df = (margins(m, fresh, refs, name+"_fresh_pilot_AUDIT")-gfresh)/PILOT
        for s in GRID:
            strength(sc, s)
            det = evaluate(m, fresh, f"{name}_fresh_{s:.2f}_AUDIT")
            cap = evaluate(m, split["cap_audit"], f"{name}_cap_audit_{s:.2f}")
            actual = margins(m, fresh, refs, f"{name}_fresh_margin_{s:.2f}_AUDIT")
            good = group_values(fresh, [x["valid"] for x in det], True)
            pred = group_values(fresh, gfresh+s*df-q if math.isfinite(q)
                                else np.full(len(fresh), -np.inf))
            declared = {f: v > MIN_MARGIN for f, v in pred.items()}
            residual_groups = defaultdict(list)
            for row, v in zip(fresh, np.abs(actual-gfresh-s*df)):
                residual_groups[row["family"]].append(float(v))
            key = f"{name}@{s:.2f}"
            fresh_results[key] = {"old_n": len(det), "old_loss_n": sum(not x["valid"] for x in det),
                "old_family_n": len(good), "old_family_loss_n": sum(not x for x in good.values()),
                "family_pass": good, "new_n": len(cap), "new_success_n": sum(x["valid"] for x in cap),
                "new_accuracy": accuracy(cap), "new_gain": accuracy(cap)-base_ca,
                "new_floor_pass": accuracy(cap) >= CAP_ABS and accuracy(cap)-base_ca >= CAP_GAIN-1e-12}
            certificates[key] = {"declared_family_n": sum(declared.values()),
                "false_declared_family_n": sum(declared[f] and not good[f] for f in good),
                "family_n": len(good), "declarations": declared,
                "envelope_miss_family_n": sum(max(v) > q for v in residual_groups.values())}
        del m; gc.collect(); torch.cuda.empty_cache()
    oracle_pool = [k for k, v in fresh_results.items() if v["new_floor_pass"] and v["old_family_n"]]
    oracle = min(oracle_pool, key=lambda k: (
        fresh_results[k]["old_family_loss_n"]/fresh_results[k]["old_family_n"],
        -fresh_results[k]["new_accuracy"], k)) if oracle_pool else None
    method_results = {}
    for method, chosen in choices.items():
        key = f"{chosen[0]}@{chosen[1]:.2f}" if chosen else None
        method_results[method] = {"choice": chosen, "abstained": chosen is None,
                                 "outcome": fresh_results.get(key)}
    method_results["full_retest_oracle"] = {"choice": oracle, "audit_only": True,
                                          "outcome": fresh_results.get(oracle)}
    summary = {"configuration": config, "incumbent_new_accuracy": base_ca,
               "protected_calibration_n": len(cal), "protected_fresh_n": len(fresh),
               "candidates": candidates, "shortlist": shortlist, "q": q,
               "n_calibration_families": len(family_scores), "rank": rank,
               "cap_validation": cap_grid, "admissible": admissible,
               "frozen": frozen, "methods": method_results,
               "fresh_grid": fresh_results, "certificates": certificates,
               "elapsed_seconds": time.time()-begin}
    emit("cost_events.json", events)
    emit("RESULT.json", summary)
    emit("artifact_index.json", artifact_index)
    # Repeat the complete, compact scientific record at completion so it is
    # recoverable independently of voluminous intermediate checkpoint logs.
    # Checkpoint payloads have their own hashes above; they are not silently
    # substituted for row-level results in this archive.
    scientific = {}
    for fp in sorted(root.rglob("*.json")):
        rel = fp.relative_to(root)
        if rel.parts[0] not in {"M0", "source_adapter"} and not rel.parts[0].startswith("V"):
            scientific[str(rel)] = json.loads(fp.read_text())
    emit("scientific_record.json", scientific)
    print("R141_COMPLETE "+run_id+" "+json.dumps(jsonable({
        "methods": method_results, "q": q, "elapsed_seconds": time.time()-begin})), flush=True)

if __name__ == "__main__":
    run()
