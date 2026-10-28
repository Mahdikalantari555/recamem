
import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ponytail: unified headless CLI - tmt-droid entrypoint

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

def cmd_chat(args):
    """Interactive terminal chat loop."""
    from src.model.droid_manager import DroidManager
    mgr = DroidManager(base_dir=args.base_dir)
    droid = mgr.get_droid(args.name)
    
    mode = getattr(args, 'mode', 'auto')
    print(f"Chatting with {args.name} (mode={mode}). Type 'exit' or Ctrl+C to quit.", file=sys.stderr)
    
    if args.query:
        # Single query mode
        query = args.query
        if query.strip().lower().startswith('/sleep'):
            print(droid._format_sleep_report(droid.sleep()))
            return
        # Handle /fast /slow overrides
        if query.startswith('/fast '):
            mode = 'fast'
            query = query[5:].strip()
        elif query.startswith('/slow '):
            mode = 'slow'
            query = query[6:].strip()
        
        if mode == 'fast' or (mode == 'auto' and hasattr(droid, 'decide_path')):
            try:
                if mode == 'fast':
                    hits = droid.recall(query, top_k=4)
                    if hits:
                        # Synthesize without LLM
                        parts = []
                        for h in hits:
                            t = h["text"].strip()
                            if t not in parts:
                                parts.append(t)
                        print(" ".join(parts))
                    else:
                        print(droid.chat(query))
                elif hasattr(droid, 'decide_path'):
                    decision = droid.decide_path(query)
                    if decision.get('action') == 'fast_recall' or decision.get('path') == 'fast':
                        print(f"[fast path, conf={decision.get('confidence',0):.2f}]")
                        hits = droid.recall(query, top_k=4)
                        if hits:
                            print(" ".join([h["text"] for h in hits]))
                        else:
                            print(droid.chat(query))
                    else:
                        print(f"[slow path, conf={decision.get('confidence',0):.2f}]")
                        if hasattr(droid, 'generate_with_memory'):
                            result = droid.generate_with_memory(query)
                            if isinstance(result, dict):
                                print(result.get('text', str(result)))
                            else:
                                print(result)
                        else:
                            print(droid.chat(query))
                else:
                    print(droid.chat(query))
            except Exception as e:
                print(droid.chat(query))
                print(f"[error in mode handling: {e}]", file=sys.stderr)
        else:
            # Check for explicit mode
            if mode == 'slow' and hasattr(droid, 'generate_with_memory'):
                result = droid.generate_with_memory(query)
                if isinstance(result, dict):
                    print(result.get('text', str(result)))
                else:
                    print(result)
            else:
                print(droid.chat(query))
        return
    
    # Interactive loop
    try:
        while True:
            try:
                user_input = input(f"{args.name}> ").strip()
            except EOFError:
                break
            if not user_input:
                continue
            if user_input.lower() in ('exit', 'quit', 'q'):
                break
            if user_input.strip().lower().startswith('/sleep'):
                print(droid._format_sleep_report(droid.sleep()))
                continue
            
            # Handle /fast /slow
            cur_mode = mode
            query = user_input
            if user_input.startswith('/fast '):
                cur_mode = 'fast'
                query = user_input[5:].strip()
            elif user_input.startswith('/slow '):
                cur_mode = 'slow'
                query = user_input[6:].strip()
            
            try:
                if cur_mode == 'fast':
                    hits = droid.recall(query, top_k=4)
                    if hits:
                        parts = []
                        for h in hits:
                            t = h["text"].strip()
                            if not any(t in p or p in t for p in parts):
                                parts.append(t)
                        print(" ".join(parts))
                    else:
                        print(droid.chat(query))
                elif cur_mode == 'slow' and hasattr(droid, 'generate_with_memory'):
                    result = droid.generate_with_memory(query)
                    if isinstance(result, dict):
                        print(result.get('text', str(result)))
                    else:
                        print(result)
                else:
                    # auto
                    if hasattr(droid, 'decide_path'):
                        decision = droid.decide_path(query)
                        if decision.get('action') in ('fast_recall', 'recall') or decision.get('path') == 'fast':
                            hits = droid.recall(query, top_k=4)
                            if hits:
                                print(" ".join([h["text"] for h in hits]))
                            else:
                                print(droid.chat(query))
                        else:
                            if hasattr(droid, 'generate_with_memory'):
                                result = droid.generate_with_memory(query)
                                if isinstance(result, dict):
                                    print(result.get('text', str(result)))
                                else:
                                    print(result)
                            else:
                                print(droid.chat(query))
                    else:
                        print(droid.chat(query))
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
    except KeyboardInterrupt:
        print("\nExiting.", file=sys.stderr)

def cmd_sleep(args):
    """Run one bounded sleep cycle through DroidEngine.sleep()."""
    from src.model.droid_manager import DroidManager
    mgr = DroidManager(base_dir=args.base_dir)
    droid = mgr.get_droid(args.name)
    report = droid.sleep()
    if getattr(args, "json", False):
        print(json.dumps(report, indent=2, default=str))
    else:
        print(droid._format_sleep_report(report))
    return 0


def cmd_teach(args):
    """Teach from stdin markdown ingestion."""
    from src.model.droid_manager import DroidManager
    mgr = DroidManager(base_dir=args.base_dir)
    droid = mgr.get_droid(args.name)
    
    # Read from stdin or file
    if args.file:
        text = Path(args.file).read_text(encoding='utf-8')
    else:
        # Read stdin
        if sys.stdin.isatty():
            print("Reading from stdin (Ctrl+D to finish)...", file=sys.stderr)
        text = sys.stdin.read()
    
    if not text.strip():
        print("No input text provided.", file=sys.stderr)
        return 1
    
    result = droid.teach(text, source=args.source or "cli")
    mgr.save_droid(args.name)
    
    print(f"Absorbed {result.get('propositions',0)} facts (episodic: {result.get('episodic_absorbed', result.get('propositions',0))})")
    print(f"Memory norm: {result.get('memory_norm',0):.2f}, avg surprise: {result.get('avg_surprise',0):.3f}")
    print(f"Total knowledge: {result.get('total_knowledge',0)}")
    return 0

def cmd_decide(args):
    """Score options using System-1 decision head."""
    from src.model.decision_head import DecisionHead
    from src.model.droid_manager import DroidManager
    import torch
    
    # Try to get embedding from droid anchor
    mgr = DroidManager(base_dir=args.base_dir)
    try:
        droid = mgr.get_droid(args.name)
        q_emb = droid.anchor.embed(args.query)
        if q_emb.dim() > 1:
            q_emb = q_emb.squeeze(0)
    except Exception as e:
        print(f"Warning: could not get anchor embedding, using random: {e}", file=sys.stderr)
        torch.manual_seed(hash(args.query) % 10000)
        q_emb = torch.randn(384)
    
    # If options provided, create decision head with options as prototypes
    if args.options:
        options = [o.strip() for o in args.options.split(',')]
        # Try to embed options
        option_embs = {}
        try:
            mgr = DroidManager(base_dir=args.base_dir)
            droid = mgr.get_droid(args.name)
            for opt in options:
                try:
                    emb = droid.anchor.embed(opt)
                    if emb.dim() > 1:
                        emb = emb.squeeze(0)
                    import torch.nn.functional as F
                    import torch
                    emb = emb / (torch.linalg.vector_norm(emb).clamp(min=1e-8))
                    option_embs[opt] = emb.tolist()
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
        
        if option_embs:
            dh = DecisionHead(prototypes=option_embs)
        else:
            dh = DecisionHead(prototype_names=options)
    else:
        dh = DecisionHead()
    
    decision = dh.decide(q_emb)
    
    print(f"Query: {args.query}")
    print(f"Selected action: {decision['action']}")
    print(f"Confidence: {decision['confidence']:.3f}")
    print(f"Latency: {decision['latency_ms']:.2f}ms")
    print(f"Probabilities:")
    for name, prob, sim in zip(decision['prototype_names'], decision['probabilities'], decision['similarities']):
        print(f"  {name}: prob={prob:.3f} sim={sim:.3f}")
    
    if args.json:
        print(json.dumps(decision, indent=2))
    
    return 0

def cmd_serve(args):
    """Serve MCP or REST."""
    if args.mcp:
        from src.mcp_server import MCPServer
        server = MCPServer(base_dir=args.base_dir)
        server.run_stdio()
    elif args.rest:
        # Lightweight REST microdaemon
        from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
        import json as json_lib
        
        base_dir = args.base_dir
        
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/health':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json_lib.dumps({"status": "ok"}).encode())
                elif self.path == '/droids':
                    from pathlib import Path
                    base = Path(base_dir)
                    droids = [d.name for d in base.iterdir() if d.is_dir()] if base.exists() else []
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json_lib.dumps({"droids": droids}).encode())
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def do_POST(self):
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)
                try:
                    data = json_lib.loads(body.decode('utf-8'))
                except Exception:  # noqa: BLE001
                    self.send_response(400)
                    self.end_headers()
                    return
                
                path = self.path
                try:
                    from src.model.droid_manager import DroidManager
                    mgr = DroidManager(base_dir=base_dir)
                    
                    if path == '/recall':
                        query = data.get('query', '')
                        droid_name = data.get('droid_name', 'droid-alpha')
                        top_k = data.get('top_k', 3)
                        droid = mgr.get_droid(droid_name)
                        hits = droid.recall(query, top_k=top_k)
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json_lib.dumps({"hits": hits}).encode())
                    elif path == '/teach':
                        text = data.get('text', '')
                        droid_name = data.get('droid_name', 'droid-alpha')
                        droid = mgr.get_droid(droid_name)
                        result = droid.teach(text)
                        mgr.save_droid(droid_name)
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json_lib.dumps(result).encode())
                    elif path == '/chat':
                        query = data.get('query', '')
                        droid_name = data.get('droid_name', 'droid-alpha')
                        droid = mgr.get_droid(droid_name)
                        reply = droid.chat(query)
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json_lib.dumps({"reply": reply}).encode())
                    else:
                        self.send_response(404)
                        self.end_headers()
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json_lib.dumps({"error": str(e)}).encode())
            
            def log_message(self, format, *args):
                # Log to stderr
                sys.stderr.write("%s - - [%s] %s\n" %
                                 (self.client_address[0],
                                  self.log_date_time_string(),
                                  format%args))
        
        port = args.port or 8000
        server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        print(f"REST microdaemon serving on http://127.0.0.1:{port}", file=sys.stderr)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.", file=sys.stderr)
    else:
        print("Must specify --mcp or --rest", file=sys.stderr)
        return 1

def cmd_package(args):
    """Package export/import."""
    from src.model.droid_package import DroidPackage
    
    if args.package_cmd == 'export':
        profile_dir = Path(args.base_dir) / args.name
        if not profile_dir.exists():
            print(f"Droid profile not found: {profile_dir}", file=sys.stderr)
            return 1
        result = DroidPackage.export_droid(str(profile_dir), args.out)
        print(f"Exported {args.name} to {args.out}")
        print(f"Files: {result['files']}")
    elif args.package_cmd == 'import':
        target_dir = Path(args.base_dir) / args.name
        result = DroidPackage.import_droid(args.archive, str(target_dir))
        print(f"Imported {args.archive} to {target_dir}")
        print(f"Verified: {result['verified']}")
    elif args.package_cmd == 'list':
        from src.model.droid_package import DroidPackage
        result = DroidPackage.list_contents(args.archive)
        print(f"Archive: {args.archive}")
        print(f"Files: {result['files']}")
        print(json.dumps(result['manifest'], indent=2))
    else:
        print(f"Unknown package subcommand: {args.package_cmd}", file=sys.stderr)
        return 1
    return 0

def _eval_error(message: str) -> int:
    """Report an invalid-input error on stderr and return exit code 2."""
    print(f"error: {message}", file=sys.stderr)
    return 2


def _ranking_metrics(ranked_ids: List[int], relevant_ids: set, top_k: int) -> tuple:
    """(precision@k, recall@k, reciprocal rank) for one query."""
    seen = list(ranked_ids[:top_k])
    tp = len(set(seen) & relevant_ids)
    precision = tp / top_k
    recall = tp / len(relevant_ids) if relevant_ids else 0.0
    rr = 0.0
    for rank, fid in enumerate(seen, start=1):
        if fid in relevant_ids:
            rr = 1.0 / rank
            break
    return precision, recall, rr


def _load_labeled_queries(path: str) -> List[Dict]:
    """Read a JSONL query file: {"query": "...", "fact_ids": [1, 2]}."""
    queries = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
            if not isinstance(entry, dict) or not isinstance(entry.get("query"), str):
                raise ValueError(f"{path}:{lineno}: expected an object with a string 'query'")
            fact_ids = entry.get("fact_ids")
            if not isinstance(fact_ids, list) or not fact_ids:
                raise ValueError(f"{path}:{lineno}: expected a non-empty 'fact_ids' list")
            try:
                labeled = {int(fid) for fid in fact_ids}
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{lineno}: 'fact_ids' must contain integers") from exc
            queries.append({"query": entry["query"], "fact_ids": labeled})
    if not queries:
        raise ValueError(f"{path}: no query records found")
    return queries


def cmd_eval(args):
    """Evaluate retrieval quality for a Droid profile.

    Labeled mode (``--queries``) scores against labeled fact ids. Without a
    query file, a self-recall sanity check runs over a sample of stored facts:
    each fact's own text is used as the query and the metric counts whether the
    fact is returned. The output names the mode it actually ran.
    """
    profile_name = (args.name or "").strip().lower().replace(" ", "-")
    if not profile_name:
        return _eval_error("--name is required")
    if args.top_k < 1:
        return _eval_error(f"--top-k must be >= 1 (got {args.top_k})")
    if args.sample < 1:
        return _eval_error(f"--sample must be >= 1 (got {args.sample})")

    profile_dir = Path(args.base_dir) / profile_name
    if not (profile_dir / "config.json").exists():
        return _eval_error(f"droid profile not found: {profile_dir}")

    labeled_queries = None
    if args.queries:
        try:
            labeled_queries = _load_labeled_queries(args.queries)
        except (OSError, ValueError) as exc:
            return _eval_error(str(exc))

    try:
        from src.model.droid_manager import DroidManager

        mgr = DroidManager(base_dir=args.base_dir)
        droid = mgr.get_droid(profile_name)
        facts = droid.knowledge.total_count()
        if facts == 0:
            return _eval_error(f"droid profile {profile_name} has an empty knowledge store; nothing to evaluate")

        top_k = args.top_k
        precision_sum = 0.0
        recall_sum = 0.0
        rr_sum = 0.0

        if labeled_queries is not None:
            mode = "labeled"
            samples = len(labeled_queries)
            for entry in labeled_queries:
                hits = droid.recall(entry["query"], top_k=top_k, threshold=0.0)
                ranked = [h.get("id") for h in hits]
                p, r, rr = _ranking_metrics(ranked, entry["fact_ids"], top_k)
                precision_sum += p
                recall_sum += r
                rr_sum += rr
        else:
            mode = "self_recall"
            active = droid.knowledge.get_active_facts()
            if not active:
                return _eval_error(f"droid profile {profile_name} has no active facts; nothing to evaluate")
            samples = min(args.sample, len(active))
            # Deterministic sample so repeated runs report the same number.
            chosen = random.Random(0).sample(active, samples)
            for fact in chosen:
                hits = droid.recall(fact["text"], top_k=top_k, threshold=0.0)
                ranked = [h.get("id") for h in hits]
                p, r, rr = _ranking_metrics(ranked, {int(fact["id"])}, top_k)
                precision_sum += p
                recall_sum += r
                rr_sum += rr
    except Exception as exc:  # evaluation failure, not invalid input
        print(f"error: evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if samples == 0:
        return _eval_error("evaluation measured no samples")

    # Stable key=value order, no ANSI: a shell can split on '='.
    print(f"mode={mode}")
    print(f"profile={profile_name}")
    print(f"facts={facts}")
    print(f"samples={samples}")
    print(f"top_k={top_k}")
    print(f"precision_at_k={precision_sum / samples:.3f}")
    print(f"recall_at_k={recall_sum / samples:.3f}")
    print(f"mrr={rr_sum / samples:.3f}")
    return 0

def cmd_research(args):
    """Acquire knowledge about a topic from the profile's search provider."""
    from src.model.droid_manager import DroidManager

    profile_name = (args.name or "").strip().lower().replace(" ", "-")
    if not profile_name:
        return _eval_error("--name is required")
    try:
        mgr = DroidManager(base_dir=args.base_dir)
        droid = mgr.get_droid(profile_name)
    except Exception as exc:
        print(f"error: research failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    result = droid.research(args.topic)
    print(f"topic={args.topic}")
    print(f"profile={profile_name}")
    print(f"searched={'true' if result.get('searched') else 'false'}")
    if result.get("facts_learned"):
        print(f"facts_learned={int(result['facts_learned'])}")
    else:
        reason = result.get("reason") or "no facts were learned"
        print(f"facts_learned_skipped={reason}")
    if result.get("searched"):
        sources = ",".join(result.get("sources") or [])
        print(f"sources={sources}")
    if not result.get("available", True):
        print(
            "error: external acquisition is unavailable: "
            f"{result.get('reason') or 'no search provider configured'}",
            file=sys.stderr,
        )
        return 1
    return 0


def _bench_resolve_checkpoint(args) -> Optional[Path]:
    """Explicit checkpoint, or the one a profile manifest declares."""
    if getattr(args, "checkpoint", None):
        return Path(args.checkpoint)
    profile = getattr(args, "profile", None)
    if profile:
        manifest = Path(profile) / "config.json"
        if manifest.is_file():
            try:
                config = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
            declared = config.get("byte_rtu_checkpoint") or config.get("rtu_checkpoint")
            if declared:
                return Path(profile) / str(declared)
    return None


def _bench_tokens(corpus: Optional[str], count: int) -> "torch.Tensor":
    """A deterministic byte slice of the corpus (synthetic when none is given)."""
    import torch

    if corpus:
        data = b""
        for name in corpus.split(","):
            path = Path(name.strip())
            if path.is_file():
                data += path.read_bytes() + b"\n"
    else:
        data = b"The quick brown fox jumps over the lazy dog. " * 32
    if not data:
        data = b"bench"
    repeats = max(1, count // len(data) + 1)
    return torch.tensor(list((data * repeats)[:count]), dtype=torch.long)


def _bench_frozen_pass(model, tokens) -> None:
    import torch

    with torch.no_grad():
        model.reset()
        for byte in tokens.tolist():
            model.step(torch.tensor(byte, dtype=torch.long), frozen=True)


def _bench_perplexity(model, documents) -> tuple:
    import torch

    total, counted = 0.0, 0
    with torch.no_grad():
        for document_id in sorted(documents):
            data = documents[document_id]
            if len(data) < 2:
                continue
            model.reset()
            logits = None
            for index, byte in enumerate(data):
                if logits is not None:
                    log_probs = torch.log_softmax(logits.reshape(-1).float(), dim=-1)
                    total += float(-log_probs[byte].item())
                    counted += 1
                if index + 1 < len(data):
                    (_, _, _), (logits, _) = model.step(
                        torch.tensor(byte, dtype=torch.long), frozen=True
                    )
    return total, counted


def cmd_bench(args):
    """Measure the documented metrics and write one machine-readable report.

    A measurement that could not be taken is printed as ``<metric>_skipped=``
    with its reason and never as a value.
    """
    from src.model import benchmarks
    from src.model.probes import load_probes, score_probes
    from src.model.rtu import Model

    measurements: Dict[str, object] = {}
    checkpoint = _bench_resolve_checkpoint(args)
    missing_reason = None
    model = None

    if checkpoint is None:
        missing_reason = "no checkpoint supplied (use --checkpoint or --profile)"
    elif not checkpoint.is_file():
        missing_reason = f"checkpoint not found: {checkpoint}"
    if missing_reason:
        for name in ("throughput", "perplexity", "probe_score"):
            measurements[name] = benchmarks.skipped(missing_reason)
    else:
        try:
            import torch

            torch.manual_seed(args.seed)
            model = Model(dim=args.dim, layers=args.layers, vocab_size=args.vocab)
            model.load(str(checkpoint))
            model.eval()
        except Exception as exc:
            print(f"error: benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

        tokens = _bench_tokens(args.corpus, args.tokens)
        measurements["throughput"] = benchmarks.measure_throughput(
            lambda: _bench_frozen_pass(model, tokens),
            int(tokens.numel()),
            runs=args.runs,
            warmup=1,
            path="sequential",
            sequence_length=int(tokens.numel()),
            hardware=benchmarks.hardware_description(),
        )
        measurements["rss"] = benchmarks.measure_resident_memory([str(checkpoint)])
        measurements["throughput"]["checkpoint_hash"] = benchmarks.sha256_file(checkpoint)
        documents = {}
        if args.corpus:
            for name in args.corpus.split(","):
                path = Path(name.strip())
                if path.is_file():
                    documents[path.name] = path.read_bytes()
        if documents:
            measurements["perplexity"] = benchmarks.measure_held_out_perplexity(
                lambda held: _bench_perplexity(model, held),
                documents,
                held_out_fraction=args.held_out_fraction,
                seed=args.seed,
            )
        else:
            measurements["perplexity"] = benchmarks.skipped(
                "no corpus supplied for the held-out split"
            )
        if args.probes:
            probe_path = Path(args.probes)
            if probe_path.is_file():
                probe_set = load_probes(probe_path)
                for error in probe_set.errors:
                    print(
                        f"# probe line {error['line']} rejected: {error['reason']}",
                        file=sys.stderr,
                    )
                measurements["probe_score"] = score_probes(
                    model, probe_set, checkpoint_path=checkpoint, probe_path=probe_path
                )
            else:
                measurements["probe_score"] = benchmarks.skipped(
                    f"probe file not found: {probe_path}"
                )
        else:
            measurements["probe_score"] = benchmarks.skipped("no probe file supplied")

    # The parameter count is a property of the configuration, not of the
    # checkpoint, so it is measurable even when no checkpoint was supplied.
    if model is not None:
        params = benchmarks.count_trainable_parameters(model)
    else:
        import torch

        torch.manual_seed(args.seed)
        params = benchmarks.count_trainable_parameters(
            Model(dim=args.dim, layers=args.layers, vocab_size=args.vocab)
        )
    measurements["params"] = benchmarks.measurement(
        params,
        metric="params",
        config={"dim": args.dim, "layers": args.layers, "vocab": args.vocab},
    )
    measurements.setdefault("rss", benchmarks.measure_resident_memory([]))

    report = benchmarks.build_report(
        command=" ".join(sys.argv),
        config={
            "seed": args.seed,
            "tokens": args.tokens,
            "runs": args.runs,
            "dim": args.dim,
            "layers": args.layers,
            "vocab": args.vocab,
            "corpus": args.corpus,
            "probes": args.probes,
            "checkpoint": str(checkpoint) if checkpoint else "",
        },
        measurements=measurements,
        hardware=benchmarks.hardware_description(),
        dependencies=benchmarks.dependency_versions(),
    )
    report_path = benchmarks.write_report(report, args.report)

    throughput = measurements["throughput"]
    if benchmarks.is_measured(throughput):
        print(f"tokens_per_second={throughput['value']:.2f}")
    else:
        print(f"tokens_per_second_skipped={throughput['reason']}")
    print(f"params={int(measurements['params']['value'])}")
    rss = measurements["rss"]
    if benchmarks.is_measured(rss):
        print(f"rss_mb={rss['value']:.1f}")
    else:
        print(f"rss_mb_skipped={rss['reason']}")
    perplexity = measurements["perplexity"]
    if benchmarks.is_measured(perplexity):
        print(f"held_out_perplexity={perplexity['value']:.3f}")
    else:
        print(f"held_out_perplexity_skipped={perplexity['reason']}")
    probes = measurements["probe_score"]
    if benchmarks.is_measured(probes):
        for name, stats in probes.get("per_phenomenon", {}).items():
            print(f"grammar_accuracy.{name}={stats['accuracy']:.3f}")
        print(f"grammar_accuracy={probes['accuracy']:.3f}")
    else:
        print(f"grammar_accuracy_skipped={probes['reason']}")
    print(f"outcome={report['outcome']}")
    print(f"report={report_path}")
    return 0


def cmd_download_model(args):
    """Download GGUF model."""
    try:
        from src.model.gguf_backend import GgufBackend
        backend = GgufBackend()
        path = backend.download_model(args.repo_id, args.filename, args.local_dir)
        print(f"Downloaded model to {path}")
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        # Fallback to huggingface_hub direct
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=args.repo_id, filename=args.filename, local_dir=args.local_dir)
            print(f"Downloaded via huggingface_hub to {path}")
        except Exception as e2:
            print(f"Fallback also failed: {e2}", file=sys.stderr)
            return 1
    return 0

def cmd_generate(args):
    """Direct LLM generation with memory."""
    from src.model.droid_manager import DroidManager
    mgr = DroidManager(base_dir=args.base_dir)
    droid = mgr.get_droid(args.name)
    
    if hasattr(droid, 'generate_with_memory'):
        result = droid.generate_with_memory(args.query, strategy=args.strategy)
        if isinstance(result, dict):
            print(result.get('text', json.dumps(result, indent=2)))
            if args.verbose:
                print(json.dumps(result, indent=2), file=sys.stderr)
        else:
            print(result)
    else:
        print("GGUF backend not available, falling back to chat", file=sys.stderr)
        print(droid.chat(args.query))

def main(argv: Optional[List[str]] = None, _deprecation_notice: bool = False):
    if _deprecation_notice:
        print(
            "deprecated: `python -m src.cli` is deprecated; use `recamem` instead",
            file=sys.stderr,
        )
    parser = argparse.ArgumentParser(prog='tmt-droid', description='Unified headless CLI for Droid')
    parser.add_argument('--base-dir', default=None, help='Profiles directory (default: recamem home, or legacy droids/)')
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # chat
    p_chat = subparsers.add_parser('chat', help='Interactive terminal chat')
    p_chat.add_argument('--name', default='droid-alpha', help='Droid name')
    p_chat.add_argument('--mode', choices=['auto', 'fast', 'slow'], default='auto', help='Generation mode')
    p_chat.add_argument('query', nargs='?', help='Single query (if omitted, interactive loop)')
    p_chat.set_defaults(func=cmd_chat)
    
    # teach
    p_teach = subparsers.add_parser('teach', help='Teach from stdin or file')
    p_teach.add_argument('--name', default='droid-alpha', help='Droid name')
    p_teach.add_argument('--file', help='File to read instead of stdin')
    p_teach.add_argument('--source', help='Source tag')
    p_teach.set_defaults(func=cmd_teach)
    
    # sleep
    p_sleep = subparsers.add_parser('sleep', help='Run one consolidation cycle')
    p_sleep.add_argument('--name', default='droid-alpha', help='Droid name')
    p_sleep.add_argument('--json', action='store_true', help='Emit the structured report')
    p_sleep.set_defaults(func=cmd_sleep)

    # decide
    p_decide = subparsers.add_parser('decide', help='System-1 option scoring')
    p_decide.add_argument('query', help='Query text')
    p_decide.add_argument('--options', help='Comma-separated options')
    p_decide.add_argument('--name', default='droid-alpha', help='Droid name')
    p_decide.add_argument('--json', action='store_true', help='Output JSON')
    p_decide.set_defaults(func=cmd_decide)
    
    # serve
    p_serve = subparsers.add_parser('serve', help='Serve MCP or REST')
    p_serve.add_argument('--mcp', action='store_true', help='Serve MCP stdio')
    p_serve.add_argument('--rest', action='store_true', help='Serve REST microdaemon')
    p_serve.add_argument('--port', type=int, default=8000, help='REST port')
    p_serve.set_defaults(func=cmd_serve)
    
    # package
    p_pkg = subparsers.add_parser('package', help='Portable brain packages')
    p_pkg_sub = p_pkg.add_subparsers(dest='package_cmd', required=True)
    p_export = p_pkg_sub.add_parser('export', help='Export droid to .droid archive')
    p_export.add_argument('name', help='Droid name')
    p_export.add_argument('--out', required=True, help='Output .droid file')
    p_import = p_pkg_sub.add_parser('import', help='Import .droid archive')
    p_import.add_argument('archive', help='Archive path')
    p_import.add_argument('--name', required=True, help='Target droid name')
    p_list = p_pkg_sub.add_parser('list', help='List archive contents')
    p_list.add_argument('archive', help='Archive path')
    p_pkg.set_defaults(func=cmd_package)
    
    # eval
    p_eval = subparsers.add_parser('eval', help='Retrieval evaluation')
    p_eval.add_argument('--name', required=True, help='Droid name')
    p_eval.add_argument('--queries', help='JSONL file of {"query": ..., "fact_ids": [...]} labels')
    p_eval.add_argument('--top-k', type=int, default=5, help='Cutoff k (default 5)')
    p_eval.add_argument('--sample', type=int, default=50, help='Facts sampled for the self-recall check (default 50)')
    p_eval.set_defaults(func=cmd_eval)

    # research
    p_research = subparsers.add_parser(
        'research', help='Acquire knowledge about a topic from the web'
    )
    p_research.add_argument('topic', help='Topic to research')
    p_research.add_argument('--name', default='droid-alpha', help='Droid name')
    p_research.set_defaults(func=cmd_research)

    # bench
    p_bench = subparsers.add_parser(
        'bench', help='Measure throughput, size, held-out perplexity and grammar probes'
    )
    p_bench.add_argument('--profile', help='Droid profile whose manifest names the checkpoint')
    p_bench.add_argument('--checkpoint', help='Byte-RTU checkpoint to measure')
    p_bench.add_argument('--corpus', help='Comma-separated corpus files for throughput and perplexity')
    p_bench.add_argument('--tokens', type=int, default=2000, help='Tokens per timed run (default 2000)')
    p_bench.add_argument('--probes', help='Grammar probe JSONL file')
    p_bench.add_argument('--report', default='reports/bench-report.json', help='Report path')
    p_bench.add_argument('--runs', type=int, default=3, help='Timed runs (default 3)')
    p_bench.add_argument('--dim', type=int, default=64, help='Model dimension')
    p_bench.add_argument('--layers', type=int, default=2, help='Model layer count')
    p_bench.add_argument('--vocab', type=int, default=256, help='Vocabulary size')
    p_bench.add_argument('--seed', type=int, default=0, help='Construction and split seed')
    p_bench.add_argument('--held-out-fraction', type=float, default=0.25,
                         help='Document-level held-out fraction (default 0.25)')
    p_bench.set_defaults(func=cmd_bench)
    
    # download-model
    p_dl = subparsers.add_parser('download-model', help='Download GGUF model')
    p_dl.add_argument('repo_id', help='HF repo id')
    p_dl.add_argument('filename', help='Filename or pattern')
    p_dl.add_argument('--local-dir', default='models', help='Local dir')
    p_dl.set_defaults(func=cmd_download_model)
    
    # generate
    p_gen = subparsers.add_parser('generate', help='Generate with memory')
    p_gen.add_argument('query', help='Query')
    p_gen.add_argument('--name', default='droid-alpha', help='Droid name')
    p_gen.add_argument('--strategy', default='auto', choices=['auto', 'prompt', 'hidden', 'kv_cache'])
    p_gen.add_argument('--verbose', action='store_true')
    p_gen.set_defaults(func=cmd_generate)
    
    args = parser.parse_args(argv)
    if getattr(args, "base_dir", None) in (None, ""):
        # Profiles live in the recamem home, or in a legacy droids/ directory
        # that already exists (recamem/identity).
        from recamem.identity import default_base_dir

        args.base_dir = str(default_base_dir())
    return args.func(args)

if __name__ == '__main__':
    # The historical module entry point keeps working for one release and names
    # its replacement (recamem/identity).
    sys.exit(main(_deprecation_notice=True) or 0)
