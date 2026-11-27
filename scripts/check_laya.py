"""Step-by-step Laya diagnostic. Run it before the app:   .venv/bin/python scripts/check_laya.py [vega|vega-4b|classic]
Each step prints OK/FAIL with the exact error, so you can see where "Laya running locally" breaks."""
import glob
import os
import platform
import sys
import time
import urllib.request

os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def step(name, fn, fatal=True):
    t = time.time()
    try:
        out = fn()
        print("[ OK ] %-34s %5.1fs  %s" % (name, time.time() - t, out or ""), flush=True)
        return out
    except Exception as exc:  # noqa: BLE001
        print("[FAIL] %-34s %s: %s" % (name, type(exc).__name__, str(exc)[:300]), flush=True)
        if fatal:
            print("\nStopped at this step. See the README troubleshooting section.")
            sys.exit(1)


def torch_info():
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    return "torch %s, best device: %s" % (torch.__version__, dev)


def laya_ver():
    import laya
    return "laya %s" % laya.__version__


def hf_reach():
    import urllib.error
    repos = ["https://huggingface.co/nandakishorm/vega-U", "https://huggingface.co/convaiinnovations/laya"]
    url = repos[1] if MODEL == "classic" else repos[0]
    ctx = None
    try:                                     # python.org builds ship without CA certs; certifi comes with requests
        import certifi
        import ssl
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    try:
        status = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=10, context=ctx).status
    except urllib.error.HTTPError as exc:   # a 401/403 still proves the host is reachable
        status = exc.code
    return "huggingface.co reachable (HTTP %s for %s)" % (status, url.split("co/")[1])


def cache_info():
    root = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    pats = ["models--nandakishorm--vega*", "models--convaiinnovations--laya*", "models--Qwen--*"]
    files = sorted({f for p in pats for f in glob.glob(os.path.join(root, "hub", p, "**", "*.safetensors"), recursive=True)})
    size = sum(os.path.getsize(f) for f in files if os.path.isfile(f)) / 1e9
    return "%d weight file(s) cached, %.2f GB in %s" % (len(files), size, root)


holder = {}


def load_engine():
    from oncall_gate.engine import LayaEngine
    eng = holder["e"] = LayaEngine(MODEL)
    eng.predict({"alert": "warm-up", "service": "x", "tier": "tier-3", "source": "x"})
    d = eng.describe()
    return "%s | device=%s loaded=%s" % (d["model_label"], d["device"], d["checkpoints_loaded"])


def real_predict():
    from oncall_gate.alerts import BY_ID
    from oncall_gate.engine import render_state
    from oncall_gate.policy import review_signal
    alert, eng = BY_ID["D2"], holder["e"]
    eng.predict(render_state(alert))            # warm
    t = time.time()
    a = eng.predict(render_state(alert))["answers"]
    ms = (time.time() - t) * 1000
    return ("%.0f ms | category=%s impact=%.2f recovering=%.2f sustained=%.2f risk=%.2f severity=%.2f | %s" % (
        ms, a["category"]["choice"], a["customer_impact"]["noul"], a["self_recovering"]["noul"],
        a["sustained"]["noul"], a["business_risk"]["noul"], a["severity"]["score"], review_signal(a)[1]))


MODEL = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("LAYA_MODEL", "vega")).lower()
print("Laya local diagnostic  (python %s on %s, LAYA_MODEL=%s)\n" % (platform.python_version(), platform.platform(terse=True), MODEL))
if sys.version_info < (3, 10):
    print("[FAIL] laya >= 0.5 needs Python 3.10 or newer. Recreate .venv with a newer python3.")
    sys.exit(1)
step("PyTorch imports", torch_info)
step("Laya imports", laya_ver)
step("Hugging Face reachable", hf_reach, fatal=False)
step("Cached weights (before)", cache_info, fatal=False)
print("       ... loading the model; the FIRST run downloads several GB, please wait ...", flush=True)
step("Load Laya + warm-up call", load_engine)
step("Cached weights (after)", cache_info, fatal=False)
step("Real alert through Laya", real_predict)
print("\nALL GOOD: Laya is running locally. Start the app (app.py) and the banner should say it is ready.")
print("To prove no network is used per call, rerun with HF_HUB_OFFLINE=1.")
