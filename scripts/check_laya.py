"""Step-by-step Laya diagnostic. Run it before the app:   python scripts/check_laya.py
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
    req = urllib.request.Request("https://huggingface.co/convaiinnovations/laya", method="HEAD")
    return "huggingface.co reachable (HTTP %s)" % urllib.request.urlopen(req, timeout=10).status


def cache_info():
    root = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    files = glob.glob(os.path.join(root, "hub", "models--convaiinnovations--laya*", "**", "*.safetensors"), recursive=True)
    size = sum(os.path.getsize(f) for f in files if os.path.isfile(f)) / 1e9
    return "%d weight file(s) cached, %.2f GB in %s" % (len(files), size, root)


holder = {}


def load_router():
    from laya import Router
    holder["r"] = Router()
    holder["r"].predict({"alert": "warm-up"}, {"q": {"type": "noul", "instructions": "Is this a test?"}})
    r = holder["r"]
    dev = getattr(r, "device", None)
    return "device=%s loaded=%s" % (dev() if callable(dev) else dev, getattr(r, "loaded", None))


def real_predict():
    from oncall_gate.alerts import BY_ID
    from oncall_gate.engine import render_state
    from oncall_gate.questions import QUESTIONS
    alert = BY_ID["D2"]
    r = holder["r"]
    r.predict(render_state(alert), QUESTIONS)            # warm
    t = time.time()
    res = r.predict(render_state(alert), QUESTIONS)
    ms = (time.time() - t) * 1000
    a = res["answers"]
    return ("%.0f ms | category=%s impact=%.2f recovering=%.2f sustained=%.2f risk=%.2f severity=%.2f" % (
        ms, a["category"]["choice"], a["customer_impact"]["noul"], a["self_recovering"]["noul"],
        a["sustained"]["noul"], a["business_risk"]["noul"], a["severity"]["score"]))


print("Laya local diagnostic  (python %s on %s)\n" % (platform.python_version(), platform.platform(terse=True)))
step("PyTorch imports", torch_info)
step("Laya imports", laya_ver)
step("Hugging Face reachable", hf_reach, fatal=False)
step("Cached weights (before)", cache_info, fatal=False)
print("       ... loading the model; the FIRST run downloads ~1.3 GB, please wait ...", flush=True)
step("Load Laya + warm-up call", load_router)
step("Cached weights (after)", cache_info, fatal=False)
step("Real alert through Laya", real_predict)
print("\nALL GOOD: Laya is running locally. Start the app (app.py) and the banner should say it is ready.")
print("To prove no network is used per call, rerun with HF_HUB_OFFLINE=1.")
