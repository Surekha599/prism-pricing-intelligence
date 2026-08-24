"""Smoke-test all PRISM pages via Streamlit AppTest."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

PAGES = ["overview", "pricing", "promolab", "segments", "forecast",
         "inventory", "recs", "simulator", "explorer", "about"]

at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
at.run()
assert not at.exception, f"startup failed: {at.exception}"

# nav radio is the first radio in the sidebar
fails = []
for p in PAGES:
    at.sidebar.radio[0].set_value(p).run(timeout=180)
    if at.exception:
        fails.append((p, str(at.exception[0].value)[:300]))
    else:
        print(f"  ok  {p}")

if fails:
    print("\nFAILURES:")
    for p, e in fails:
        print(f"  {p}: {e}")
    sys.exit(1)
print("\nall pages render cleanly")
