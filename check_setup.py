"""Pre-flight check: python deps, Ollama server, model. Run: python check_setup.py"""
import importlib
import sys

ok = True
print("Political Domino - setup check\n")
for module, pip_name in [("streamlit", "streamlit"), ("pypdf", "pypdf"), ("docx", "python-docx"), ("requests", "requests")]:
    try:
        importlib.import_module(module)
        print(f"  [ok]   Python package {pip_name}")
    except ImportError:
        ok = False
        print(f"  [FAIL] Python package {pip_name} is missing -> pip install -r requirements.txt")

from political_domino import config, llm  # noqa: E402

client = llm.OllamaClient()
status = client.server_status()
if not status.reachable:
    ok = False
    print(f"  [FAIL] Ollama is not reachable at {config.OLLAMA_URL}")
    print("         Install from https://ollama.com, then start it (`ollama serve` or the desktop app).")
else:
    print(f"  [ok]   Ollama is running ({len(status.models)} model(s) installed)")
    if llm.model_installed(config.DEFAULT_MODEL, status.models):
        print(f"  [ok]   Model {config.DEFAULT_MODEL} is installed")
    else:
        ok = False
        print(f"  [FAIL] Model {config.DEFAULT_MODEL} is not installed -> ollama pull {config.DEFAULT_MODEL}")

print("\nReady." if ok else "\nFix the items above, then run: streamlit run app.py")
sys.exit(0 if ok else 1)
