#!/usr/bin/env bash
# Re-run, on Claude and OpenAI, every file the Modal model has already parsed, so the three can
# be compared in the viewer (Run / Compare with).
#
#   OPENAI_MODEL=<name> scripts/compare_providers.sh            # both providers, every file
#   scripts/compare_providers.sh claude                         # one provider
#   CLAUDE_MODEL=claude-haiku-5-5 scripts/compare_providers.sh claude
#   MAX_LINES=200 scripts/compare_providers.sh claude           # small documents only
#   TAG=haiku CLAUDE_MODEL=claude-haiku-5-5 scripts/compare_providers.sh claude   # -> out-claude-haiku/
#
# The file list is read from out/ (the Modal results): every file that needed the model there,
# smallest first. Files in known formats (AWS CUR, FOCUS) need no model and are left out.
# Each provider writes to its own folder, out-<provider>/ (out-<provider>-<TAG>/ with TAG set). Files already done are skipped, so
# the script can be stopped and started again. Keys come from .env (ANTHROPIC_API_KEY,
# OPENAI_API_KEY).
set -uo pipefail
cd "$(dirname "$0")/.."

BASELINE=${BASELINE:-out}
INPUTS=${INPUTS:-cost-estimate-inputs}
MAX_LINES=${MAX_LINES:-0}          # 0 = no limit; otherwise skip documents with more tagged lines
PROVIDERS=("$@")
[ ${#PROVIDERS[@]} -eq 0 ] && PROVIDERS=(claude openai)

FILES=$(uv run python - "$BASELINE" "$MAX_LINES" <<'PY'
import glob, json, sys
from collections import defaultdict
baseline, limit = sys.argv[1], int(sys.argv[2])
lines, needs_model = defaultdict(int), set()
for path in glob.glob(f"{baseline}/*/document.json"):
    doc = json.load(open(path))
    name = doc["source_file"]
    if (doc.get("extraction_method") or "").startswith("llm"):
        needs_model.add(name)
    try:
        lines[name] += sum(1 for _ in open(path.replace("document.json", "tags.jsonl")))
    except OSError:
        pass
for name in sorted(needs_model, key=lambda n: (lines[n], n)):
    if not limit or lines[name] <= limit:
        print(name)
PY
)
[ -z "$FILES" ] && { echo "no model-parsed documents found in $BASELINE/"; exit 1; }
echo "Files:"; echo "$FILES" | sed 's/^/  /'

for provider in "${PROVIDERS[@]}"; do
  case $provider in
    claude) model=${CLAUDE_MODEL:-} ;;
    openai) model=${OPENAI_MODEL:-}
            [ -z "$model" ] && { echo "skipping openai: set OPENAI_MODEL=<name>"; continue; } ;;
    *)      model=${MODEL:-} ;;
  esac
  dest="out-$provider${TAG:+-$TAG}"
  echo; echo "=== $provider ${model:+($model) }-> $dest/"
  while IFS= read -r file; do
    uv run invoice-parser parse "$INPUTS/$file" -o "$dest" -v --resume \
      --provider "$provider" ${model:+--model "$model"} \
      || echo "FAILED: $file on $provider"
  done <<< "$FILES"
done

echo; echo "Compare in the viewer: uv run --extra viewer streamlit run app/viewer.py"
