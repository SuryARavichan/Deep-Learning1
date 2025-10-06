
"""
from pathlib import Path
import json
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_PARAMS = {
"model_name": "nateraw/bert-base-uncased-emotion",
"cache_dir": "./model_cache"
}




def read_params(path: str = "params.json") -> dict:
p = Path(path)
if not p.exists():
logger.warning("params.json not found, using default parameters")
return DEFAULT_PARAMS
try:
return json.loads(p.read_text(encoding="utf-8"))
except Exception as e:
logger.exception("Failed to read params.json: %s", e)
return DEFAULT_PARAMS




def download_models(params: dict):
model_name = params.get("model_name", DEFAULT_PARAMS["model_name"])
cache_dir = Path(params.get("cache_dir", DEFAULT_PARAMS["cache_dir"]))
cache_dir.mkdir(parents=True, exist_ok=True)


logger.info("Pre-downloading model %s into %s", model_name, cache_dir)
# Download tokenizer and model to the cache dir
AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir), local_files_only=False)
AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=str(cache_dir), local_files_only=False)
logger.info("Model and tokenizer downloaded successfully")




if __name__ == "__main__":
params = read_params()
download_models(params)