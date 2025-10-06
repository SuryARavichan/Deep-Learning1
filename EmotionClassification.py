Emotion classification wrapper. Loads model once and reuses it across requests.
logger = logging.getLogger(__name__)




class EmotionClassifier:
def __init__(self, model_name: str = None, cache_dir: str = None, device: str = None):
params = read_params()
self.model_name = model_name or params.get("model_name", "nateraw/bert-base-uncased-emotion")
self.cache_dir = cache_dir or params.get("cache_dir", "./model_cache")


# Choose device: cuda if available else cpu
if device:
self.device = torch.device(device)
else:
self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


self._tokenizer = None
self._model = None
self._labels = None
self._load()


def _load(self):
logger.info("Loading tokenizer and model: %s", self.model_name)
self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name, cache_dir=self.cache_dir)
self._model.to(self.device)
# Try to read labels from model config
cfg = getattr(self._model, "config", None)
if cfg and getattr(cfg, "id2label", None):
# id2label is a dict mapping int->label
self._labels = [cfg.id2label[i] for i in sorted(cfg.id2label.keys())]
else:
# fallback to generic labels
self._labels = [f"LABEL_{i}" for i in range(self._model.config.num_labels)]
logger.info("Loaded model with labels: %s", self._labels)


def classify(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
if not text or not text.strip():
return []


inputs = self._tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
with torch.no_grad():
outputs = self._model(**inputs)
logits = outputs.logits
probs = F.softmax(logits, dim=-1).cpu().squeeze(0)


topk = torch.topk(probs, k=min(top_k, probs.size(0)))
results = []
for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
label = self._labels[idx] if idx < len(self._labels) else str(idx)
results.append({"label": label, "score": float(score)})
return results




# Create a module-level singleton to avoid reloading in server imports
_default_classifier: EmotionClassifier = None




def get_default_classifier() -> EmotionClassifier:
global _default_classifier
if _default_classifier is None:
params = read_params()
_default_classifier = EmotionClassifier(model_name=params.get("model_name"), cache_dir=params.get("cache_dir"))
return _default_classifier




def classify_emotions(text: str, top_k: int = 3):
clf = get_default_classifier()
return clf.classify(text, top_k=top_k)




if __name__ == "__main__":
# quick test
c = get_default_classifier()
sample = "I love this! It's amazing and makes my day."
print(c.classify(sample, top_k=5))
