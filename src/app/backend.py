"""Model backend for the app: the evaluated generation path plus the Task 3/5 decoders for benchmarking.

`ModelBackend.load(cfg)` loads the real target (base + LoRA, merged) and the draft. `ModelBackend.tiny()` builds
tiny random Qwen2 models with the real tokenizer for --smoke runs: same code path, seconds instead of minutes,
and its random outputs exercise the retry/fallback path.
"""
import json
import re
import time
from dataclasses import dataclass

import torch

from src.data.dataset import prompt_text
from src.data.load import LABELS
from src.decoding.manual_decoder import ManualQwen2, kv_bytes_per_token
from src.eval.predict import greedy_config, load_model

EOS_TOKENS = ("<|im_end|>", "<|endoftext|>")


@dataclass
class GenOut:
    text: str
    n_tokens: int
    seconds: float
    hit_eos: bool


def _resolve_device(cfg: dict) -> tuple[str, torch.dtype]:
    want = cfg.get("device", "auto")
    if want == "cuda" or (want == "auto" and torch.cuda.is_available()):
        return "cuda", torch.float16
    return "cpu", getattr(torch, cfg.get("dtype", "bfloat16"))


def _load_plain(model_id: str, dtype, device):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation="sdpa").to(device).eval()


class ModelBackend:
    def __init__(self, tok, model, prompt_version: str, info: dict, draft=None, draft_loader=None):
        self.tok, self.model, self.prompt_version, self.info = tok, model, prompt_version, info
        self.device = model.device
        self.eos_ids = {tok.convert_tokens_to_ids(t) for t in EOS_TOKENS}
        self.target_dec = ManualQwen2(model)            # shares the merged weights (no extra memory)
        self._draft, self._draft_loader, self._draft_dec = draft, draft_loader, None
        self.kv_bytes_target = kv_bytes_per_token(model.config, torch.finfo(model.dtype).bits // 8)

    # ---------- construction ----------
    @classmethod
    def load(cls, cfg: dict, log=print) -> "ModelBackend":
        res = cfg["resolved"]
        device, dtype = _resolve_device(cfg)
        if device == "cpu" and cfg.get("threads"):
            torch.set_num_threads(int(cfg["threads"]))
        t0 = time.perf_counter()
        log(f"loading {cfg['model_id']} + adapter {res['adapter']} ({dtype}, {device}) ...")
        tok, model = load_model(cfg["model_id"], res["adapter"], dtype, device)
        load_s = time.perf_counter() - t0
        info = {"model_id": cfg["model_id"], "adapter": res["adapter"], "adapter_source": res["source"],
                "prompt_version": res["prompt_version"], "draft_model": cfg["draft_model"], "dtype": str(dtype).replace("torch.", ""),
                "device": torch.cuda.get_device_name(0) if device == "cuda" else f"cpu ({torch.get_num_threads()} threads)",
                "load_seconds": round(load_s, 1)}
        log(f"target loaded in {load_s:.1f} s")

        def draft_loader():
            from transformers import AutoTokenizer
            dtok = AutoTokenizer.from_pretrained(cfg["draft_model"])
            # Speculative decoding needs identical tokenization (D-008, D-021).
            assert tok.get_vocab() == dtok.get_vocab() and tok.all_special_tokens == dtok.all_special_tokens, \
                "draft and target tokenizers differ"
            return _load_plain(cfg["draft_model"], dtype, device)

        b = cls(tok, model, res["prompt_version"], info, draft_loader=draft_loader)
        if cfg.get("preload_draft", True):
            t1 = time.perf_counter()
            b.draft_dec
            log(f"draft {cfg['draft_model']} loaded in {time.perf_counter() - t1:.1f} s")
        return b

    @classmethod
    def tiny(cls, prompt_version: str = "v1", seed: int = 0, tokenizer_id: str = "Qwen/Qwen2.5-0.5B-Instruct") -> "ModelBackend":
        """Random 2-layer Qwen2 target + draft with the real tokenizer (for --smoke; outputs are gibberish)."""
        from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
        tok = AutoTokenizer.from_pretrained(tokenizer_id)
        tok.padding_side = "left"

        def make(s, layers):
            torch.manual_seed(s)
            c = Qwen2Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                            rope_parameters={"rope_type": "default", "rope_theta": 10000.0}, tie_word_embeddings=False)
            return Qwen2ForCausalLM(c).to(torch.float32).eval()

        info = {"model_id": "tiny-random-qwen2 (smoke)", "adapter": None, "adapter_source": "smoke test",
                "prompt_version": prompt_version, "draft_model": "tiny-random-qwen2-draft (smoke)",
                "dtype": "float32", "device": "cpu", "load_seconds": 0.0}
        return cls(tok, make(seed, 2), prompt_version, info, draft=make(seed + 1, 1))

    # ---------- screening path (identical to src/eval/predict.py) ----------
    def prompt(self, sentences, criterion_type: str, criterion_text: str) -> str:
        row = {"sentences": sentences, "criterion_type": criterion_type, "criterion_text": criterion_text}
        return prompt_text(self.tok, row, self.prompt_version)

    def encode(self, prompt: str) -> torch.Tensor:
        return self.tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

    def count_tokens(self, prompt: str) -> int:
        return len(self.tok(prompt, add_special_tokens=False)["input_ids"])

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int, repetition_penalty: float = 1.0) -> GenOut:
        ids = self.encode(prompt)
        t0 = time.perf_counter()
        out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                  generation_config=greedy_config(self.tok, max_new_tokens, repetition_penalty))
        secs = time.perf_counter() - t0
        new = out[0, ids.shape[1]:].tolist()
        return GenOut(self.tok.decode(new, skip_special_tokens=True), len(new), secs,
                      bool(new) and new[-1] in self.eos_ids)

    @torch.no_grad()
    def score_labels(self, prompt: str, criterion_type: str, evidence_ids: list[int]) -> dict[str, float]:
        """Fallback (D-028): log-probability of each allowed answer in the exact training target format."""
        from src.data.prompt import build_target
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        end = self.tok.convert_tokens_to_ids("<|im_end|>")
        seqs, spans = [], []
        for label in LABELS[criterion_type]:
            c_ids = self.tok(build_target(label, evidence_ids), add_special_tokens=False)["input_ids"] + [end]
            seqs.append(p_ids + c_ids)
            spans.append((len(p_ids), len(p_ids) + len(c_ids)))
        n = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id
        ids = torch.tensor([s + [pad] * (n - len(s)) for s in seqs], device=self.device)       # right padding
        mask = torch.tensor([[1] * len(s) + [0] * (n - len(s)) for s in seqs], device=self.device)
        logp = torch.log_softmax(self.model(input_ids=ids, attention_mask=mask).logits.float(), dim=-1)
        scores = {}
        for b, (label, (a, z)) in enumerate(zip(LABELS[criterion_type], spans)):
            tgt = ids[b, a:z]
            scores[label] = float(logp[b, a - 1:z - 1].gather(-1, tgt[:, None]).sum())
        return scores

    # ---------- benchmark (Tasks 3 and 5) ----------
    @property
    def draft_dec(self) -> ManualQwen2:
        if self._draft_dec is None:
            if self._draft is None:
                self._draft = self._draft_loader()
            self._draft_dec = ManualQwen2(self._draft)
            self.kv_bytes_draft = kv_bytes_per_token(self._draft.config, torch.finfo(self._draft.dtype).bits // 8)
        return self._draft_dec

    @property
    def draft_loaded(self) -> bool:
        return self._draft_dec is not None


_IDS_RE = re.compile(r'"evidence_ids"\s*:\s*\[([^\]]*)\]')


def salvage_evidence_ids(text: str) -> list[int]:
    """Best-effort evidence IDs from a malformed answer, e.g. one cut off after the ID list."""
    m = _IDS_RE.search(text or "")
    if not m:
        return []
    try:
        return [int(x) for x in json.loads("[" + m.group(1) + "]") if isinstance(x, int)]
    except (json.JSONDecodeError, ValueError):
        return [int(x) for x in re.findall(r"\d+", m.group(1))]
