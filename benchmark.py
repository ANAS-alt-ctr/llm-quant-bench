"""
LLM Quantization Benchmark + Visualizer
Compatible with Kaggle Notebooks & Google Colab (GPU required)

- Tries PREFERRED_MODEL first, then ordered fallbacks (stops at first accessible model).
- Verifies repo access via huggingface_hub metadata download before loading tokenizer.
- Benchmarks: 4-bit (NF4), 8-bit (INT8), 16-bit (FP16).
- Fixes INT8 generation issues by casting floating inputs to float16 on CUDA and passing only input_ids/attention_mask to generate.
- Aggressive OOM handling and memory flushing between runs.
- Produces a dark-mode 3-panel style PNG at 300 DPI.
"""

import os
import sys
import time
import gc
import warnings
import logging
import subprocess
from typing import List, Optional, Dict, Any

# -------------------- Silent dependency upgrade --------------------
def _silent_upgrade(packages: List[str]):
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-U", "--no-warn-script-location"] + packages,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception:
        # best-effort; continue if environment already has packages
        pass

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

_PACKAGES = [
    "bitsandbytes>=0.46.1",
    "transformers>=4.42.0",
    "accelerate>=0.30.0",
    "huggingface_hub"
]
_silent_upgrade(_PACKAGES)

# -------------------- Imports (after upgrade) --------------------
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login, hf_hub_download

# -------------------- Configuration (user-specified) --------------------
PREFERRED_MODEL = os.getenv("PREFERRED_MODEL", "google/gemma-2-9b-it")
FALLBACK_CANDIDATES = [
    "google/gemma-2-9b",
    "meta-llama/Llama-3-7b-Instruct",
    "nvidia/Alpamayo-R1-10B",
    "google/gemma-2-9b-it"
]
BENCHMARK_PROMPT = os.getenv("BENCHMARK_PROMPT", "Explain quantum computing in 3 clear bullet points.")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "128"))
OUTPUT_PNG_NAME = os.getenv("OUTPUT_PNG_NAME", "llm_benchmark.png")

# -------------------- HF token resolution --------------------
def resolve_hf_token() -> str:
    token = os.getenv("HF_TOKEN", "hf_xxxxxxxxxxxxxxxxxxxxx").strip()
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            print("[+] Retrieved HF_TOKEN from Kaggle Secrets.")
            return token.strip()
    except Exception:
        pass
    try:
        from google.colab import userdata
        token = userdata.get('HF_TOKEN')
        if token:
            print("[+] Retrieved HF_TOKEN from Colab userdata.")
            return token.strip()
    except Exception:
        pass
    return ""

# -------------------- Utilities --------------------
def flush_cuda():
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

# Silence bitsandbytes info lines
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16")

# -------------------- Benchmarker --------------------
class UniversalLLMBenchmarker:
    def __init__(self, preferred_model: str, fallback_candidates: List[str], token: str):
        self.preferred_model = preferred_model
        self.fallback_candidates = fallback_candidates
        self.token = token
        self.model_id: Optional[str] = None
        self.tokenizer: Optional[AutoTokenizer] = None

        if not torch.cuda.is_available():
            raise RuntimeError("CRITICAL ERROR: CUDA GPU not detected! Enable GPU in Kaggle/Colab settings.")

        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = sum([torch.cuda.get_device_properties(i).total_memory for i in range(gpu_count)]) / (1024**3)
        print(f"[+] GPU Detected: {gpu_count} x {gpu_name}")
        print(f"[+] Total VRAM Available: {round(total_vram, 2)} GB")

        self._authenticate_and_select_model()

    def _authenticate_and_select_model(self):
        # Authenticate if token provided
        if self.token:
            try:
                login(token=self.token)
                print("[+] Hugging Face authentication successful.")
            except Exception as e:
                print(f"[!] HF Auth Warning: {e}")

        # Build ordered list: preferred first, then fallbacks (deduplicated)
        ordered: List[str] = []
        if self.preferred_model:
            ordered.append(self.preferred_model)
        for c in self.fallback_candidates:
            if c not in ordered:
                ordered.append(c)

        last_exc = None
        for candidate in ordered:
            print(f"[+] Checking access for candidate: {candidate}")
            # Try to download a small metadata file to verify repo access (does not download weights)
            ok = False
            for fname in ("config.json", "tokenizer_config.json", "tokenizer.json"):
                try:
                    hf_hub_download(repo_id=candidate, filename=fname, token=self.token if self.token else None, repo_type="model")
                    ok = True
                    break
                except Exception as e:
                    last_exc = e
                    continue

            if not ok:
                print(f"[!] Access check failed for {candidate}: {last_exc}")
                continue

            # If metadata download succeeded, attempt to load tokenizer
            try:
                print(f"[+] Loading tokenizer for: {candidate}")
                tok = AutoTokenizer.from_pretrained(candidate, token=self.token if self.token else None, trust_remote_code=True)
                self.model_id = candidate
                self.tokenizer = tok
                print(f"[✓] Selected model: {candidate}")
                break
            except Exception as e:
                last_exc = e
                print(f"[!] Tokenizer load failed for {candidate}: {e}")
                continue

        if self.tokenizer is None:
            raise RuntimeError(f"Failed to select a usable model from candidates. Last error: {last_exc}")

        # Ensure pad token exists
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _flush_memory(self):
        flush_cuda()

    def _safe_model_from_pretrained(self, quant_config: Optional[BitsAndBytesConfig], dtype: Optional[torch.dtype]):
        """
        Wraps AutoModelForCausalLM.from_pretrained with OOM handling and retries.
        """
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=quant_config,
                dtype=dtype if quant_config is None else None,
                device_map="auto",
                token=self.token if self.token else None,
                trust_remote_code=True,
                attn_implementation="sdpa"
            )
            return model, None
        except torch.cuda.OutOfMemoryError as oom:
            self._flush_memory()
            return None, f"OOM: {oom}"
        except Exception as e:
            self._flush_memory()
            return None, str(e)

    def benchmark_variant(
        self,
        variant_name: str,
        quant_config: Optional[BitsAndBytesConfig] = None,
        dtype: Optional[torch.dtype] = torch.float16
    ) -> Dict[str, Any]:
        self._flush_memory()
        print("\n" + "="*70)
        print(f" RUNNING BENCHMARK: {variant_name} | Model: {self.model_id}")
        print("="*70)

        start_load_time = time.time()
        model, err = self._safe_model_from_pretrained(quant_config, dtype)
        if model is None:
            print(f"[-] Failed to load model for {variant_name}: {err}")
            return {
                "Variant": variant_name,
                "Status": f"FAILED (LOAD: {err})",
                "Speed (tok/s)": 0.0,
                "Peak VRAM (GB)": 0.0,
                "Model VRAM (GB)": 0.0,
                "Load Time (s)": 0.0,
                "Generated Text": f"Error: {err}"
            }

        load_time = round(time.time() - start_load_time, 2)

        # Prepare prompt (support chat template if available)
        messages = [{"role": "user", "content": BENCHMARK_PROMPT}]
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            try:
                formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                formatted_prompt = BENCHMARK_PROMPT
        else:
            formatted_prompt = BENCHMARK_PROMPT

        # Tokenize and move only the tensors we need to CUDA and float16 to avoid bfloat16 casts
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True)
        device = "cuda"
        inputs_cuda: Dict[str, torch.Tensor] = {}
        for k, v in inputs.items():
            # Keep integer tensors as-is; cast floating tensors to float16 to avoid implicit bfloat16->float16 casts
            if v.dtype.is_floating_point:
                inputs_cuda[k] = v.to(device).to(torch.float16)
            else:
                inputs_cuda[k] = v.to(device)

        model_vram = round(torch.cuda.memory_allocated() / (1024 ** 3), 2)

        try:
            # Warm-up short generation to initialize kernels
            warm_inputs = {k: inputs_cuda[k] for k in ("input_ids", "attention_mask") if k in inputs_cuda}
            _ = model.generate(**warm_inputs, max_new_tokens=5, do_sample=False)

            gen_start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **warm_inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            gen_duration = time.time() - gen_start

            input_tokens = inputs_cuda["input_ids"].shape[1]
            total_tokens = outputs[0].shape[0]
            new_tokens = total_tokens - input_tokens

            tokens_per_sec = round(new_tokens / gen_duration, 2) if gen_duration > 0 else 0.0
            peak_vram = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)
            decoded_text = self.tokenizer.decode(outputs[0][input_tokens:], skip_special_tokens=True).strip()

            print(f"[✓] Success -> Speed: {tokens_per_sec} tok/s | Peak VRAM: {peak_vram} GB | Load: {load_time}s")

            result = {
                "Variant": variant_name,
                "Status": "SUCCESS",
                "Speed (tok/s)": tokens_per_sec,
                "Peak VRAM (GB)": peak_vram,
                "Model VRAM (GB)": model_vram,
                "Load Time (s)": load_time,
                "Generated Text": decoded_text
            }

        except torch.cuda.OutOfMemoryError:
            print(f"[-] OOM ERROR during generation in {variant_name}!")
            result = {
                "Variant": variant_name,
                "Status": "FAILED (OOM Gen)",
                "Speed (tok/s)": 0.0,
                "Peak VRAM (GB)": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
                "Model VRAM (GB)": model_vram,
                "Load Time (s)": load_time,
                "Generated Text": "Error: Out of Memory during generation."
            }
        except Exception as e:
            print(f"[-] Generation failed for {variant_name}: {e}")
            result = {
                "Variant": variant_name,
                "Status": f"FAILED (GEN: {e})",
                "Speed (tok/s)": 0.0,
                "Peak VRAM (GB)": 0.0,
                "Model VRAM (GB)": model_vram,
                "Load Time (s)": load_time,
                "Generated Text": f"Error: {e}"
            }
        finally:
            try:
                del model
            except Exception:
                pass
            self._flush_memory()

        return result

    def run_all(self) -> pd.DataFrame:
        results: List[Dict[str, Any]] = []

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        results.append(self.benchmark_variant("4-bit (NF4)", quant_config=nf4_config))

        int8_config = BitsAndBytesConfig(load_in_8bit=True)
        results.append(self.benchmark_variant("8-bit (INT8)", quant_config=int8_config))

        results.append(self.benchmark_variant("16-bit (FP16)", quant_config=None, dtype=torch.float16))

        return pd.DataFrame(results)

# -------------------- Visualization --------------------
def generate_linkedin_chart(df: pd.DataFrame, model_name: str, save_path: str = OUTPUT_PNG_NAME):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), dpi=300)

    colors = ['#00FFC6', '#3B82F6', '#EF4444']
    variants = df['Variant'].tolist()

    speeds = df.get('Speed (tok/s)', pd.Series([0]*len(variants))).tolist()
    bars1 = axes[0].bar(variants, speeds, color=colors, width=0.45, edgecolor='white', linewidth=0.8)
    axes[0].set_title('Inference Speed (Tokens/sec)\n[Higher is Better]', fontsize=11, fontweight='bold', pad=12, color='#00FFC6')
    axes[0].set_ylabel('Tokens / Second', fontsize=10, fontweight='bold')
    axes[0].grid(axis='y', linestyle='--', alpha=0.25)
    max_sp = max(speeds) if any(speeds) else 1.0
    for bar in bars1:
        yval = bar.get_height()
        lbl = f"{yval:.2f}" if yval > 0 else "N/A"
        axes[0].text(bar.get_x() + bar.get_width()/2, yval + (max_sp * 0.03), lbl, ha='center', va='bottom', fontsize=10, fontweight='bold', color='white')

    vram = df.get('Peak VRAM (GB)', pd.Series([0]*len(variants))).tolist()
    bars2 = axes[1].bar(variants, vram, color=colors, width=0.45, edgecolor='white', linewidth=0.8)
    axes[1].set_title('Peak VRAM Usage (GB)\n[Lower is Better]', fontsize=11, fontweight='bold', pad=12, color='#3B82F6')
    axes[1].set_ylabel('Memory (GB)', fontsize=10, fontweight='bold')
    axes[1].grid(axis='y', linestyle='--', alpha=0.25)
    max_vr = max(vram) if any(vram) else 1.0
    for bar in bars2:
        yval = bar.get_height()
        lbl = f"{yval:.2f} GB" if yval > 0 else "N/A"
        axes[1].text(bar.get_x() + bar.get_width()/2, yval + (max_vr * 0.03), lbl, ha='center', va='bottom', fontsize=10, fontweight='bold', color='white')

    load_times = df.get('Load Time (s)', pd.Series([0]*len(variants))).tolist()
    bars3 = axes[2].bar(variants, load_times, color=colors, width=0.45, edgecolor='white', linewidth=0.8)
    axes[2].set_title('Model Load Time (Seconds)\n[Lower is Better]', fontsize=11, fontweight='bold', pad=12, color='#F59E0B')
    axes[2].set_ylabel('Seconds', fontsize=10, fontweight='bold')
    axes[2].grid(axis='y', linestyle='--', alpha=0.25)
    max_lt = max(load_times) if any(load_times) else 1.0
    for bar in bars3:
        yval = bar.get_height()
        lbl = f"{yval:.1f}s" if yval > 0 else "N/A"
        axes[2].text(bar.get_x() + bar.get_width()/2, yval + (max_lt * 0.03), lbl, ha='center', va='bottom', fontsize=10, fontweight='bold', color='white')

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU"
    fig.suptitle(f'LLM Quantization Benchmark | Model: {model_name}\nHardware: {gpu_name}', fontsize=14, fontweight='bold', y=1.04, color='#F3F4F6')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[+] High-Resolution Chart PNG saved to: '{save_path}'")
    try:
        plt.show()
    except Exception:
        pass

# -------------------- Main --------------------
def main():
    print("="*70)
    print(" STARTING REFINED BENCHMARK SUITE & VISUALIZER")
    print("="*70)

    hf_token = resolve_hf_token()
    benchmarker = UniversalLLMBenchmarker(preferred_model=PREFERRED_MODEL, fallback_candidates=FALLBACK_CANDIDATES, token=hf_token)
    summary_df = benchmarker.run_all()

    print("\n" + "="*70)
    print(" SUMMARY TABLE")
    print("="*70)
    cols = ["Variant", "Status", "Speed (tok/s)", "Peak VRAM (GB)", "Model VRAM (GB)", "Load Time (s)"]
    for c in cols:
        if c not in summary_df.columns:
            summary_df[c] = None
    print(summary_df[cols].to_string(index=False))

    generate_linkedin_chart(summary_df, model_name=benchmarker.model_id, save_path=OUTPUT_PNG_NAME)

if __name__ == "__main__":
    main()
