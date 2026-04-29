import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download

# quantization changes storage/compute format (dtype/packing) not architecture, so we 
# keep base configs and quantized weights as separate downloadable components.

# btw, if some fail, you have to make a HF_TOKEN and export it in your environment
# and go to the HF website and ask for permission to download the model
MODEL_SPECS = [
    {
        "name": "sd15",
        "kind": "trainable_full",
        "base_repo": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    },
    {
        "name": "sdxl",
        "kind": "trainable_full",
        "base_repo": "stabilityai/stable-diffusion-xl-base-1.0",
    },
    # {
    #     "name": "sd3_medium",
    #     "kind": "trainable_full",
    #     "base_repo": "stabilityai/stable-diffusion-3-medium-diffusers",
    #     "reason": "larger training footprint; not default for lean consumer-VRAM runs",
    # },
    {
        "name": "flux2_klein_base_4b",
        "kind": "trainable_full",
        "base_repo": "black-forest-labs/FLUX.2-klein-base-4B",
        "reason": "large base for full training; keep disabled by default",
    },
    # {
    #     "name": "flux2_klein_4b_fp8_diffusers",
    #     "kind": "quantized_with_base_config",
    #     "base_repo": "black-forest-labs/FLUX.2-klein-4B",
    #     "quant_repo": "Photoroom/FLUX.2-klein-4b-fp8-diffusers",
    #     "reason": "quantized inference artifact; not a primary training base",
    # },
    # {
    #     "name": "sd3_medium_gguf",
    #     "kind": "quantized_with_base_config",
    #     "base_repo": "stabilityai/stable-diffusion-3-medium-diffusers",
    #     "quant_repo": "second-state/stable-diffusion-3-medium-GGUF",
    #     "reason": "GGUF quantized weights are inference-oriented",
    # },
    # {
    #     "name": "sd35_large_gguf",
    #     "kind": "quantized_with_base_config",
    #     "base_repo": "stabilityai/stable-diffusion-3.5-large",
    #     "quant_repo": "city96/stable-diffusion-3.5-large-gguf",
    #     "reason": "quantized + large; inference-oriented and heavy for training",
    # },
]

# Candidate repos kept for reference but not active by default:
# "black-forest-labs/FLUX.2-klein-4B"
# "black-forest-labs/FLUX.2-dev"
# "stabilityai/stable-diffusion-3.5-large"
# "black-forest-labs/FLUX.2-dev-NVFP4"  # too large / inference artifact
# "black-forest-labs/FLUX.2-klein-9b-kv-fp8"  # inference-oriented FP8/KV artifact

DOWNLOAD_MARKER = ".download_complete.json"
FULL_IGNORE_PATTERNS = ["*.ckpt", "*.h5", "*.msgpack", "onnx/*", "flax/*", "*.onnx", "*.tflite"]
STRUCTURE_ALLOW_PATTERNS = [
    "model_index.json",
    "*.json",
    "**/*.json",
    "tokenizer/**",
    "tokenizer_2/**",
    "tokenizer_3/**",
    "scheduler/**",
    "feature_extractor/**",
    "preprocessor_config.json",
    "README.md",
    "LICENSE*",
]
QUANT_ALLOW_PATTERNS = ["*.gguf", "*.safetensors", "*.json", "README.md", "LICENSE*"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download candidate base models from Hugging Face."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--trainable-only",
        action="store_true",
        help="Download only trainable_full specs.",
    )
    mode.add_argument(
        "--quant-only",
        action="store_true",
        help="Download only quantized_with_base_config specs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even when validation markers exist.",
    )
    return parser.parse_args()


def prompt_path(default_path: Path) -> Path:
    raw = input(f"Model storage directory [{default_path}]: ").strip()
    selected = Path(raw).expanduser() if raw else default_path
    return selected.resolve()


def ensure_output_dir(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Output path exists but is not a directory: {path}")
        return

    choice = input(f"Create directory '{path}'? [Y/n]: ").strip().lower()
    if choice in {"", "y", "yes"}:
        path.mkdir(parents=True, exist_ok=True)
        return
    raise ValueError("Output directory does not exist and creation was declined.")


def marker_path(download_dir: Path) -> Path:
    return download_dir / DOWNLOAD_MARKER


def path_has_match(root: Path, pattern: str) -> bool:
    return any(root.rglob(pattern))


def validate_download(download_dir: Path, kind: str, component: str) -> bool:
    if not download_dir.exists() or not download_dir.is_dir():
        return False
    if not any(download_dir.iterdir()):
        return False

    if kind == "trainable_full" and component == "full":
        has_model_index = (download_dir / "model_index.json").exists()
        has_config = path_has_match(download_dir, "config.json")
        has_weights = path_has_match(download_dir, "*.safetensors") or path_has_match(download_dir, "*.bin")
        return has_model_index and has_config and has_weights

    if kind == "quantized_with_base_config" and component == "structure":
        has_model_index = (download_dir / "model_index.json").exists()
        has_config = path_has_match(download_dir, "config.json")
        return has_model_index and has_config

    if kind == "quantized_with_base_config" and component == "quant_weights":
        return path_has_match(download_dir, "*.gguf") or path_has_match(download_dir, "*.safetensors")

    return False


def is_download_complete(
    download_dir: Path,
    kind: str,
    component: str,
) -> bool:
    marker = marker_path(download_dir)
    if not marker.exists():
        return False
    return validate_download(download_dir, kind, component)


def write_download_marker(
    download_dir: Path,
    model_name: str,
    repo_id: str,
    kind: str,
    component: str,
) -> None:
    payload = {
        "model_name": model_name,
        "repo_id": repo_id,
        "kind": kind,
        "component": component,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    marker_path(download_dir).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_output_dir() -> Path:
    default = (Path(__file__).resolve().parents[2] / "base_models").resolve()
    return prompt_path(default)


def migrate_existing_repo_folder(base_dir: Path, repo_id: str, destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        return
    legacy = base_dir / repo_id.replace("/", "_")
    if not legacy.exists() or legacy.resolve() == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.move(str(legacy), str(destination))
    print(f"[MIGRATE] {legacy} -> {destination}")


def download_full_model(repo_id: str, output_path: Path, token: str | None) -> None:
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(output_path),
        token=token,
        ignore_patterns=FULL_IGNORE_PATTERNS,
    )


def download_structure_only(repo_id: str, output_path: Path, token: str | None) -> None:
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(output_path),
        token=token,
        allow_patterns=STRUCTURE_ALLOW_PATTERNS,
    )


def download_quant_weights(repo_id: str, output_path: Path, token: str | None) -> None:
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(output_path),
        token=token,
        allow_patterns=QUANT_ALLOW_PATTERNS,
    )


def filtered_specs(args: argparse.Namespace) -> list[dict]:
    if args.trainable_only:
        return [spec for spec in MODEL_SPECS if spec["kind"] == "trainable_full"]
    if args.quant_only:
        return [spec for spec in MODEL_SPECS if spec["kind"] == "quantized_with_base_config"]
    return MODEL_SPECS


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    output_dir = resolve_output_dir()
    ensure_output_dir(output_dir)

    print(f"Base output directory: {output_dir}")

    counters = {
        "full": {"downloaded": 0, "skipped": 0, "failed": 0},
        "structure": {"downloaded": 0, "skipped": 0, "failed": 0},
        "quant_weights": {"downloaded": 0, "skipped": 0, "failed": 0},
    }
    failures: list[dict] = []

    for spec in filtered_specs(args):
        model_name = spec["name"]
        kind = spec["kind"]
        model_root = output_dir / model_name

        def process_component(component: str, repo_id: str, bucket: str, downloader) -> None:
            component_dir = model_root / component
            migrate_existing_repo_folder(output_dir, repo_id, component_dir)
            component_dir.mkdir(parents=True, exist_ok=True)

            if not args.force and is_download_complete(component_dir, kind, component):
                print(f"[SKIP] {model_name}/{component} <- {repo_id}")
                counters[bucket]["skipped"] += 1
                return

            print(f"[DL]   {model_name}/{component} <- {repo_id}")
            try:
                if args.force and component_dir.exists():
                    shutil.rmtree(component_dir)
                    component_dir.mkdir(parents=True, exist_ok=True)
                downloader(repo_id, component_dir, token)
                if not validate_download(component_dir, kind, component):
                    raise ValueError("validation failed after download")
                write_download_marker(component_dir, model_name, repo_id, kind, component)
                counters[bucket]["downloaded"] += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[FAIL] {model_name}/{component} <- {repo_id}: {exc}")
                counters[bucket]["failed"] += 1
                failures.append(
                    {
                        "model_name": model_name,
                        "component": component,
                        "repo_id": repo_id,
                        "error": str(exc),
                    }
                )

        if kind == "trainable_full":
            process_component("full", spec["base_repo"], "full", download_full_model)
        elif kind == "quantized_with_base_config":
            process_component("structure", spec["base_repo"], "structure", download_structure_only)
            process_component("quant_weights", spec["quant_repo"], "quant_weights", download_quant_weights)

    print("\nDownload summary")
    print("- Full models:")
    print(f"  - Downloaded: {counters['full']['downloaded']}")
    print(f"  - Skipped: {counters['full']['skipped']}")
    print(f"  - Failed: {counters['full']['failed']}")
    print("- Structures:")
    print(f"  - Downloaded: {counters['structure']['downloaded']}")
    print(f"  - Skipped: {counters['structure']['skipped']}")
    print(f"  - Failed: {counters['structure']['failed']}")
    print("- Quant weights:")
    print(f"  - Downloaded: {counters['quant_weights']['downloaded']}")
    print(f"  - Skipped: {counters['quant_weights']['skipped']}")
    print(f"  - Failed: {counters['quant_weights']['failed']}")
    if failures:
        print("\nFailed:")
        for item in failures:
            print(
                f"- {item['model_name']} / {item['component']} / {item['repo_id']}:\n"
                f"  {item['error']}"
            )


if __name__ == "__main__":
    main()
