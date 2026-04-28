import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download

# Candidate repos under consideration for experiments.
REPO_IDS = [
    # "black-forest-labs/FLUX.2-klein-4B",
    # "stabilityai/stable-diffusion-3-medium-diffusers",
    # "black-forest-labs/FLUX.2-dev",
    # "stabilityai/stable-diffusion-3.5-large",
    # "stabilityai/stable-diffusion-xl-base-1.0",
    # "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "second-state/stable-diffusion-3-medium-GGUF",
    "city96/stable-diffusion-3.5-large-gguf",
    "black-forest-labs/FLUX.2-dev-NVFP4",
    "black-forest-labs/FLUX.2-klein-9b-kv-fp8",
]

DOWNLOAD_MARKER = ".download_complete.json"
IGNORE_PATTERNS = ["*.bin", "*.ckpt", "*.h5", "*.msgpack", "*.bin.index.json"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download candidate base models from Hugging Face."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base directory where model folders will be stored.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token. Defaults to HF_TOKEN env var.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Non-interactive mode: use defaults without prompts.",
    )
    return parser.parse_args()


def prompt_path(default_path: Path) -> Path:
    raw = input(f"Model storage directory [{default_path}]: ").strip()
    selected = Path(raw).expanduser() if raw else default_path
    return selected.resolve()


def ensure_output_dir(path: Path, non_interactive: bool) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Output path exists but is not a directory: {path}")
        return

    if non_interactive:
        path.mkdir(parents=True, exist_ok=True)
        return

    choice = input(f"Create directory '{path}'? [Y/n]: ").strip().lower()
    if choice in {"", "y", "yes"}:
        path.mkdir(parents=True, exist_ok=True)
        return
    raise ValueError("Output directory does not exist and creation was declined.")


def model_dir(base_dir: Path, repo_id: str) -> Path:
    return base_dir / repo_id.replace("/", "_")


def marker_path(download_dir: Path) -> Path:
    return download_dir / DOWNLOAD_MARKER


def is_download_complete(download_dir: Path) -> bool:
    marker = marker_path(download_dir)
    return marker.exists() and any(download_dir.iterdir())


def write_download_marker(download_dir: Path, repo_id: str) -> None:
    payload = {
        "repo_id": repo_id,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    marker_path(download_dir).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_output_dir(args: argparse.Namespace) -> Path:
    default = (Path(__file__).resolve().parents[2] / "base_models").resolve()
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    if args.yes:
        return default
    return prompt_path(default)


def main() -> None:
    args = parse_args()
    token = args.token or os.environ.get("HF_TOKEN")
    output_dir = resolve_output_dir(args)
    ensure_output_dir(output_dir, non_interactive=args.yes)

    print(f"Base output directory: {output_dir}")

    skipped = []
    downloaded = []
    failed = []

    for repo_id in REPO_IDS:
        download_dir = model_dir(output_dir, repo_id)
        download_dir.mkdir(parents=True, exist_ok=True)

        if is_download_complete(download_dir):
            print(f"[SKIP] {repo_id} -> {download_dir} (already complete)")
            skipped.append(repo_id)
            continue

        print(f"[DL]   {repo_id} -> {download_dir}")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(download_dir),
                ignore_patterns=IGNORE_PATTERNS,
                token=token,
                resume_download=True,
            )
            write_download_marker(download_dir, repo_id)
            downloaded.append(repo_id)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {repo_id}: {exc}")
            failed.append(repo_id)

    print("\nDownload summary")
    print(f"- Downloaded: {len(downloaded)}")
    print(f"- Skipped:    {len(skipped)}")
    print(f"- Failed:     {len(failed)}")
    if failed:
        print("- Failed repos:")
        for repo_id in failed:
            print(f"  - {repo_id}")


if __name__ == "__main__":
    main()
