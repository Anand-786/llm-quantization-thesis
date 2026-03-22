"""
Result saving and logging utilities.
All experiments save results as JSON for easy comparison later.
"""

import json
import os
from datetime import datetime


def save_result(result: dict, save_dir: str, filename: str = None):
    """
    Save an experiment result dict as a JSON file.
    
    Args:
        result: Dictionary with experiment results.
        save_dir: Directory to save in (created if missing).
        filename: Optional filename. Auto-generated if None.
    
    Returns:
        Path to saved file.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Add metadata
    result["_saved_at"] = datetime.now().isoformat()
    
    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = result.get("model", "unknown").split("/")[-1]
        task = result.get("task", "unknown")
        filename = f"{task}_{model_short}_{ts}.json"
    
    if not filename.endswith(".json"):
        filename += ".json"
    
    path = os.path.join(save_dir, filename)
    
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Result saved → {path}")
    return path


def load_all_results(results_dir: str) -> list:
    """Load all JSON result files from a directory."""
    results = []
    if not os.path.exists(results_dir):
        return results
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(results_dir, fname)) as f:
                results.append(json.load(f))
    return results


def save_to_drive(result: dict, drive_path: str, filename: str = None):
    """
    Save result to Google Drive (for Colab use).
    Drive should be mounted at /content/drive/MyDrive/.
    
    Args:
        result: Result dictionary.
        drive_path: Path under MyDrive, e.g. 'thesis_results/task01'.
        filename: Optional filename.
    """
    full_path = os.path.join("/content/drive/MyDrive", drive_path)
    return save_result(result, full_path, filename)


def print_result_summary(result: dict):
    """Pretty-print a result dict."""
    print("\n" + "=" * 60)
    print(f"  Task:   {result.get('task', '?')}")
    print(f"  Model:  {result.get('model', '?')}")
    print(f"  Scheme: {result.get('scheme', '?')}")
    print(f"  Alpha:  {result.get('alpha', '?')}")
    print("-" * 60)
    
    metrics = result.get("metrics", {})
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:.<30s} {v:.4f}")
        else:
            print(f"  {k:.<30s} {v}")
    
    avg = result.get("avg_zeroshot_acc")
    if avg is not None:
        print("-" * 60)
        print(f"  {'AVG ZERO-SHOT ACC':.<30s} {avg:.4f}")
    
    ppl = result.get("wikitext2_ppl")
    if ppl is not None:
        print(f"  {'WIKITEXT-2 PPL':.<30s} {ppl:.2f}")
    
    print("=" * 60 + "\n")
