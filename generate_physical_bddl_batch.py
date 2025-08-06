# generate_physical_bddl_batch.py
import os
from argparse import Namespace
from pathlib import Path
import numpy as np

from libero.libero.utils.task_generation_utils import register_task_info, generate_bddl_from_task_info
from create_comb_bddl import create_and_register_scene  # <-- adjust if your module name differs

# Unified output root directory for physical-dense scenes
OUTPUT_ROOT = "libero/libero/bddl_files/physical"

# ---- Batch configurations ----
# Each dict is one configuration; "copies" is how many variants to generate
CONFIGS = [
    {
        "task_language": "Reach the white bowl",
        "target_object": "akita_black_bowl",
        "target_position": (0.0, 0.0),
        "region_range": (0.10, 0.30),
        "object_pairs": [("white_cabinet", "milk"), ("none", "ketchup"), ("wooden_shelf", "none")],
        "copies": 6,  # generate 2 BDDL variants for this config
    },
    {
        "task_language": "Reach the white bowl",
        "target_object": "akita_black_bowl",
        "target_position": (0.0, 0.0),
        "region_range": (0.10, 0.30),
        "object_pairs": [("none", "ketchup"), ("wooden_shelf", "milk")],
        "copies": 3,  # generate 3 BDDL variants for this config
    },
]

def fmt_float(x, nd=3):
    """Format float with fixed decimals to keep folder names clean."""
    return f"{float(x):.{nd}f}"

def config_folder_name(object_pairs, region_range):
    """
    Build a safe folder name from (object_pairs, region_range).
    Example:
      pairs-white_cabinet-milk__none-ketchup__wooden_shelf-none__rad-0.100-0.300
    """
    pair_tokens = []
    for imm, mov in object_pairs:
        imm_s = (imm or "none")
        mov_s = (mov or "none")
        pair_tokens.append(f"{imm_s}-{mov_s}")
    pairs_str = "__".join(pair_tokens)

    rmin, rmax = region_range
    rad_str = f"{fmt_float(rmin)}-{fmt_float(rmax)}"
    return f"pairs-{pairs_str}__rad-{rad_str}"

def ensure_tuple_list(x):
    """Ensure object_pairs and similar lists contain tuples."""
    return [tuple(t) for t in x]

def make_args(task_language,
              target_object,
              target_position,
              region_range,
              object_pairs,
              folder: str):
    """
    Construct a Namespace compatible with your original argparse.
    """
    goal_object = f"{target_object}_1"  # consistent with your script
    return Namespace(
        scene_name="comb_scene",  # fixed (matches your script)
        goal_object=goal_object,
        goal_region=f"kitchen_table_{target_object}_init_region",  # used for "On" goal type
        folder=folder,
        task_language=task_language,
        target_object=target_object,
        target_position=tuple(target_position),
        region_range=tuple(region_range),
        object_pairs=ensure_tuple_list(object_pairs),
    )

def main():
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
    all_generated, all_failures = [], []

    for ci, cfg in enumerate(CONFIGS):
        # Name config folder from object_pairs + region_range
        cfg_folder = config_folder_name(
            ensure_tuple_list(cfg["object_pairs"]),
            cfg["region_range"]
        )
        base_folder = Path(OUTPUT_ROOT) / cfg_folder
        base_folder.mkdir(parents=True, exist_ok=True)

        copies = int(cfg.get("copies", 1))
        for vi in range(copies):
            # Different random seed per variant to diversify placements
            seed = (ci + 1) * 10_000 + vi
            np.random.seed(seed)

            out_folder = base_folder / f"v{vi}"

            args = make_args(
                task_language=cfg["task_language"],
                target_object=cfg["target_object"],
                target_position=cfg["target_position"],
                region_range=cfg["region_range"],
                object_pairs=cfg["object_pairs"],
                folder=str(out_folder),
            )

            # 1) Register the scene class (uses @register_mu(scene_type="comb") internally)
            create_and_register_scene(args)

            # 2) Register the task (On target, matching your main script)
            register_task_info(
                language=args.task_language,
                scene_name=args.scene_name,
                objects_of_interest=[args.goal_object],
                goal_states=[("reached", f"{cfg['target_object']}_1")],
            )

            # 3) Generate BDDL files for this variant
            bddl_file_names, failures = generate_bddl_from_task_info(folder=str(out_folder))

            print(f"[CFG {cfg_folder} v{vi}] seed={seed}")
            print("  Generated:", bddl_file_names)
            print("  Failures :", failures)

            all_generated.extend(bddl_file_names or [])
            all_failures.extend(failures or [])

    print("\n=== SUMMARY ===")
    print("Total generated:", len(all_generated))
    if all_generated:
        for f in all_generated:
            print("  +", f)
    if all_failures:
        print("Failures:")
        for f in all_failures:
            print("  -", f)
    else:
        print("No failures.")

if __name__ == "__main__":
    main()
