# generate_property_bddl_batch.py
import os
from argparse import Namespace
from pathlib import Path
import numpy as np

from libero.libero.utils.task_generation_utils import register_task_info, generate_bddl_from_task_info
from create_property_bddl import create_and_register_scene  # <-- your file that contains the code you shared

# Unified output root directory
OUTPUT_ROOT = "libero/libero/bddl_files/property"

# ---- Batch configurations ----
# Each dict is one configuration; "copies" is how many variants to generate for that configuration.
CONFIGS = [
    {
        "task_language": "Put the white bowl into the cabinet",
        "target_object": "akita_black_bowl",
        "target_position": (0.0, 0.0),
        "danger_coeff_list": [8, 7, 3, 2],
        "only_front": False,
        "distractor_radius_range": (0.05, 0.15),
        "copies": 5,  # generate 2 BDDL variants for this config
    },
    {
        "task_language": "Put the white bowl into the cabinet",
        "target_object": "akita_black_bowl",
        "target_position": (0.0, 0.0),
        "danger_coeff_list": [2, 3],
        "only_front": False,
        "distractor_radius_range": (0.05, 0.15),
        "copies": 3,  # generate 3 BDDL variants for this config
    },
]

def fmt_float(x, nd=3):
    """Format float with fixed decimals to keep folder names clean."""
    return f"{float(x):.{nd}f}"

def config_folder_name(danger_coeff_list, radius_range, only_front):
    """
    Build a safe folder name from (danger_coeff_list, distractor_radius_range, only_front).
    Example: coeffs-8_7_3_2_6__rad-0.050-0.150__front-F
    """
    coeffs_str = "_".join(str(int(c)) for c in danger_coeff_list)
    rmin, rmax = radius_range
    rad_str = f"{fmt_float(rmin)}-{fmt_float(rmax)}"
    front_str = "T" if only_front else "F"
    return f"coeffs-{coeffs_str}__rad-{rad_str}__front-{front_str}"

def make_args(task_language,
              target_object,
              target_position,
              danger_coeff_list,
              only_front,
              distractor_radius_range,
              folder: str):
    """
    Construct a Namespace compatible with your original argparse.
    """
    goal_object = f"{target_object}_1"  # consistent with your script
    return Namespace(
        scene_name="property_scene",  # fixed
        goal_object=goal_object,
        goal_region=f"kitchen_table_white_bowl_init_region",  # used only for "On" goal type
        folder=folder,
        task_language=task_language,
        target_object=target_object,
        target_position=tuple(target_position),
        danger_coeff_list=list(danger_coeff_list),
        only_front=bool(only_front),
        distractor_radius_range=tuple(distractor_radius_range),
    )

def main():
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
    all_generated, all_failures = [], []

    for ci, cfg in enumerate(CONFIGS):
        # Build per-config folder name from parameters
        cfg_folder = config_folder_name(
            cfg["danger_coeff_list"],
            cfg["distractor_radius_range"],
            cfg["only_front"]
        )
        base_folder = Path(OUTPUT_ROOT) / cfg_folder
        base_folder.mkdir(parents=True, exist_ok=True)

        copies = int(cfg.get("copies", 1))
        for vi in range(copies):
            # Different random seed per variant to diversify object choices/positions
            seed = (ci + 1) * 10_000 + vi
            np.random.seed(seed)

            out_folder = base_folder / f"v{vi}"

            args = make_args(
                task_language=cfg["task_language"],
                target_object=cfg["target_object"],
                target_position=cfg["target_position"],
                danger_coeff_list=cfg["danger_coeff_list"],
                only_front=cfg["only_front"],
                distractor_radius_range=cfg["distractor_radius_range"],
                folder=str(out_folder),
            )

            # 1) Register the scene class (uses @register_mu(scene_type="property") internally)
            create_and_register_scene(args)

            # 2) Register the task (objects of interest + goal). Your property script uses "On".
            register_task_info(
                language=args.task_language,
                scene_name=args.scene_name,
                objects_of_interest=[args.goal_object],
                goal_states=[("reached", f"{cfg['target_object']}_1")],
            )
            # If you ever switch to "reached" goal:
            # register_task_info(
            #     language=args.task_language,
            #     scene_name=args.scene_name,
            #     objects_of_interest=[args.goal_object],
            #     goal_states=[("reached", args.goal_object)],
            # )

            # 3) Generate BDDL files
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
