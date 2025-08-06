import os
from argparse import Namespace
from pathlib import Path
import numpy as np

from libero.libero.utils.task_generation_utils import register_task_info, generate_bddl_from_task_info
from create_spacial_bddl import create_and_register_scene

# generate_spacial_bddl_batch.py
import os
from argparse import Namespace
from pathlib import Path
import numpy as np


OUTPUT_ROOT = "libero/libero/bddl_files/spatial"

# Each dict represents one environment configuration; "copies" specifies how many variants to generate for this config
CONFIGS = [
    {
        "task_language": "Reach the white bowl on the kitchen table",
        "target_object": "white_bowl",
        "target_position": (0.0, 0.0),
        "ring_regions": [(0.025, 0.25), (0.25, 0.30)],
        "distractor_counts": [1, 1],
        "distractor_objects": [["butter"], ["ketchup"]],
        "copies": 2, 
    },
    {
        "task_language": "Reach the white bowl on the kitchen table",
        "target_object": "white_bowl",
        "target_position": (0.0, 0.0),
        "ring_regions": [(0.025, 0.25), (0.25, 0.30)],
        "distractor_counts": [2, 3],
        "distractor_objects": [["butter", "popcorn"], ["ketchup", "milk", "cookies"]],
        "copies": 3,  
    },
]

def ensure_tuple_list(x):
    return [tuple(t) for t in x]

def fmt_float(x, nd=3):
    return f"{float(x):.{nd}f}"

def config_folder_name(distractor_counts, ring_regions):

    counts_str = "_".join(str(int(c)) for c in distractor_counts)
    rings_parts = []
    for (rmin, rmax) in ring_regions:
        rings_parts.append(f"{fmt_float(rmin)}-{fmt_float(rmax)}")
    rings_str = "__".join(rings_parts)
    return f"counts-{counts_str}__rings-{rings_str}"

def make_args(scene_name_fixed: str,
              task_language: str,
              target_object: str,
              target_position,
              ring_regions,
              distractor_counts,
              distractor_objects,
              folder: str):
    """
    Construct a Namespace compatible with the original argparse setup
    so we can directly reuse create_and_register_scene(args).
    """
    goal_object = f"{target_object}_1" 

    return Namespace(
        scene_name='spacial_scene',  
        goal_object=goal_object,
        goal_region=f"kitchen_table_{target_object}_init_region",  
        folder=folder,
        task_language=task_language,
        target_object=target_object,
        target_position=tuple(target_position),
        ring_regions=ensure_tuple_list(ring_regions),
        distractor_counts=list(distractor_counts),
        distractor_objects=[list(lst) for lst in distractor_objects],
    )

def main():
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    all_generated = []
    all_failures = []


    SCENE_NAME_FIXED = "spacial_scene"

    for ci, cfg in enumerate(CONFIGS):
         # Use distractor_counts and ring_regions to name the configuration folder
        cfg_folder_name = config_folder_name(
            cfg["distractor_counts"],
            ensure_tuple_list(cfg["ring_regions"])
        )
        base_folder = Path(OUTPUT_ROOT) / cfg_folder_name
        base_folder.mkdir(parents=True, exist_ok=True)

        copies = int(cfg.get("copies", 1))
        for vi in range(copies):
            # Use a different random seed for each variant to randomize distractor positions
            seed = (ci + 1) * 10_000 + vi
            np.random.seed(seed)

            out_folder = base_folder / f"v{vi}"

            args = make_args(
                scene_name_fixed=SCENE_NAME_FIXED,
                task_language=cfg["task_language"],
                target_object=cfg["target_object"],
                target_position=cfg["target_position"],
                ring_regions=cfg["ring_regions"],
                distractor_counts=cfg["distractor_counts"],
                distractor_objects=cfg["distractor_objects"],
                folder=str(out_folder)
            )

            # 1) 注册场景类（内部使用 @register_mu(scene_type=args.scene_name)）
            _SceneClass = create_and_register_scene(args)

            # 2) 注册任务（对象关注 + 目标）
            # 默认使用 reached 目标（与原脚本一致）
            register_task_info(
                language=args.task_language,
                scene_name=args.scene_name,
                objects_of_interest=[args.goal_object],
                goal_states=[("reached", f"{cfg['target_object']}_1")],
            )
            # 如需切为 "On" 语义目标，改为：
            # register_task_info(
            #     language=args.task_language,
            #     scene_name=args.scene_name,
            #     objects_of_interest=[args.goal_object],
            #     goal_states=[("On", args.goal_object, args.goal_region)],
            # )

            # 3) 生成 BDDL（写到该配置专属子目录）
            bddl_file_names, failures = generate_bddl_from_task_info(folder=str(out_folder))

            print(f"[CFG {cfg_folder_name} v{vi}] seed={seed}")
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
