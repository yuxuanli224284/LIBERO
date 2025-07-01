import numpy as np
from libero.libero.utils.mu_utils import InitialSceneTemplates, register_mu
from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info
from libero.libero.utils.task_generation_utils import register_task_info, generate_bddl_from_task_info

class SpacialSceneDynamic(InitialSceneTemplates):
    def __init__(self, target_object, target_position, ring_regions, distractor_counts, distractor_objects, workspace_name="kitchen_table"):
        self.workspace_name = workspace_name
        self.target_object = target_object
        self.target_position = target_position
        self.ring_regions = ring_regions
        self.distractor_counts = distractor_counts
        self.distractor_objects = distractor_objects

        fixture_num_info = {"kitchen_table": 1}
        object_num_info = {target_object: 1}
        for region_objs in distractor_objects:
            for obj in region_objs:
                object_num_info[obj] = object_num_info.get(obj, 0) + 1

        super().__init__(workspace_name, fixture_num_info, object_num_info)

    def define_regions(self):
        self.regions.update(self.get_region_dict(
            region_centroid_xy=self.target_position,
            region_name=f"{self.target_object}_init_region",
            # region_name="target_init_region",
            target_name=self.workspace_name,
            region_half_len=0.025
        ))

        for i, (r_min, r_max) in enumerate(self.ring_regions):
            for j in range(self.distractor_counts[i]):
                angle = np.random.uniform(0, 2*np.pi)
                radius = np.random.uniform(r_min, r_max)
                pos = self.target_position + radius * np.array([np.cos(angle), np.sin(angle)])
                pos = pos.tolist()
                obj_name = self.distractor_objects[i][j]
                region_name = f"{obj_name}_{i}_{j}_region"
                self.regions.update(self.get_region_dict(
                    region_centroid_xy=pos,
                    region_name=region_name,
                    target_name=self.workspace_name,
                    region_half_len=0.001
                ))

        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    @property
    def init_states(self):
        states = [("On", f"{self.target_object}_1", f"{self.workspace_name}_{self.target_object}_init_region")]
        for i, objs in enumerate(self.distractor_objects):
            for j, obj in enumerate(objs):
                region_name = f"{obj}_{i}_{j}_region"
                states.append(("On", f"{obj}_1", f"{self.workspace_name}_{region_name}"))
        return states


@register_mu(scene_type="spacial")

class SpacialScene(SpacialSceneDynamic):
    def __init__(self):
        super().__init__(
            target_object="white_bowl",
            target_position=(0.0, 0.0),
            ring_regions=[(0.025, 0.25), (0.25, 0.30)],
            distractor_counts=[2, 3],
            distractor_objects=[["butter", "popcorn"], ["ketchup", "milk", "cookies"]]
        )
scene_name = "spacial_scene"

register_task_info("Put the white bowl into the cabinet",
                   scene_name=scene_name,
                   objects_of_interest=["white_bowl_1"],
                   goal_states=[("In", "white_bowl_1", "kitchen_table_white_bowl_init_region")])

bddl_file_names, failures = generate_bddl_from_task_info(folder="/home/lyx/LIBERO/libero/libero/bddl_files/libero_spatial")

print("Generated:", bddl_file_names)
print("Failures:", failures)

