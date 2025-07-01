from libero.libero.envs.objects import get_object_dict, get_object_fn
from libero.libero.envs.predicates import get_predicate_fn_dict, get_predicate_fn
import numpy as np
from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info
from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
from libero.libero.utils.task_generation_utils import register_task_info, get_task_info, generate_bddl_from_task_info

@register_mu(scene_type="test")
class TestScene1(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "kitchen_table": 1,
            "wooden_cabinet": 1,
        }

        object_num_info = {
            "akita_black_bowl": 1,
            "plate": 1,
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(region_centroid_xy=[0.0, -0.30], 
                                 region_name="wooden_cabinet_init_region", 
                                 target_name=self.workspace_name, 
                                 region_half_len=0.01,
                                 yaw_rotation=(np.pi, np.pi))
        )

        self.regions.update(
            self.get_region_dict(region_centroid_xy=[0., 0.0], 
                                 region_name="akita_black_bowl_init_region", 
                                 target_name=self.workspace_name, 
                                 region_half_len=0.025)
        )

        self.regions.update(
            self.get_region_dict(region_centroid_xy=[0.0, 0.25], 
                                 region_name="plate_init_region", 
                                 target_name=self.workspace_name, 
                                 region_half_len=0.025)
        )
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    @property
    def init_states(self):
        states = [
            ("On", "akita_black_bowl_1", "kitchen_table_akita_black_bowl_init_region"),
            ("On", "plate_1", "kitchen_table_plate_init_region"),
            ("On", "wooden_cabinet_1", "kitchen_table_wooden_cabinet_init_region")]
        return states

@register_mu(scene_type="test")
class TestScene2(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "kitchen_table": 1,
            # "wooden_cabinet": 1,
        }

        object_num_info = {
            "akita_black_bowl": 1,
            "plate": 2,
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(region_centroid_xy=[0.0, -0.25], 
                                 region_name="wooden_cabinet_init_region", 
                                 target_name=self.workspace_name, 
                                 region_half_len=0.001,
                                 yaw_rotation=(np.pi, np.pi))
        )

        self.regions.update(
            self.get_region_dict(region_centroid_xy=[0., 0.0], 
                                 region_name="akita_black_bowl_init_region", 
                                 target_name=self.workspace_name, 
                                 region_half_len=0.025)
        )

        self.regions.update(
            self.get_region_dict(region_centroid_xy=[0.0, 0.25], 
                                 region_name="plate_init_region", 
                                 target_name=self.workspace_name, 
                                 region_half_len=0.001)
        )
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    @property
    def init_states(self):
        states = [
            ("On", "akita_black_bowl_1", "kitchen_table_akita_black_bowl_init_region"),
            ("On", "plate_1", "kitchen_table_plate_init_region"),
            ("On", "plate_2", "kitchen_table_wooden_cabinet_init_region"),
            # ("Open", "wooden_cabinet_1", "wooden_cabinet_1_top_drawer"),
            ]
        return states

scene_name = "test_scene2"
language = "Your Language 1"
register_task_info(language,
                    scene_name=scene_name,
                    objects_of_interest=["plate_1", "akita_black_bowl_1"],
                    goal_states=[("In", "akita_black_bowl_1", "kitchen_table_plate_init_region")]
)

# This is the default path to store all the pddl scene files. Here we store the files in the temporary folder. If you want to directly add files into the libero codebase, get the default path use the following commented lines:
# from libero.libero import get_libero_path
# YOUR_BDDL_FILE_PATH = get_libero_path("bddl_files")

# YOUR_BDDL_FILE_PATH = "tmp/pddl_files"
YOUR_BDDL_FILE_PATH = "/home/lyx/LIBERO/libero/libero/bddl_files/libero_spatial"
bddl_file_names, failures = generate_bddl_from_task_info(folder=YOUR_BDDL_FILE_PATH)

print(bddl_file_names)

print("Encountered some failures: ", failures)

with open(bddl_file_names[0], "r") as f:
    content = f.read()
print(content)
