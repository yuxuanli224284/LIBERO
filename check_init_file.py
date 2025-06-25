import pickle

# 1. check the file type: 
# file libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_c.init

# 2. unzip the file to pkl file(Since it's a ZIP archive, extract it using unzip:)
# unzip pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate.init

# home/lyx/LIBERO/libero/libero/init_files/libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_c.init

with open('libero/libero/init_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.init', 'rb') as f:
    data = pickle.load(f)
print(data)
print(data.shape)