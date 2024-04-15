#!/usr/bin/env python

import imageio
import os
from tqdm import tqdm

# finding how many timesteps (images) exist
next_one_exists = True
number_of_timesteps = 0
while next_one_exists:
    to_check = number_of_timesteps + 1
    next_one_exists = os.path.isfile('../images/2D/' + str(to_check) + '_2D.png')
    if next_one_exists:
        number_of_timesteps += 1

duration_of_each_image = 100 # miliseconds


filenames_2D = [None] * number_of_timesteps
filenames_3D = [None] * number_of_timesteps


for i in list(range(number_of_timesteps)):
    filenames_2D[i] = '../images/2D/' + str(i+1) + '_2D.png'
    filenames_3D[i] = '../images/3D/' + str(i+1) + '_3D.png'

# 2D
with imageio.get_writer('../analytical_solution_2D.gif', mode='I', duration=duration_of_each_image) as writer:
    for filename in tqdm(filenames_2D):
        image = imageio.imread(filename)
        writer.append_data(image)

# 3D
with imageio.get_writer('../analytical_solution_3D.gif', mode='I', duration=duration_of_each_image) as writer:
    for filename in tqdm(filenames_3D):
        image = imageio.imread(filename)
        writer.append_data(image)