#!/usr/bin/env python

import imageio

# hardcoded ...
number_of_timesteps = 101
duration_of_each_image = 100 # miliseconds


filenames_2D = [None] * number_of_timesteps
filenames_3D = [None] * number_of_timesteps


for i in list(range(number_of_timesteps)):
    filenames_2D[i] = '../images/2D/' + str(i+1) + '_2D.png'
    filenames_3D[i] = '../images/3D/' + str(i+1) + '_3D.png'

# 2D
with imageio.get_writer('../analytical_solution_2D.gif', mode='I', duration=duration_of_each_image) as writer:
    for filename in filenames_2D:
        image = imageio.imread(filename)
        writer.append_data(image)

# 3D
with imageio.get_writer('../analytical_solution_3D.gif', mode='I', duration=duration_of_each_image) as writer:
    for filename in filenames_3D:
        image = imageio.imread(filename)
        writer.append_data(image)