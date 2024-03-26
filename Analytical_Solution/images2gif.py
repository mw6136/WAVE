#!/usr/bin/env python

import imageio

# hardcoded ...
number_of_timesteps = 3
duration_of_each_image = 500 # miliseconds


filenames = [None] * number_of_timesteps

for i in list(range(number_of_timesteps)):
    filenames[i] = './images/' + str(i+1) + '.png'


import imageio
with imageio.get_writer('./analytical_solution.gif', mode='I', duration=duration_of_each_image) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)