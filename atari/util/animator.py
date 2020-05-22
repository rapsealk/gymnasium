#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib import animation


def display_frames_as_gif(frames, filename='animation.gif'):
    plt.figure(figsize=(frames[0].shape[1] / 72, frames[0].shape[0] / 72), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    anim = animation.FuncAnimation(plt.gcf(), lambda x: patch.set_data(frames[x]), frames=len(frames), interval=50)

    writer = animation.PillowWriter(fps=30)
    anim.save(filename, writer=writer)
