#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 08:45:01 2018

@author: ron
"""

import sys
sys.path.append('/home/ron/research/ron_postptv_code')

#from load_traj_h5 import *
from traj_pairs import get_frame_traj
import numpy as np
from mayavi import mlab
from flowtracks.io import Scene
import moviepy.editor as mpy
import os

#import mayavi_viz


def get_traj_by_id(id_list,traj_list):
    '''
    fetches a list of trajectores that have the id's in id_list from a 
    list of trajectories
    '''
    ids = [tr.trajid() for tr in traj_list]
    chosen_trajs = []
    for i in range(len(ids)):
        if ids[i] in id_list:
            chosen_trajs.append(traj_list[i])
    return chosen_trajs


get_V = lambda tr: np.mean(np.sum(tr.velocity()**2, axis=1)**0.5)



filename =  'smoothed_connected_tr_sample.h5'
X0 = (40.0, 0.0, 18.0) #(76.0,-25.0,52.0)

s = Scene(filename)
trajs = []

e=0
for tr in s.iter_trajectories():
    if len(tr)>10:
        trajs.append(tr)
        e += 1
    if e >=40000:
        break


ft = get_frame_traj(trajs)
frames = [tr.time()[0] for tr in trajs]
t0 = min(frames) #+ 200
Vmax = get_V(max(trajs, key = get_V))*0.85

times_to_plot = int(t0) # 191011186202000
counter = 0

fig = mlab.figure(size=(1720, 1500), bgcolor=(1,1,1))



def make_frame(t):
    global counter
    global Vmax
    tube_length = 10
    mlab.clf() # clear the figure (to reset the colors)
    
    ploted = []
    
    for i in range(0,-1*tube_length,-1):
        key = times_to_plot + counter + i
        try:
            trajids = ft[key]
        except:
            trajids = []
     
        for id_ in trajids:
            if id_ in ploted: continue
            tr = get_traj_by_id([id_],trajs)[0]
            ind = np.where(tr.time() == key)[0][0]
            
            #mlab.points3d(X0[0]-1000.0*tr.pos()[ind,0], 
            #              X0[1]+1000.0*tr.pos()[ind,2], 
            #              X0[2]+1000.0*tr.pos()[ind,1], 
            #        color = (1.0,0.4,0.4), figure=fig, scale_factor=1.6)
            
            s = (ind - tube_length - i) * (ind > tube_length + i)
            e = ind + tube_length + i
            v = min(get_V(tr)/Vmax , 1.0)
            c = (v , 1 - 4*(0.5-v)**2, (1.0 - v)*0.9)
            mlab.plot3d(1000.0*tr.pos()[s:e, 0], 
                        1000.0*tr.pos()[s:e, 2], 
                        1000.0*tr.pos()[s:e, 1], 
                        color = c, figure=fig, tube_radius=0.2)
            ploted.append(id_)
        
    counter += 1
    R = 50.0
    theta =  counter*0.05 - 100
    point = [X0[0] +0*R*np.cos(theta/180.0*3.1415),
             X0[1] +0*R*np.sin(theta/180.0*3.1415),
             X0[2]]
    mlab.view(azimuth = theta, distance=140,
              elevation=145, focalpoint=point, roll=None)
    #mlab.show()
    save_name = os.path.join(os.getcwd(), 'img_out', '%05d.jpg'%(counter))
    mlab.savefig(save_name, size=(1100, 850), figure=fig)
    #return mlab.screenshot(figure=fig, antialiased=False)

print 'plotting %d frames'%(int(max(frames) - t0))
for i in range(4):#int(max(frames) - t0) ):
    make_frame(1)














