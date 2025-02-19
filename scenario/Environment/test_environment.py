#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import YpacaraiMap
import curses
import matplotlib.pyplot as plt

env = YpacaraiMap.Environment()
env.reset()

screen = curses.initscr()
curses.noecho()
curses.cbreak()
screen.keypad(True)

fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(16,8))

ax1.set_xticks(np.arange(env.S.shape[1])-0.5)
ax1.set_yticks(np.arange(env.S.shape[0])-0.5)
ax1.grid(True, color = np.asarray([0,110,5])/255,linewidth = 1, alpha = 0.4)
ax1.axes.xaxis.set_ticklabels([])
ax1.axes.yaxis.set_ticklabels([])

ax2.set_xticks(np.arange(env.S.shape[1]))
ax2.set_yticks(np.arange(env.S.shape[0]))
ax2.grid(True, linewidth = 0.5, alpha = 0.1, drawstyle = 'steps-mid')
plt.setp(ax2.get_xticklabels(), rotation_mode="anchor")


ax3.set_xticks(np.arange(env.S.shape[1]))
ax3.set_yticks(np.arange(env.S.shape[0]))
ax3.grid(True, linewidth = 0.5, alpha = 0.2, drawstyle = 'steps-mid')
plt.setp(ax3.get_xticklabels(), rotation_mode="anchor")

fig.suptitle('Ship status map')

try:
    while True:
        char = screen.getch()
        if char == ord('q'):
            break
        elif char == ord('r'):
            print("reset")
            obs = env.reset()
            VM = obs['visited_map']
            IM = obs['importance_map']
            
            ax2.imshow(VM.T, cmap = 'gray')
            ax3.imshow(IM.T,interpolation='bicubic', cmap = 'jet')
            plt.pause(0.002)
            fig.show()
            continue
        elif char == ord('6'):
            print('RIGHT\n\r')
            accion = 2
        elif char == ord('4'):
            print('IZQUIERDA\n\r')     
            accion = 3
        elif char == ord('8'):
            print('UP\n\r')     
            accion = 0
        elif char == ord('2'):
            print('DOWN\n\r')   
            accion = 1
        elif char == ord('9'):
            print('UP+RIGHT\n\r')   
            accion = 4
        elif char == ord('7'):
            print('UP+LEFT\n\r')   
            accion = 5
        elif char == ord('3'):
            print('DOWN+RIGHT\n\r')   
            accion = 6
        elif char == ord('1'):
            print('DOWN+LEFT\n\r')   
            accion = 7
                    
        
        print(f"The action type {accion} will be performed")
        
        obs,rew,done,info = env.step(accion)
        print(f"The reward for this action has been: {rew:.3f}")
        print(f"OBS takes the value of [X,Y] = [{obs['position'][0]},{obs['position'][1]}]")
        
        VM = obs['visited_map']
        IM = obs['importance_map']
    
        ax1.imshow(env.render())
        ax2.imshow(VM, cmap = 'gray')
        ax3.imshow(IM,interpolation='nearest', cmap = 'jet_r')
        plt.pause(0.002)
        fig.show()

finally:
    curses.nocbreak(); screen.keypad(0); curses.echo()
    curses.endwin()
