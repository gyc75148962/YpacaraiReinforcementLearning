#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对创建的环境进行验证测试。设计的环境将被加载并
检查所有必要的函数
对要执行的任务进行有效培训。
"""
import numpy as np
import matplotlib.pyplot as plt
import YpacaraiMap
from time import sleep


env = YpacaraiMap.Environment()


env.reset()

N = 1000 # Numero de accionea aleatorias
acciones = np.random.randint(0,7,size = N)


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

fig.suptitle('Mapas de estado del barco')

for i in range(acciones.size):
    
    print("Se va a realizar una acción tipo {}".format(acciones[i]))
    
    obs,rew,done,info = env.step(acciones[i])
    
    print("La recompensa de esta acción ha sido: {0:.3f}".format(rew))
    print("OBS toma el valor de [X,Y] = [{},{}]".format(obs['position'][0],obs['position'][1]))
        

print("Ciclo terminado con éxito.")

VM = obs['visited_map']
IM = obs['importance_map']

ax1.imshow(env.render())
ax2.imshow(VM, cmap = 'gray')
ax3.imshow(IM,interpolation='bicubic', cmap = 'jet_r')

plt.show()