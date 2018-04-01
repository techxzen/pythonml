# -*- coding: utf-8 -*
# 特别注意：上面这条语句一定要加在源代码的第一行！！！！

# @file  : practice.py
# @author: chencheng
# @date  : 20180401



import numpy as np
import matplotlib.pyplot as plt


def main():
    ''' data '''
    x1 = np.linspace(0, 5) #default, num=50, dtype=float64
    x2 = np.linspace(0, 2)
    
    y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
    y2 = np.cos(2 * np.pi * x2)
    
    ''' interactive mode on '''
    plt.ion()
    '''
    在交互模式下：
    plt.plot(x)或plt.imshow(x)是直接出图像，不需要plt.show()
    如果在脚本中使用ion()命令开启了交互模式，没有使用ioff()关闭的话，则图像会一闪而过，并不会常留。要想防止这种情况，需要在plt.show()之前加上ioff()命令。
    
    在阻塞模式下：
    打开一个窗口以后必须关掉才能打开下一个新的窗口。这种情况下，默认是不能像Matlab一样同时开很多窗口进行对比的。
    plt.plot(x)或plt.imshow(x)是直接出图像，需要plt.show()后才能显示图像
    '''
    
    
    ''' figure and axes '''
    fig, ax  = plt.subplots(2, 1)
    
    print(ax)
    
    ''' plot '''
    ax[0].plot(x1, y1, 'o-')
    ax[0].set(title = 'A tale of 2 subplots')
    ax[0].set(ylabel = 'Damped oscillation')
    
    ax[1].plot(x2, y2, '.-')
    ax[1].set(xlabel = 'time (s)')
    ax[1].set(ylabel = 'Undamped')
    ax[1].grid()
    
    ''' show '''
    #plt.show()
    
    
    ''' take a break '''
    plt.pause(1)
    
    
    ''' clear '''
    ax[0].cla()
    ax[1].cla()
    
    ''' redraw '''
    ax[0].hist(x1[:10])
    ax[1].scatter(x1, y1)
    
    ''' interactive mode off '''
    plt.ioff()
    plt.show()
    
    
    


if __name__ == "__main__":
    main()