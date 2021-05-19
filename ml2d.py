#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Morris-Lecar model on a 2D lattice
# FvW 03/2018

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


def ml2d(N, T, t0, dt, s, D, gL, VL, gCa, VCa, gK, VK, C, I0,
         V0, V1, V2, V3, V4, phi, stim, blocks):
    C1 = 1.0/C
    # initialize Morris-Lecar system
    v, w = V0*np.ones((N,N)), 0.015*np.ones((N,N))
    dv, dw = np.zeros((N,N)), np.zeros((N,N))
    s_sqrt_dt = s*np.sqrt(dt)
    X = np.zeros((T,N,N))
    # stimulation protocol
    I = np.zeros((t0+T,N,N))
    for st in stim:
        t_on, t_off = st[0]
        x0, x1 = st[1]
        y0, y1 = st[2]
        I[t0+t_on:t0+t_off, x0:x1, y0:y1] = I0

    # steady-state functions
    m_inf = lambda v: 0.5*(1 + np.tanh((v-V1)/V2))
    w_inf = lambda v: 0.5*(1 + np.tanh((v-V3)/V4))
    lambda_w = lambda v: phi*np.cosh((v-V3)/(2*V4))

    # iterate
    for t in range(1, t0+T):
        if (t%100 == 0): print(f"    t = {t:d}/{t0+T:d}\r", end="")
        # ML equations
        dv = 1/C*(-gL*(v-VL)-gCa*m_inf(v)*(v-VCa)-gK*w*(v-VK)+I[t,:,:])+D*L(v)
        dw = lambda_w(v) * (w_inf(v) - w)
        # Ito stochastic integration
        v += (dv*dt + s_sqrt_dt*np.random.randn(N,N))
        w += (dw*dt)
        # dead block(s):
        for bl in blocks:
            v[bl[0][0]:bl[0][1], bl[1][0]:bl[1][1]] = 0.0
            w[bl[0][0]:bl[0][1], bl[1][0]:bl[1][1]] = 0.0
        if (t >= t0):
            X[t-t0,:,:] = v
    print("\n")
    return X


def animate_video(fname, x):
    # BW
    y = 255 * (x-x.min()) / (x.max()-x.min())
    # BW inverted
    #y = 255 * ( 1 - (x-x.min()) / (x.max()-x.min()) )
    y = y.astype(np.uint8)
    nt, nx, ny = x.shape
    #print(f"nt = {nt:d}, nx = {nx:d}, ny = {ny:d}")
    # write video using opencv
    frate = 60
    out = cv2.VideoWriter(fname, \
                          cv2.VideoWriter_fourcc(*'mp4v'), \
                          frate, (nx,ny))
    print("[+] Animate:")
    for i in range(0,nt):
        print(f"\ti = {i:d}/{nt:d}\r", end="")
        img = np.ones((nx, ny, 3), dtype=np.uint8)
        for j in range(3): img[:,:,j] = y[i,::-1,:]
        out.write(img)
    out.release()
    print("")


def L(x):
    # Laplace operator
    # periodic boundary conditions
    xU = np.roll(x, shift=-1, axis=0)
    xD = np.roll(x, shift=1, axis=0)
    xL = np.roll(x, shift=-1, axis=1)
    xR = np.roll(x, shift=1, axis=1)
    Lx = xU + xD + xL + xR - 4*x
    # non-periodic boundary conditions
    Lx[0,:] = 0.0
    Lx[-1,:] = 0.0
    Lx[:,0] = 0.0
    Lx[:,-1] = 0.0
    return Lx


def main():
    print("Morris-Lecar (ML) lattice model\n")
    # Example: D=0.8, gCa=5, gK=6, C=5, s=2.0
    N = 128
    T = 20000
    t0 = 1000
    dt = 0.01
    s = 2
    D = 0.8
    # Morris-Lecar parameters
    gL = 2.0
    VL = -60.0
    gCa = 5.0 # 4.0
    VCa = 120.0
    gK = 6.0 # 8.0
    VK = -84 # -80.0 #-84.0
    C = 5.0 # 20.0
    V0 = -60.0
    V1 = -1.2
    V2 = 18.0
    V3 = 12.0 # 2.0
    V4 = 17.4 # 30.0
    phi = 1/15 # 0.04
    I = 90.0 # 110
    print("[+] Lattice size N: ", N)
    print("[+] Time steps T: ", T)
    print("[+] Warm-up steps t0: ", t0)
    print("[+] Integration time step dt: ", dt)
    print("[+] Noise intensity s: ", s)
    print("[+] Diffusion coefficient D: ", D)
    print("[+] ML parameter gL: ", gL)
    print("[+] ML parameter VL: ", VL)
    print("[+] ML parameter gCa: ", gCa)
    print("[+] ML parameter VCa: ", VCa)
    print("[+] ML parameter gK: ", gK)
    print("[+] ML parameter VK: ", VK)
    print("[+] ML parameter C: ", C)
    print("[+] ML parameter V0: ", V0)
    print("[+] ML parameter V1: ", V1)
    print("[+] ML parameter V2: ", V2)
    print("[+] ML parameter V3: ", V3)
    print("[+] ML parameter V4: ", V4)
    print("[+] ML parameter phi: ", phi)
    print("[+] Stimulation current I: ", I)

    # stimulation current configuration
    stim = [ [[50,250], [0,5], [0,5]],
             [[6200,6700], [25,30], [0,15]] ]
    #stim = [ [[50,250], [0,5], [0,5]],
    #         [[6200,6700], [25,30], [0,15]] ]
    #stim = []

    # dead blocks, array of elementy [[x0,x1], [y0,y1]]
    blocks = [ [[0,15], [5,10]] ]
    #blocks = []

    # run simulation
    data = ml2d(N, T, t0, dt, s, D, gL, VL, gCa, VCa, gK, VK, C, I,
                V0, V1, V2, V3, V4, phi, stim, blocks)
    print("[+] data dimensions: ", data.shape)

    # plot mean voltage
    m = np.mean(np.reshape(data, (T,N*N)), axis=1)
    plt.figure(figsize=(12,4))
    plt.plot(m, "-k")
    plt.tight_layout()
    plt.show()

    # save data
    #fname1 = f"ml2d_I_{I:.2f}_s_{s:.2f}_D_{D:.2f}.npy"
    #np.save(fname1, data)
    #print("[+] Data saved as: ", fname1)

    # video
    fname2 = f"ml2d_I_{I:.2f}_s_{s:.2f}_D_{D:.2f}.mp4"
    animate_video(fname2, data)
    print("[+] Data saved as: ", fname2)


if __name__ == "__main__":
    os.system("clear")
    main()
