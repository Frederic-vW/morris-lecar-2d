#!/usr/local/bin/julia
# last tested Julia version: 1.6.1
# Morris-Lecar model on a 2D lattice
# FvW 03/2018

using NPZ
using PyCall
using PyPlot
using Statistics
using VideoIO
@pyimport matplotlib.animation as anim

function ml2d(N, T, t0, dt, s, D, gL, VL, gCa, VCa, gK, VK, C, I0,
			  V0, V1, V2, V3, V4, phi, stim, blocks)
    C1 = 1.0/C
    # initialize Morris-Lecar system
    v = V0*ones(Float64,N,N)
    w = 0.015*ones(Float64,N,N)
    dv = zeros(Float64,N,N)
    dw = zeros(Float64,N,N)
    s_sqrt_dt = s*sqrt(dt)
    X = zeros(Float64,T,N,N)
    # stimulation protocol
    I = zeros(Float64,t0+T,N,N)
    for st in stim
        t_on, t_off = st[1]
        x0, x1 = st[2]
        y0, y1 = st[3]
        I[t0+t_on:t0+t_off, x0:x1, y0:y1] .= I0
    end

    # iterate
    for t in range(1, stop=t0+T, step=1)
        (t%100 == 0) && print("    t = ", t, "/", t0+T, "\r")
        # ML equations
        dv = C1.*(-gL.*(v.-VL) -gCa.*0.5.*(1.0 .+ tanh.((v.-V1)/V2)).*(v.-VCa)
                 -gK.*w.*(v.-VK) .+ I[t,:,:]) .+ D.*L(v)
        dw = phi*cosh.((v.-V3)/(2*V4)) .* (0.5*(1.0 .+ tanh.((v.-V3)/V4))-w)
        # Ito stochastic integration
        v += (dv*dt + s_sqrt_dt*randn(N,N))
        w += (dw*dt)
        # dead block(s):
        for bl in blocks
            v[bl[1][1]:bl[1][2], bl[2][1]:bl[2][2]] .= 0.0
            w[bl[1][1]:bl[1][2], bl[2][1]:bl[2][2]] .= 0.0
        end
        (t > t0) && (X[t-t0,:,:] = v)
    end
    println("\n")
    return X
end

function animate_pyplot(fname, data)
    """
    Animate 3D array as .mp4 using PyPlot, save as `fname`
    array dimensions:
        1: time
        2, 3: space
    """
    nt, nx, ny = size(data)
    vmin = minimum(data)
    vmax = maximum(data)
    # setup animation image
    fig = figure(figsize=(6,6))
    axis("off")
    t = imshow(data[1,:,:], origin="lower", cmap=ColorMap("gray"),
			   vmin=vmin, vmax=vmax)
    tight_layout()
    # frame generator
    println("[+] animate")
    function animate(i)
        (i%100 == 0) && print("    t = ", i, "/", nt, "\r")
        t.set_data(data[i+1,:,:])
    end
    # create animation
    ani = anim.FuncAnimation(fig, animate, frames=nt, interval=10)
    println("\n")
    # save animation
    ani[:save](fname, bitrate=-1,
               extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    show()
end

function animate_video(fname, data)
    """
    Animate 3D array as .mp4 using VideoIO, save as `fname`
    array dimensions:
        1: time
        2, 3: space
    """
    # BW
    y = UInt8.(round.(255*(data .- minimum(data))/(maximum(data)-minimum(data))))
    # BW inverted
    #y = UInt8.(round.(255 .- 255*(data .- minimum(data))/(maximum(data)-minimum(data))))
    encoder_options = (color_range=2, crf=0, preset="medium")
    framerate=30
    T = size(data,1)
    open_video_out(fname, y[1,end:-1:1,:], framerate=framerate,
                   encoder_options=encoder_options) do writer
        for i in range(2,stop=T,step=1)
            write(writer, y[i,end:-1:1,:])
        end
    end
end

function L(x)
    # Laplace operator
    # periodic boundary conditions
    xU = circshift(x, [-1 0])
    xD = circshift(x, [1 0])
    xL = circshift(x, [0 -1])
    xR = circshift(x, [0 1])
    Lx = xU + xD + xL + xR - 4x
    # non-periodic boundary conditions
    Lx[1,:] .= 0.0
    Lx[end,:] .= 0.0
    Lx[:,1] .= 0.0
    Lx[:,end] .= 0.0
    return Lx
end

function main()
    println("Morris-Lecar (ML) lattice model\n")
    N = 128
    T = 15000
    t0 = 1000
	dt = 0.01
	s = 2.0
	D = 0.8
	# Morris-Lecar parameters
    gL = 2.0
    VL = -60.0
    gCa = 5.5 # 4.0
    VCa = 120.0
    gK = 8.0
    VK = -84.0
    C = 5.0 # 20.0
    I = 120.0
    V0 = -60.0
    V1 = -1.2
    V2 = 18.0
    V3 = 12.0 # 2.0
    V4 = 17.4 # 30.0
    phi = 1.0/15.0 # 0.04
    println("[+] Lattice size N: ", N)
    println("[+] Time steps T: ", T)
    println("[+] Warm-up steps t0: ", t0)
	println("[+] Integration time step dt: ", dt)
	println("[+] Noise intensity s: ", s)
	println("[+] Diffusion coefficient D: ", D)
    println("[+] ML parameter gL: ", gL)
    println("[+] ML parameter VL: ", VL)
    println("[+] ML parameter gCa: ", gCa)
    println("[+] ML parameter VCa: ", VCa)
    println("[+] ML parameter gK: ", gK)
    println("[+] ML parameter VK: ", VK)
    println("[+] ML parameter C: ", C)
    println("[+] ML parameter V0: ", V0)
    println("[+] ML parameter V1: ", V1)
    println("[+] ML parameter V2: ", V2)
    println("[+] ML parameter V3: ", V3)
    println("[+] ML parameter V4: ", V4)
    println("[+] ML parameter phi: ", phi)
	println("[+] Stimulation current I: ", I)

    # stim protocol, array of elements [[t0,t1], [x0,x1], [y0,y1]]
    # double spiral 1
    #stim = [ [[50,250], [1,5], [1,5]],
    #         [[5800,6200], [20,25], [1,35]] ]
    # double spiral 2
    #stim = [ [[50,250], [1,5], [1,5]],
    #         [[5800,6200], [18,23], [5,26]] ]
    # spiral wave 1
    #stim = [ [[50,250], [1,5], [1,5]],
    #         [[6300,6800], [30,35], [5,20]] ]
    # spiral wave 2
    stim = [ [[50,250], [1,5], [1,5]],
             [[6400,6900], [30,35], [5,20]] ]
    #stim = []

    # dead blocks, array of elementy [[x0,x1], [y0,y1]]
    blocks = [ [[1,15], [5,10]] ]
    #blocks = []

    # run simulation
    data = ml2d(N, T, t0, dt, s, D, gL, VL, gCa, VCa, gK, VK, C, I,
				V0, V1, V2, V3, V4, phi, stim, blocks)
    println("[+] Data dimensions: ", size(data))

    # plot mean voltage
    m = mean(reshape(data, (T,N*N)), dims=2)
    plot(m, "-k"); show()

    # save data
    I_str = rpad(I, 4, '0') # stim. current amplitude as 4-char string
    s_str = rpad(s, 4, '0') # noise as 4-char string
    D_str = rpad(D, 4, '0') # diffusion coefficient as 4-char string
    #fname1 = string("ml2d_I_", I_str, "_s_", s_str, "_D_", D_str, ".npy")
    #npzwrite(fname1, data)
    #println("[+] Data saved as: ", fname1)

    # video
    fname2 = string("ml2d_I_", I_str, "_s_", s_str, "_D_", D_str, ".mp4")
    #animate_pyplot(fname2, data) # slow
    animate_video(fname2, data) # fast
    println("[+] Data saved as: ", fname2)
end

main()
