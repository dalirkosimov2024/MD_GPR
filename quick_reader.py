import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter
from matplotlib.animation import FuncAnimation,  PillowWriter


def dat_reader(filename="1_rho.dat"):
    path = os.getcwd()
    filename = f"{path}/script/data/{filename}"

    data = np.loadtxt(filename, usecols=(0, 1))
    r = data[:, 0]
    g_r = data[:, 1]
    label = "QMD"
    return r, g_r, label

def rdf_reader(filename, n_last=5):

    rdf_blocks = []

    current_x = []
    current_y = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) == 2:
                if current_x and current_y:
                    rdf_blocks.append((
                        np.array(current_x, dtype=float),
                        np.array(current_y, dtype=float)
                    ))

                current_x = []
                current_y = []

            elif len(parts) == 4:
                current_x.append(float(parts[1]))
                current_y.append(float(parts[2]))

    if current_x and current_y:
        rdf_blocks.append((
            np.array(current_x, dtype=float),
            np.array(current_y, dtype=float)
        ))

    # Take last n blocks
    last_blocks = rdf_blocks[-n_last:]

    # Stack and average
    r = last_blocks[0][0]  # assume same r grid
    g_stack = np.array([block[1] for block in last_blocks])

    g_r = np.mean(g_stack, axis=0)



    return r, g_r



def read_rdf_blocks(filename="rti_drive_total.rdf"):
    rdf_blocks = []
    current_x = []
    current_y = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            # LAMMPS RDF timestep/header line
            if len(parts) == 2:
                if current_x and current_y:
                    rdf_blocks.append((
                        np.array(current_x, dtype=float),
                        np.array(current_y, dtype=float)
                    ))

                current_x = []
                current_y = []

            # RDF data line
            elif len(parts) == 4:
                current_x.append(float(parts[1]))
                current_y.append(float(parts[2]))

    if current_x and current_y:
        rdf_blocks.append((
            np.array(current_x, dtype=float),
            np.array(current_y, dtype=float)
        ))

    return rdf_blocks


def rdf_movie(filename="rti_drive_light.rdf" ,output="rti_drive_light.gif", fps=10):
    rdf_blocks = read_rdf_blocks(filename)

    fig, ax = plt.subplots()

    r0, g0 = rdf_blocks[0]
    line, = ax.plot(r0, g0)

    ax.set_xlabel("r")
    ax.set_ylabel("g(r)")
    ax.set_title("RDF evolution")

    ax.set_xlim(r0.min(), r0.max())

    ymax = max(np.max(g) for _, g in rdf_blocks)
    ax.set_ylim(0, 1.1 * ymax)

    dt = 5e-5
    timestep = 5000

    def update(frame):
        r, g_r = rdf_blocks[frame]
        line.set_data(r, g_r)
        ax.set_title(f"RDF evolution | {frame*dt*timestep} ps")
        return line,

    anim = FuncAnimation(
        fig,
        update,
        frames=len(rdf_blocks),
        interval=100 / fps,
        blit=True
    )

    writer = PillowWriter(fps=fps)
    anim.save(output, writer=writer)

    plt.close(fig)


def g_r_reader():

    

    r_drive,g_r_drive = rdf_reader(filename="drive.rdf")
    g_r_drive = g_r_drive[1:]
    r_drive = r_drive[1:]

    r_equil,g_r_equil = rdf_reader(filename="rdf.rdf")
    g_r_equil = g_r_equil[1:]
    r_equil = r_equil[1:]

    r_qmd, gr_qmd,label = dat_reader()

    linewidth = 2

    u = -13 * np.log(g_r_equil)
    plt.plot(r_equil, u)

    plt.plot(r_qmd,gr_qmd, label="QMD benchmark",linewidth=linewidth)
    plt.plot(r_equil,g_r_equil,linestyle=":",label="Thermal equilibrium ",linewidth=linewidth)
    plt.plot(r_drive,g_r_drive,linestyle="--",label= r"Hydrostatic equilibrium $\rho = \rho(x,g)$", linewidth=linewidth)


    plt.axhline(y=1, color="red", alpha=0.3, label="Ideal gas normalisation")
    plt.xlabel("r, Angstrom")
    plt.ylabel("g(r)")
    plt.title(r"g(r) comparison for shock-tube setup. 13eV, 1$\rho_0$ Be")
    plt.legend()
    plt.show()


def denisty_calculator():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    fname = "hydro_equil_rho_y_time.dat"

    frames = []

    with open(fname) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):

        line = lines[i].strip()

        if line.startswith("#") or len(line) == 0:
            i += 1
            continue

        header = line.split()

        # timestep nchunks totalcount
        if len(header) == 3:
            
            
            timestep = int(header[0])

            nchunks = int(header[1])

            i += 1

            block = []

            for _ in range(nchunks):
                vals = [float(x) for x in lines[i].split()]
                block.append(vals)
                i += 1

            df = pd.DataFrame(
                block,
                columns=[
                    "chunk",
                    "x",
                    "count",
                    "density"
                ]
            )

            dt = 5e-5

            timestep = timestep # subtract time from previous sim

            df["timestep"] = timestep * dt 

            frames.append(df)

        else:
            i += 1

    data = pd.concat(frames, ignore_index=True)
    

    # Create x-t density matrix
    pivot = data.pivot(
        index="timestep",
        columns="x",
        values="density"
    )

    return pivot

def dens_plot(pivot):
    for timestep, row in pivot.iloc[::2000].iterrows():
        plt.plot(row.index, row.values+10, label=f"t = {timestep}")
    plt.legend()
    plt.show()

def fin_dnes_plot(pivot):

    

    final_density = pivot.iloc[-1]

    mask = (final_density.index >= 3) & (final_density.index <= 995)


    coeff =  np.polyfit(final_density.index[mask], final_density.values[mask], 1)
    print(coeff)



    plt.figure(figsize=(8,5))
    plt.plot(final_density.index, final_density.values)
    y_fit = coeff[0]*final_density.index + coeff[1]
    #plt.plot(final_density.index, y_fit, label = r"$n_i$ = " f"{round(coeff[0],5)}x +{round(coeff[1],3)}")

    dx = 6
    x_bins = np.arange(3, 995, dx)
    y_bins  = coeff[0]*x_bins + coeff[1]


    #plt.bar(x_bins, y_bins, width=dx, align="edge", alpha=0.3, edgecolor="k", label=f"Bin width : {dx} Angstrom")

    fontsize = 16
    plt.xlabel("x Angstrom",fontsize=fontsize)
    plt.axhline(y=0.123, color="red", label=r"Initial density, 0.123 A$^{-3}$")
    plt.ylabel(r"Number density A$^{-3}$",fontsize=fontsize)
    #plt.title(f"Final denisty profile " r"1$\rho_0$ Be" f", {ramp} A/ps$^3$ ramp")
    plt.title(f"Final denisty profile " r"Be, 1.2 million ions",fontsize=fontsize)

    plt.legend()
    plt.show()

def hydrostatic_equil_checker(pivot):

    timestep_array = coeff_array = []

    for timestep in pivot.index:
        density = pivot.loc[timestep]
        mask = (density.index >= 3) & (density.index <= 995)


        coeff =  np.polyfit(density.index[mask], density.values[mask], 1)
        coeff_array = np.append(coeff_array, coeff[0])
        timestep_array = np.append(timestep_array, timestep)

    mask = (timestep_array>= 10) 
    mean = np.mean(coeff_array[mask])
    fontsize = 16

    plt.figure(figsize=(8,5))
    plt.plot(timestep_array,coeff_array )
    plt.axhline(y=0, color="black", linestyle="--", label = "Thermodynamic equilibium")
    plt.axhline(y=mean, color="red", linestyle="--", label = r"Converging $d \rho / dx$" f" : {round(mean,4)}")



    plt.xlabel("Time (ps)",fontsize=fontsize)
    plt.ylabel(r"Density gradient ($n_i$ / $x$) [ A$^{-3}$ / A] ",fontsize=fontsize)
    plt.title("Spatial density gradient against time, " r"Be, 300,000 ions",fontsize=fontsize)
    plt.legend()
    plt.tight_layout()
    plt.show()





def density_map_plotter(pivot, ramp=0.05): 

    plt.figure(figsize=(10,6))

    plt.imshow(
        pivot.values,
        cmap="plasma",
        aspect="auto",
        origin="lower",
        extent=[
            pivot.columns.min(),
            pivot.columns.max(),
            pivot.index.min(),
            pivot.index.max()
        ]
    )
    fontsize = 16
    #plt.hlines(y=0.05,xmin = 0.5,xmax=99.5,color="red",
    #           linestyle="--",label=r"Piston release, $u_s$ = 890 A/ps ",
    #           linewidth=3)
    plt.xlabel("y (Angstrom)", fontsize=fontsize)
    plt.ylabel("Time (ps)",fontsize=fontsize)
    plt.colorbar(label="Density (Å$^{-3}$)")
    #plt.title(r"1$\rho_0$ Be, " f"{ramp} A/ps$^3$ ramp" ,fontsize=fontsize)
    plt.title(r"Be, hydrostatic equilibrium, 1.2 million ions" ,fontsize=fontsize)


    #plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.show()

def sound_speed_plotter():
    up = [50,100, 200, 300,400,600]
    us = [295,346, 427, 547, 630, 890]

    coeff =  np.polyfit(up, us, 1)

    p = np.poly1d(coeff)
    x = np.linspace(0,700)
    fontsize = 10

    plt.scatter(up, us, label="MD results")
    plt.plot(x, x*coeff[0]+coeff[1], label= r"Linear fit, $c_s$ = " f"{round(coeff[1],1)} A/ps")
    plt.xlabel(r"Piston speed, $u_p$ A/ps",fontsize=fontsize)
    plt.ylabel(r"Shock speed, $u_s$ A/ps",fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.title(r"1$\rho_0$ Be, speed of sound fit")
    plt.show()


def gravity_ramp_reader():
    filename = "gravity.dat"
    data = np.loadtxt(filename, usecols=(0, 1))
    time = data[:, 0]
    g = data[:, 1]
    

    plt.plot(time,g)
    plt.xlabel("Time (ps)")
    plt.ylabel(r"Gravity A/ps$^2$")
    plt.title(" A/ps$^3$ ramp")
    plt.show()

def temp_reader():
    filename = "hydro_equil_temp.dat"
    data = np.loadtxt(filename, usecols=(0, 1))
    time = data[:, 0] 
    temp = data[:, 1]

    time = time *5e-5


    
    mask = (time >= 0.5) 
    mean = np.mean(temp[mask])

    fontsize=16



    

    plt.plot(time,temp)
    plt.axhline(y=temp[-1], color="red", linestyle="--", label = f"Final temp: {round(mean,0)} K")
    plt.xlabel("Time (ps)",fontsize=fontsize)
   
    plt.ylabel("Temparature (K)",fontsize=fontsize)
    plt.title("Temperature (K) vs time (ps) " r"Be, 3.1 million ions ",fontsize=fontsize)
    plt.legend()
    plt.tight_layout()
    plt.show()

def pressure_reader():
    filename = "hydro_equil_press.dat"
    data = np.loadtxt(filename, usecols=(0, 1))
    time = data[:, 0] 
    pres = data[:, 1]

    time = time *5e-5

 


    #temp_filter = savgol_filter(temp,window_length=20, polyorder=3)
    
    mask = (time >= 0.5) 
    

    pres  = pres / 1e7 # from Bar to TPa

    mean = np.mean(pres[mask])

    fontsize=16

    

    plt.plot(time,pres)
    plt.axhline(y=pres[-1], color="red", linestyle="--", label = f"Final pressure: {round(mean,3)} TPa")
    plt.xlabel("Time (ps)",fontsize=fontsize)
   
    plt.ylabel("Pressure (TPa)",fontsize=fontsize)
    plt.title("Pressure (TPa) vs time (ps) " r"Be, 3.1 million ions ",fontsize=fontsize)
    plt.legend()
    plt.show()

def csv_reader():
    import pandas as pd
    import matplotlib.pyplot as plt

    # Read the CSV file
    data = pd.read_csv("temp.csv")

    # Plot temperature vs timestep
    plt.figure(figsize=(8, 5))
    plt.plot(data["Step"], data["Temp"], linewidth=1.5)

    plt.xlabel("Timestep")
    plt.ylabel("Temperature (K)")
    plt.title("Temperature vs Timestep")


    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    pivot = denisty_calculator()
    if False:
        rh, grh = rdf_reader("therm_equil_heavy.rdf")
        rl, grl = rdf_reader("therm_equil_light.rdf")
        rt, grt = rdf_reader("therm_equil_total.rdf")

        rdft, grdft , label = dat_reader()

        plt.plot(rh,grh/2, label="heavy Be (Yukawa)")
        plt.plot(rl, grl/2, label="light Be (Yukawa)")
        plt.plot(rt, grt, label="total Be (Yukawa)")
        plt.plot(rdft, grdft, label = "QMD benchmark")
        plt.title(r"1$\rho_0$ Be, RDF comparison, thermal equilibrium, 200,000 ions" )
        plt.xlabel("r, Angstrom")
        plt.ylabel("g(r)")

        plt.legend()
        plt.show()
    density_map_plotter(pivot)
    fin_dnes_plot(pivot)
    hydrostatic_equil_checker(pivot)
    #dens_plot(pivot)
    #rdf_movie("hydro_equil_light.rdf", "light_rdf.gif")
    temp_reader()
    pressure_reader()
    #gravity_ramp_reader()
    #rdf_blocks = read_rdf_blocks("rti_drive.output")
    #print(rdf_blocks)
    #rdf_movie()

    #r,gr, label = dat_reader()
    #u = -13*np.log(gr) 
    #r_array = np.arange(0.001, 5, 0.001)
    #V = 5.8**2 / r_array**4 + 2**2/r_array * np.exp(-1.9*r_array) 
    #r_heavy,gr_heavy = rdf_reader("therm_equil_heavy.rdf")
    #plt.plot(r,gr)
    #print(f"gr: {gr}")
    #plt.plot(r, u)
    #plt.plot(r_array,V)
    #plt.ylim(-1,10)
    #plt.xlim(0,6)
 
    #plt.plot(r_heavy, gr_heavy)
    #plt.xlabel("x (Angstrom)")
    #plt.ylabel("g(r)")
    #plt.title(r"1$\rho_0$ Be, $\kappa = 1.866$, b = 10.49" )
    #plt.show()
