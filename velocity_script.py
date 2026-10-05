import numpy as np
import sys

import numpy as np


def reader():

    # ============================================================
    # INPUTS
    # ============================================================

    dump_file = "rti_equil.output"
    output_prefix = "530A_short"

    dt = 1.0       # ps between dump frames

    nx = 26
    ny = 156

    # Simulation box dimensions in Angstrom
    Lx = 530
    Ly = 3180
    Lz = 6.0

    # Atomic masses in amu
    MASS_TYPE_3 = 9.0
    MASS_TYPE_4 = 40.0

    # Conversion:
    # 1 amu / Angstrom^3 = 1.66053906660 g/cm^3
    AMU_A3_TO_G_CM3 = 1.66053906660


    # ============================================================
    # READ LAMMPS DUMP
    # ============================================================

    def read_dump_frame(f):

        line = f.readline()

        if not line:
            return None

        # TIMESTEP
        timestep = int(f.readline())

        # NUMBER OF ATOMS
        f.readline()
        n_atoms = int(f.readline())

        # BOX BOUNDS
        f.readline()

        xlo, xhi = map(float, f.readline().split()[:2])
        ylo, yhi = map(float, f.readline().split()[:2])
        zlo, zhi = map(float, f.readline().split()[:2])

        # ATOMS header
        atom_header = f.readline().strip().split()[2:]

        data = []

        for _ in range(n_atoms):

            data.append(
                list(map(float, f.readline().split()))
            )

        data = np.array(data)

        return (
            timestep,
            atom_header,
            data,
            (xlo, xhi, ylo, yhi, zlo, zhi)
        )


    # ============================================================
    # CONVERT SCALED COORDINATES TO ANGSTROM
    # ============================================================

    def get_positions(header, data):

        x = data[:, header.index("xs")] * Lx
        y = data[:, header.index("ys")] * Ly
        z = data[:, header.index("zs")] * Lz

        return x, y, z


    # ============================================================
    # GRID GEOMETRY
    # ============================================================

    dx_grid = Lx / nx
    dy_grid = Ly / ny

    # Each x-y cell spans entire z direction
    cell_volume = dx_grid * dy_grid * Lz

    print("Grid:")
    print(f"dx = {dx_grid:.6f} A")
    print(f"dy = {dy_grid:.6f} A")
    print(f"Cell volume = {cell_volume:.6f} A^3")


    # ============================================================
    # MAIN
    # ============================================================

    with open(dump_file, "r") as f:

        previous = read_dump_frame(f)

        frame_number = 0

        while True:

            current = read_dump_frame(f)

            if current is None:
                break

            timestep, header, data, box = current

            (
                previous_timestep,
                previous_header,
                previous_data,
                _
            ) = previous


            # ----------------------------------------------------
            # Current positions
            # ----------------------------------------------------

            x, y, z = get_positions(
                header,
                data
            )


            # ----------------------------------------------------
            # Previous positions
            # ----------------------------------------------------

            xp, yp, zp = get_positions(
                previous_header,
                previous_data
            )


            # ----------------------------------------------------
            # Atom IDs
            # ----------------------------------------------------

            ids = data[
                :, header.index("id")
            ].astype(int)

            ids_previous = previous_data[
                :, previous_header.index("id")
            ].astype(int)


            # ----------------------------------------------------
            # Atom types
            # ----------------------------------------------------

            atom_types = data[
                :, header.index("type")
            ].astype(int)


            # ----------------------------------------------------
            # Assign particle masses [amu]
            # ----------------------------------------------------

            mass = np.zeros(len(atom_types))

            mass[atom_types == 3] = MASS_TYPE_3
            mass[atom_types == 4] = MASS_TYPE_4


            # Check for unknown atom types
            unknown_types = np.unique(
                atom_types[
                    (atom_types != 3) &
                    (atom_types != 4)
                ]
            )

            if len(unknown_types) > 0:

                print(
                    "WARNING: Undefined masses for atom types:",
                    unknown_types
                )


            # ----------------------------------------------------
            # Match atoms by ID
            # ----------------------------------------------------

            previous_dict = {
                atom_id: i
                for i, atom_id in enumerate(ids_previous)
            }


            # ----------------------------------------------------
            # Calculate particle velocities
            # ----------------------------------------------------

            vx = np.zeros(len(ids))
            vy = np.zeros(len(ids))

            for i, atom_id in enumerate(ids):

                j = previous_dict[atom_id]

                dx = x[i] - xp[j]
                dy = y[i] - yp[j]

                # Periodic boundary correction
                dx -= np.rint(dx / Lx) * Lx
                dy -= np.rint(dy / Ly) * Ly

                vx[i] = dx / dt
                vy[i] = dy / dt


            # ====================================================
            # PARTICLE MOMENTUM
            #
            # p_i = m_i v_i
            #
            # Units:
            # amu * Angstrom / ps
            # ====================================================

            px = mass * vx
            py = mass * vy


            # ====================================================
            # PARTICLE KINETIC ENERGY
            #
            # KE_i = 1/2 m_i (vx_i^2 + vy_i^2)
            #
            # Units:
            # amu * Angstrom^2 / ps^2
            #
            # NOTE:
            # This is currently 2D kinetic energy because only
            # vx and vy are reconstructed.
            # ====================================================

            ke = 0.5 * mass * (
                vx**2 +
                vy**2
            )


            # ====================================================
            # CREATE X-Y GRID
            # ====================================================

            grid_vx = np.zeros((ny, nx))
            grid_vy = np.zeros((ny, nx))

            # Total mass in each cell
            grid_mass = np.zeros((ny, nx))

            # Sum of particle momentum
            grid_px = np.zeros((ny, nx))
            grid_py = np.zeros((ny, nx))

            # Sum of particle kinetic energy
            grid_ke = np.zeros((ny, nx))

            # Number of particles in each cell
            count = np.zeros((ny, nx))


            # ----------------------------------------------------
            # Determine grid cell for each particle
            # ----------------------------------------------------

            ix = np.floor(
                x / Lx * nx
            ).astype(int)

            iy = np.floor(
                y / Ly * ny
            ).astype(int)

            ix = np.clip(
                ix,
                0,
                nx - 1
            )

            iy = np.clip(
                iy,
                0,
                ny - 1
            )


            # ====================================================
            # ACCUMULATE PARTICLE QUANTITIES
            # ====================================================

            for i in range(len(x)):

                j = iy[i]
                k = ix[i]

                # Velocity
                grid_vx[j, k] += vx[i]
                grid_vy[j, k] += vy[i]

                # Mass
                grid_mass[j, k] += mass[i]

                # Momentum
                grid_px[j, k] += px[i]
                grid_py[j, k] += py[i]

                # Kinetic energy
                grid_ke[j, k] += ke[i]

                # Number of particles
                count[j, k] += 1


            # ====================================================
            # AVERAGE QUANTITIES
            # ====================================================

            mask = count > 0


            # ----------------------------------------------------
            # Average velocity
            #
            # <v> = (1/N) sum_i v_i
            # ----------------------------------------------------

            grid_vx[mask] /= count[mask]
            grid_vy[mask] /= count[mask]


            # ----------------------------------------------------
            # Average momentum per particle
            #
            # <p> = (1/N) sum_i m_i v_i
            # ----------------------------------------------------

            grid_px[mask] /= count[mask]
            grid_py[mask] /= count[mask]


            # ----------------------------------------------------
            # Average kinetic energy per particle
            #
            # <KE> = (1/N) sum_i 1/2 m_i v_i^2
            # ----------------------------------------------------

            grid_ke[mask] /= count[mask]


            # ====================================================
            # MASS DENSITY
            #
            # rho = total mass in cell / cell volume
            #
            # Output units: g/cm^3
            # ====================================================

            grid_density = (
                grid_mass
                / cell_volume
                * AMU_A3_TO_G_CM3
            )


            # ====================================================
            # VORTICITY
            #
            # omega_z = dv_y/dx - dv_x/dy
            # ====================================================

            dvy_dx = np.gradient(
                grid_vy,
                dx_grid,
                axis=1
            )

            dvx_dy = np.gradient(
                grid_vx,
                dy_grid,
                axis=0
            )

            omega_z = (
                dvy_dx -
                dvx_dy
            )


            # ====================================================
            # OUTPUT GRID DUMP
            # ====================================================

            filename = (
                f"{output_prefix}_"
                f"{frame_number:05d}.dump"
            )

            with open(filename, "w") as out:

                n_grid = nx * ny

                out.write(
                    "ITEM: TIMESTEP\n"
                )

                out.write(
                    f"{timestep}\n"
                )

                out.write(
                    "ITEM: NUMBER OF ATOMS\n"
                )

                out.write(
                    f"{n_grid}\n"
                )

                out.write(
                    "ITEM: BOX BOUNDS pp pp pp\n"
                )

                out.write(
                    f"0.0 {Lx}\n"
                )

                out.write(
                    f"0.0 {Ly}\n"
                )

                out.write(
                    f"0.0 {Lz}\n"
                )


                # ------------------------------------------------
                # OUTPUT COLUMNS
                # ------------------------------------------------

                out.write(
                    "ITEM: ATOMS "
                    "id type "
                    "x y z "
                    "vx vy vz "
                    "vorticity "
                    "density "
                    "px py "
                    "ke\n"
                )


                particle_id = 1


                # ------------------------------------------------
                # WRITE GRID CELLS
                # ------------------------------------------------

                for j in range(ny):

                    for i in range(nx):

                        X = (
                            i + 0.5
                        ) * dx_grid

                        Y = (
                            j + 0.5
                        ) * dy_grid

                        Z = Lz / 2.0


                        out.write(

                            f"{particle_id} 1 "

                            f"{X:.6f} "
                            f"{Y:.6f} "
                            f"{Z:.6f} "

                            f"{grid_vx[j,i]:.8e} "
                            f"{grid_vy[j,i]:.8e} "

                            f"0.0 "

                            f"{omega_z[j,i]:.8e} "

                            f"{grid_density[j,i]:.8e} "

                            f"{grid_px[j,i]:.8e} "
                            f"{grid_py[j,i]:.8e} "

                            f"{grid_ke[j,i]:.8e}\n"
                        )

                        particle_id += 1


            # ----------------------------------------------------
            # Advance to next frame
            # ----------------------------------------------------

            previous = current

            frame_number += 1

            print(
                f"Finished frame {frame_number}, "
                f"timestep {timestep}"
            )

def file_maker():
    import numpy as np
    import glob
    import os
    import csv

    keyword = "530A_0"
    directory = "."
    output_file = "velocity_vorticity_vs_time.csv"

    def read_dump_file(filename):

        results = []

        with open(filename, "r") as f:

            while True:

                line = f.readline()

                if not line:
                    break

                if "TIMESTEP" not in line:
                    continue

                timestep = int(f.readline().strip())

                # NUMBER OF ATOMS
                line = f.readline()

                if "NUMBER OF ATOMS" not in line:
                    print(f"WARNING: Bad format in {filename}")
                    break

                n_atoms = int(f.readline().strip())

                # BOX BOUNDS
                line = f.readline()

                if "BOX BOUNDS" not in line:
                    print(f"WARNING: BOX BOUNDS missing in {filename}")
                    break

                f.readline()
                f.readline()
                f.readline()

                # ATOM HEADER
                header_line = f.readline().strip()

                if "ITEM: ATOMS" not in header_line:
                    print(f"WARNING: ATOMS header missing in {filename}")
                    break

                columns = header_line.split()[2:]

                required = ["y", "vx", "vy", "vorticity"]

                if not all(col in columns for col in required):

                    print(f"Skipping {filename}, timestep {timestep}")

                    for _ in range(n_atoms):
                        f.readline()

                    continue

                y_col = columns.index("y")
                vx_col = columns.index("vx")
                vy_col = columns.index("vy")
                vort_col = columns.index("vorticity")

                vx = []
                vy = []
                vort = []

                for i in range(n_atoms):

                    values = f.readline().split()

                    y = float(values[y_col])

                    # Only include required y region
                    if y < 3180:

                        vx.append(float(values[vx_col]))
                        vy.append(float(values[vy_col]))
                        vort.append(float(values[vort_col]))

                vx = np.array(vx)
                vy = np.array(vy)
                vort = np.array(vort)

                if len(vx) == 0:
                    continue


                # ================================================
                # SEPARATE POSITIVE AND NEGATIVE VELOCITIES
                # ================================================

                vx_positive = vx[vx > 0]
                vx_negative = vx[vx < 0]

                vy_positive = vy[vy > 0]
                vy_negative = vy[vy < 0]

                vort_positive = vort[vort > 0]
                vort_negative = vort[vort < 0]


                # ================================================
                # AVERAGES
                # ================================================

                avg_vx_positive = (
                    np.mean(vx_positive)
                    if len(vx_positive) > 0
                    else np.nan
                )

                avg_vx_negative = (
                    np.mean(vx_negative)
                    if len(vx_negative) > 0
                    else np.nan
                )

                avg_vy_positive = (
                    np.mean(vy_positive)
                    if len(vy_positive) > 0
                    else np.nan
                )

                avg_vy_negative = (
                    np.mean(vy_negative)
                    if len(vy_negative) > 0
                    else np.nan
                )

                avg_vort_positive = (
                    np.mean(vort_positive)
                    if len(vort_positive) > 0
                    else np.nan
                )

                avg_vort_negative = (
                np.mean(vort_negative)
                if len(vort_negative) > 0
                else np.nan
            )



                avg_vort = np.mean(np.abs(vort))


                results.append([
                    timestep,
                    avg_vx_positive,
                    avg_vx_negative,
                    avg_vy_positive,
                    avg_vy_negative,
                    avg_vort_positive,
                    avg_vort_negative

                ])

        return results


    # ============================================================
    # FIND FILES
    # ============================================================

    pattern = os.path.join(directory, f"*{keyword}*")
    files = sorted(glob.glob(pattern))

    print(f"Found {len(files)} matching files.")

    if len(files) == 0:
        raise RuntimeError(
            f"No files containing '{keyword}' were found."
        )


    # ============================================================
    # PROCESS
    # ============================================================

    all_results = []

    for number, filename in enumerate(files, start=1):

        print(f"[{number}/{len(files)}] {filename}")

        results = read_dump_file(filename)

        all_results.extend(results)


    # Sort by timestep
    all_results.sort(key=lambda row: row[0])

    # ============================================================
# CALCULATE TIME DERIVATIVES
# ============================================================

    data = np.array(all_results, dtype=float)

    timestep = data[:, 0]

    vx_pos = data[:, 1]
    vx_neg = data[:, 2]

    vy_pos = data[:, 3]
    vy_neg = data[:, 4]

    vort_pos = data[:, 5]
    vort_neg = data[:, 6]


    # Convert timestep to physical time in ps
    # CHANGE this if your LAMMPS timestep is different
    time = timestep * 5e-5


    # Calculate derivatives
    dvx_pos_dt = np.gradient(vx_pos, time)
    dvx_neg_dt = np.gradient(vx_neg, time)

    dvy_pos_dt = np.gradient(vy_pos, time)
    dvy_neg_dt = np.gradient(vy_neg, time)

    dvort_pos_dt = np.gradient(vort_pos, time)
    dvort_neg_dt = np.gradient(vort_neg, time)


    # Add derivatives onto all_results
    all_results_with_derivatives = []

    for i in range(len(all_results)):

        all_results_with_derivatives.append([
            timestep[i],

            vx_pos[i],
            vx_neg[i],

            vy_pos[i],
            vy_neg[i],

            vort_pos[i],
            vort_neg[i],

            dvx_pos_dt[i],
            dvx_neg_dt[i],

            dvy_pos_dt[i],
            dvy_neg_dt[i],

            dvort_pos_dt[i],
            dvort_neg_dt[i]
        ])


    # ============================================================
    # WRITE CSV
    # ============================================================

    with open(output_file, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "timestep",
            "avg_vx_positive",
            "avg_vx_negative",
            "avg_vy_positive",
            "avg_vy_negative",
            "avg_vort_positive",
            "avg_vort_negative",
            "dvx_positive_dt",
            "dvx_negative_dt",
            "dvy_positive_dt",
            "dvy_negative_dt",
            "dvort_positive_dt",
            "dvort_negative_dt"
        ])

        writer.writerows(all_results_with_derivatives)

    print("\nDone.")
    print(f"Processed {len(all_results_with_derivatives)} valid timesteps.")
    print(f"Output: {output_file}")


# ================================================================
# PLOTTER
# ================================================================

def plotter():

    import numpy as np
    import matplotlib.pyplot as plt

    data = np.genfromtxt(
        "velocity_vorticity_vs_time.csv",
        delimiter=",",
        names=True
    )

    # Your conversion from timestep -> ps
    time = data["timestep"] * 5e-5

    vx_pos = data["avg_vx_positive"]
    vx_neg = data["avg_vx_negative"]

    vy_pos = data["avg_vy_positive"]
    vy_neg = data["avg_vy_negative"]

    vort_pos = data["avg_vort_positive"]
    vort_neg = data["avg_vort_negative"]

    vx_pos_der = data["dvx_positive_dt"]
    vx_neg_der = data["dvx_negative_dt"]

    vy_pos_der = data["dvy_positive_dt"]
    vy_neg_der = data[ "dvy_negative_dt"]

    vort_pos_der = data["dvort_positive_dt"]
    vort_neg_der = data["dvort_negative_dt"]


    # ============================================================
    # PLOT normal
    # ============================================================

    if False:

        fig, ax1 = plt.subplots(figsize=(9, 6))


        # vx
        line1, = ax1.plot(
            time,
            vx_pos,
            label=r"$\langle v_x\rangle_{v_x>0}$",
            color = "blue"
        )

        line2, = ax1.plot(
            time,
            vx_neg,
            label=r"$\langle v_x\rangle_{v_x<0}$",
            color = "blue",
            linestyle = "--"

        )


        # vy
        line3, = ax1.plot(
            time,
            vy_pos,
            label=r"$\langle v_y\rangle_{v_y>0}$",
            color = "orange"
        )

        line4, = ax1.plot(
            time,
            vy_neg,
            label=r"$\langle v_y\rangle_{v_y<0}$",
            color = "orange",
            linestyle= "--"
        )


        ax1.axhline(
            0,
            linewidth=0.8,
            linestyle="--"
        )

        ax1.set_xlabel("Time (ps)")

        ax1.set_ylabel(
            r"Average velocity ($\AA$/ps)"
        )


        # ============================================================
        # VORTICITY ON SECOND AXIS
        # ============================================================

        ax2 = ax1.twinx()

        line5, = ax2.plot(
            time,
            vort_pos,
            label=r"$\langle \omega_z\rangle_{\omega>0}$",
            color="green"
        )

        ax2.set_ylabel(
            r"Average vorticity  (ps$^{-1}$)"
        )

        line6, = ax2.plot(
        time,
        vort_neg,
        label=r"$\langle \omega_z\rangle_{\omega<0}$",
        color="green",
        linestyle="--")



        # ============================================================
        # LEGEND
        # ============================================================

        lines = [
            line1,
            line2,
            line3,
            line4,
            line5,
            line6
        ]

        labels = [
            line.get_label()
            for line in lines
        ]

        ax1.legend(
            lines,
            labels,
            loc="best"
        )

        plt.tight_layout()
        plt.show()
        plt.close()


    # ============================================================
    # PLOT derivaitves
    # ============================================================
    if True:

        fig, ax1 = plt.subplots(figsize=(9, 6))


        # vx
        line1, = ax1.plot(
            time,
            vx_pos_der,
            label=r"$d_t \langle v_x\rangle_{v_x<0}$",
            color = "blue",
     
        )


        # vy
        line3, = ax1.plot(
            time,
            vy_neg_der,
            label=r"$ d_t \langle v_y\rangle_{v_y<0}$",
            color = "orange",
      

        )



        ax1.axhline(
            0,
            linewidth=0.8,
            linestyle="--"
        )
        line4 = ax1.axvline(
            21.24,
            linewidth=1.5,
            linestyle="--",
            color="red",
            label = r"$ t(v_x = v_y)$"
        )

        ax1.set_xlabel("Time (ps)")

        ax1.set_ylabel(
            r"Average acceleration ($\AA$/ps^2)"
        )


        # ============================================================
        # VORTICITY ON SECOND AXIS
        # ============================================================
        """
        ax2 = ax1.twinx()

        line5, = ax2.plot(
            time,
            vort_pos_der,
            label=r"$\langle \omega_z\rangle_{\omega>0}$",
            color="green"
        )

        ax2.set_ylabel(
            r"Average vorticity  (ps$^{-1}$)"
        )

        """


        # ============================================================
        # LEGEND
        # ============================================================

        lines = [
            line1,

            line3,
            line4

            #line5,
            
        ]

        labels = [
            line.get_label()
            for line in lines
        ]

        ax1.legend(
            lines,
            labels,
            loc="best"
        )

        plt.tight_layout()
        plt.show()







def fourier_2d():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import glob
    import re


    # ============================================================
    # SETTINGS
    # ============================================================

    file_pattern = "*530A_0*"     # CHANGE THIS
    frame_interval = 150              # milliseconds

    # Remove spatial mean velocity before FFT.
    # Recommended because otherwise k=0 can dominate the spectrum.
    remove_mean = True

    # Logarithmic spectrum is much easier to visualize.
    use_log = True


    # ============================================================
    # READ ONE FILE
    # ============================================================

    def read_file(filename):

        with open(filename, "r") as f:

            # ITEM: TIMESTEP
            f.readline()
            timestep = int(f.readline())

            # ITEM: NUMBER OF ATOMS
            f.readline()
            N = int(f.readline())

            # ITEM: BOX BOUNDS
            f.readline()

            xlo, xhi = map(float, f.readline().split())
            ylo, yhi = map(float, f.readline().split())
            zlo, zhi = map(float, f.readline().split())

            # ITEM: ATOMS ...
            header = f.readline().split()[2:]

            columns = {name: i for i, name in enumerate(header)}

            data = np.loadtxt(f)

        x = data[:, columns["x"]]
        y = data[:, columns["y"]]

        vx = data[:, columns["vx"]]
        vy = data[:, columns["vy"]]

        return timestep, x, y, vx, vy


    # ============================================================
    # SORT FILES BY TIMESTEP
    # ============================================================

    files = glob.glob(file_pattern)

    if len(files) == 0:
        raise RuntimeError(
            f"No files found matching: {file_pattern}"
        )


    def get_timestep(filename):

        with open(filename, "r") as f:
            f.readline()
            return int(f.readline())


    files = sorted(files, key=get_timestep)

    print(f"Found {len(files)} files")


    # ============================================================
    # DETERMINE GRID FROM FIRST FILE
    # ============================================================

    timestep, x, y, vx, vy = read_file(files[0])

    x_unique = np.sort(np.unique(x))
    y_unique = np.sort(np.unique(y))

    nx = len(x_unique)
    ny = len(y_unique)

    print(f"Grid: nx = {nx}, ny = {ny}")
    print(f"Grid points = {nx * ny}")
    print(f"Data points = {len(x)}")

    if nx * ny != len(x):
        raise RuntimeError(
            "The data do not appear to form a complete rectangular grid."
        )


    dx = np.mean(np.diff(x_unique))
    dy = np.mean(np.diff(y_unique))

    print(f"dx = {dx:.4f} A")
    print(f"dy = {dy:.4f} A")


    # ============================================================
    # WAVENUMBER AXES
    # ============================================================

    # np.fft.fftfreq gives cycles / Angstrom.
    # Multiply by 2*pi to obtain angular wavenumber rad/A.

    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)

    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)


    # ============================================================
    # CALCULATE 2D VELOCITY FFT
    # ============================================================

    def calculate_spectrum(filename):

        timestep, x, y, vx, vy = read_file(filename)

        # --------------------------------------------------------
        # Sort particles/grid points into y,x order
        # --------------------------------------------------------

        order = np.lexsort((x, y))

        vx = vx[order]
        vy = vy[order]

        # shape = (ny, nx)
        vx_grid = vx.reshape(ny, nx)
        vy_grid = vy.reshape(ny, nx)

        # --------------------------------------------------------
        # Remove k=0 bulk flow
        # --------------------------------------------------------

        if remove_mean:

            vx_grid = vx_grid - np.mean(vx_grid)
            vy_grid = vy_grid - np.mean(vy_grid)

        # --------------------------------------------------------
        # 2D FFT
        # --------------------------------------------------------

        Vx = np.fft.fft2(vx_grid)
        Vy = np.fft.fft2(vy_grid)

        Vx = np.fft.fftshift(Vx)
        Vy = np.fft.fftshift(Vy)

        # --------------------------------------------------------
        # Total velocity spectral power
        # --------------------------------------------------------

        power = (
            np.abs(Vx)**2
            +
            np.abs(Vy)**2
        )

        if use_log:
            power = np.log10(power + 1e-20)

        return timestep, power


    # ============================================================
    # FIRST FRAME
    # ============================================================

    timestep, spectrum = calculate_spectrum(files[0])


    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(
        spectrum,
        origin="lower",
        extent=[
            kx.min(),
            kx.max(),
            ky.min(),
            ky.max()
        ],
        aspect="auto",
        interpolation="bilinear",
        vmin=3,
        vmax=9
    )

    cbar = plt.colorbar(im, ax=ax)

    if use_log:
        cbar.set_label(
            r"$\log_{10}(|\tilde{v}_x|^2 + |\tilde{v}_y|^2)$"
        )
    else:
        cbar.set_label(
            r"$|\tilde{v}_x|^2 + |\tilde{v}_y|^2$"
        )

    ax.set_xlabel(r"$k_x$ ($\AA^{-1}$)")
    ax.set_ylabel(r"$k_y$ ($\AA^{-1}$)")

    title = ax.set_title(
        f"Timestep = {timestep}"
    )

    plt.tight_layout()


    # ============================================================
    # ANIMATION UPDATE
    # ============================================================

    def update(frame):

        filename = files[frame]

        timestep, spectrum = calculate_spectrum(filename)
        timestep = timestep*5e-5

        im.set_data(spectrum)




        title.set_text(
            f"Velocity Fourier Spectrum | timestep = {timestep}"
        )

        return im, title


    # ============================================================
    # RUN ANIMATION
    # ============================================================

    ani = FuncAnimation(
        fig,
        update,
        frames=len(files),
        interval=frame_interval,
        blit=False,
        repeat=True
    )

    ax.set_xlim(0, np.max(kx))
    ax.set_ylim(0, np.max(ky))

    plt.show()



def k_vs_t():

    import numpy as np
    import matplotlib.pyplot as plt
    import glob


    # ============================================================
    # SETTINGS
    # ============================================================

    file_pattern = "*530A_0*"
    use_log = True


    # ============================================================
    # READ FILE
    # ============================================================

    def read_file(filename):

        with open(filename, "r") as f:

            f.readline()
            timestep = int(f.readline())

            f.readline()
            N = int(f.readline())

            f.readline()

            xlo, xhi = map(float, f.readline().split())
            ylo, yhi = map(float, f.readline().split())
            zlo, zhi = map(float, f.readline().split())

            header = f.readline().split()[2:]

            columns = {
                name: i for i, name in enumerate(header)
            }

            data = np.loadtxt(f)

        x = data[:, columns["x"]]
        y = data[:, columns["y"]]

        vx = data[:, columns["vx"]]
        vy = data[:, columns["vy"]]

        return timestep, x, y, vx, vy


    # ============================================================
    # FIND FILES
    # ============================================================

    files = glob.glob(file_pattern)

    if len(files) == 0:
        raise RuntimeError(
            f"No files found matching {file_pattern}"
        )


    def get_timestep(filename):

        with open(filename, "r") as f:
            f.readline()
            return int(f.readline())


    files = sorted(files, key=get_timestep)

    print(f"Found {len(files)} files")


    # ============================================================
    # DETERMINE GRID
    # ============================================================

    _, x, y, _, _ = read_file(files[0])

    x_unique = np.sort(np.unique(x))
    y_unique = np.sort(np.unique(y))

    nx = len(x_unique)
    ny = len(y_unique)

    dx = np.mean(np.diff(x_unique))
    dy = np.mean(np.diff(y_unique))

    print(f"Grid = {nx} x {ny}")
    print(f"dx = {dx:.4f} A")
    print(f"dy = {dy:.4f} A")


    # ============================================================
    # WAVENUMBERS
    # ============================================================

    kx = 2*np.pi*np.fft.fftfreq(nx, d=dx)
    ky = 2*np.pi*np.fft.fftfreq(ny, d=dy)

    idx_kx0 = np.argmin(np.abs(kx))
    idx_ky0 = np.argmin(np.abs(ky))


    # ============================================================
    # STORAGE
    # ============================================================

    timesteps = []

    kx_spectra = []
    ky_spectra = []


    # ============================================================
    # LOOP THROUGH TIME
    # ============================================================

    for n, filename in enumerate(files):

        timestep, x, y, vx, vy = read_file(filename)


        # --------------------------------------------------------
        # Sort onto (y,x) grid
        # --------------------------------------------------------

        order = np.lexsort((x, y))

        vx = vx[order]
        vy = vy[order]

        vx_grid = vx.reshape(ny, nx)
        vy_grid = vy.reshape(ny, nx)


        # --------------------------------------------------------
        # Remove bulk velocities
        # --------------------------------------------------------

        vx_grid -= np.mean(vx_grid)
        vy_grid -= np.mean(vy_grid)


        # --------------------------------------------------------
        # 2D Fourier transforms
        # --------------------------------------------------------

        Vx = np.fft.fft2(vx_grid)
        Vy = np.fft.fft2(vy_grid)


        # ========================================================
        # 1. kx spectrum at ky = 0
        #
        #    Vy(ky=0, kx)
        #
        #    This captures y-directed velocity structures
        #    varying along x -> initial RTI mode.
        # ========================================================

        kx_power = np.abs(
            Vy[idx_ky0, :]
        )**2


        # ========================================================
        # 2. ky spectrum at kx = 0
        #
        #    Vx(ky, kx=0)
        #
        #    This captures x-directed velocity structures
        #    varying along y -> banded/zonal flow.
        # ========================================================

        ky_power = np.abs(
            Vx[:, idx_kx0]
        )**2


        # Remove k = 0 mode
        kx_power[idx_kx0] = 0.0
        ky_power[idx_ky0] = 0.0


        # --------------------------------------------------------
        # Store
        # --------------------------------------------------------

        timesteps.append(timestep)

        kx_spectra.append(kx_power)
        ky_spectra.append(ky_power)


        print(
            f"{n+1:4d}/{len(files)} "
            f"step={timestep}"
        )


    # ============================================================
    # CONVERT TO ARRAYS
    # ============================================================

    timesteps = np.array(timesteps)

    kx_spectra = np.array(kx_spectra)
    ky_spectra = np.array(ky_spectra)


    # ============================================================
    # POSITIVE k ONLY
    # ============================================================

    positive_x = kx > 0
    positive_y = ky > 0

    kx_positive = kx[positive_x]
    ky_positive = ky[positive_y]

    Px = kx_spectra[:, positive_x]
    Py = ky_spectra[:, positive_y]


    # ============================================================
    # SORT k
    # ============================================================

    order_x = np.argsort(kx_positive)
    order_y = np.argsort(ky_positive)

    kx_positive = kx_positive[order_x]
    ky_positive = ky_positive[order_y]

    Px = Px[:, order_x]
    Py = Py[:, order_y]


    # ============================================================
    # LOG POWER
    # ============================================================
    use_log = False
    if use_log:

        Px_plot = np.log10(Px + 1e-20)
        Py_plot = np.log10(Py + 1e-20)

    else:

        Px_plot = Px
        Py_plot = Py


    # ============================================================
    # FIGURE 1
    #
    # kx spectrum, ky = 0
    # ============================================================

    plt.figure(figsize=(9, 6))

    mesh = plt.pcolormesh(
        timesteps *5e-5,
        kx_positive,
        Px_plot.T,
        shading="auto"
    )

    cbar = plt.colorbar(mesh)

    if use_log:
        cbar.set_label(
            r"$\log_{10}|\tilde{v}_y(k_x,k_y=0)|^2$"
        )
    else:
        cbar.set_label(
            r"$|\tilde{v}_y(k_x,k_y=0)|^2$"
        )

    plt.xlabel("Time (ps)")
    plt.ylabel(r"$k_x$ ($\AA^{-1}$)")

    plt.title(
        r"$k_x$ spectrum at $k_y=0$ — vertical/RTI flow"
    )

    plt.tight_layout()

    plt.savefig(
        "kx_vs_time.png",
        dpi=900
    )
    plt.axhline(
        2*np.pi / 530,
        color="red",
        linestyle="--",
        label = r"$\lambda_\text{RTI} = 530 \AA$")
    plt.legend()

    plt.show()


    # ============================================================
    # FIGURE 2
    #
    # ky spectrum, kx = 0
    # ============================================================

    plt.figure(figsize=(9, 6))

    mesh = plt.pcolormesh(
        timesteps*5e-5,
        ky_positive,
        Py_plot.T,
        shading="auto"
    )

    cbar = plt.colorbar(mesh)

    if use_log:
        cbar.set_label(
            r"$\log_{10}|\tilde{v}_x(k_x=0,k_y)|^2$"
        )
    else:
        cbar.set_label(
            r"$|\tilde{v}_x(k_x=0,k_y)|^2$"
        )
        


    plt.xlabel("Time (ps)")
    plt.ylabel(r"$k_y$ ($\AA^{-1}$)")

    plt.title(
        r"$k_y$ spectrum at $k_x=0$ — horizontal/banded flow"
    )

    plt.tight_layout()

    plt.savefig(
        "ky_vs_time.png",
        dpi=600
    )

    plt.show()


    # ============================================================
    # SAVE DATA
    # ============================================================

    output_x = np.column_stack((
        timesteps,
        Px
    ))

    header_x = (
        "Time (ps) "
        +
        " ".join(
            [f"kx={k:.8e}" for k in kx_positive]
        )
    )

    np.savetxt(
        "kx_vs_time.dat",
        output_x,
        header=header_x
    )


    output_y = np.column_stack((
        timesteps,
        Py
    ))

    header_y = (
        "Time (ps) "
        +
        " ".join(
            [f"ky={k:.8e}" for k in ky_positive]
        )
    )

    np.savetxt(
        "ky_vs_time.dat",
        output_y,
        header=header_y
    )

def avg_k_vs_t():
    import numpy as np
    import matplotlib.pyplot as plt


    def load_spectrum(filename):
        with open(filename, "r") as f:
            header = f.readline().replace("#", "").split()

        k = np.array([float(x.split("=")[1]) for x in header[1:]])

        data = np.loadtxt(filename)

        timestep = data[:, 0]
        power = data[:, 1:]

        return timestep, k, power


    # ---------------------------------------------------------
    # Load spectra
    # ---------------------------------------------------------

    timestep_kx, kx, P_kx = load_spectrum("kx_vs_time.dat")
    timestep_ky, ky, P_ky = load_spectrum("ky_vs_time.dat")

    time_kx = timestep_kx * 5e-5
    time_ky = timestep_ky * 5e-5


    # =========================================================
    # 1. RTI mode
    # =========================================================

    lambda_RTI = 530.0       # Angstrom
    k_RTI = 2*np.pi/lambda_RTI

    # Find Fourier bin closest to seeded RTI mode
    idx_RTI = np.argmin(np.abs(kx - k_RTI))

    k_RTI_actual = kx[idx_RTI]

    # Power in seeded RTI mode
    P_RTI = P_kx[:, idx_RTI]


    # =========================================================
    # 2. Zonal-flow band
    # =========================================================

    kZ_min = 0.0069154
    kZ_max = 0.008891

    zonal_mask = (ky >= kZ_min) & (ky <= kZ_max)

    # Total power contained in the zonal band
    P_Z = np.sum(P_ky[:, zonal_mask], axis=1)


    # =========================================================
    # Information
    # =========================================================

    print("Target RTI k =", k_RTI)
    print("Actual FFT RTI k =", k_RTI_actual)

    print("\nZonal modes included:")
    print(ky[zonal_mask])


    # =========================================================
    # Plot
    # =========================================================

    plt.figure(figsize=(9,6))

    plt.plot(
        time_kx,
        P_RTI,
        label=rf"RTI: $|\tilde{{v}}_y(k_{{RTI}},0)|^2$"
    )

    plt.plot(
        time_ky,
        P_Z,
        label=rf"Zonal:  $|\tilde{{v}}_x(0,k_Z)|^2$"
    )

    plt.xlabel("Time (ps)")
    plt.ylabel("Spectral power " r"$|\tilde{v}|^2$")
    plt.legend()
    plt.title("Comparison of spectral power for RTI mode vs dominant zonal mode")

    plt.tight_layout()
    plt.savefig("RTI_vs_zonal_power.png", dpi=300)
    plt.show()


    # =========================================================
    # Save data
    # =========================================================

    # Assuming the two files contain the same timesteps
    output = np.column_stack((
        time_kx,
        P_RTI,
        P_Z
    ))

    np.savetxt(
        "RTI_vs_zonal_power.dat",
        output,
        header="time_ps P_RTI P_zonal"
    )
def ky_vs_omega():
    import numpy as np
    import matplotlib.pyplot as plt
    import glob


    # ============================================================
    # SETTINGS
    # ============================================================

    file_pattern = "*530A_0*"

    # Physical time between consecutive dump files, in ps
    dt_frame_ps = 1             # CHANGE THIS

    # Restrict displayed ky range if desired
    ky_max = 0.03                    # Angstrom^-1

    # Use log10 spectral power
    use_log = True


    # ============================================================
    # READ ONE DUMP
    # ============================================================

    def read_file(filename):

        with open(filename, "r") as f:

            f.readline()                  # ITEM: TIMESTEP
            timestep = int(f.readline())

            f.readline()                  # ITEM: NUMBER OF ATOMS
            N = int(f.readline())

            f.readline()                  # ITEM: BOX BOUNDS
            xlo, xhi = map(float, f.readline().split())
            ylo, yhi = map(float, f.readline().split())
            zlo, zhi = map(float, f.readline().split())

            header = f.readline().split()[2:]

            columns = {
                name: i for i, name in enumerate(header)
            }

            data = np.loadtxt(f)

        x  = data[:, columns["x"]]
        y  = data[:, columns["y"]]
        vx = data[:, columns["vx"]]

        return timestep, x, y, vx


    # ============================================================
    # GET FILES AND SORT BY TIMESTEP
    # ============================================================

    files = glob.glob(file_pattern)

    if len(files) == 0:
        raise RuntimeError(
            f"No files found matching {file_pattern}"
        )


    def get_timestep(filename):

        with open(filename, "r") as f:
            f.readline()
            return int(f.readline())


    files = sorted(files, key=get_timestep)

    Nt = len(files)

    print(f"Number of frames = {Nt}")


    # ============================================================
    # DETERMINE SPATIAL GRID
    # ============================================================

    _, x, y, vx = read_file(files[0])

    x_unique = np.sort(np.unique(x))
    y_unique = np.sort(np.unique(y))

    nx = len(x_unique)
    ny = len(y_unique)

    dx = np.mean(np.diff(x_unique))
    dy = np.mean(np.diff(y_unique))

    print(f"Grid = {nx} x {ny}")
    print(f"dx = {dx:.6f} A")
    print(f"dy = {dy:.6f} A")


    # ============================================================
    # SPATIAL WAVENUMBERS
    # ============================================================

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)

    idx_kx0 = np.argmin(np.abs(kx))


    # ============================================================
    # STORE Vx(kx=0, ky, t)
    # ============================================================

    # Complex because spatial FFT is complex
    Vx_zonal_time = np.zeros(
        (Nt, ny),
        dtype=complex
    )

    timesteps = np.zeros(Nt)


    # ============================================================
    # SPATIAL FFT AT EACH TIME
    # ============================================================

    for it, filename in enumerate(files):

        timestep, x, y, vx = read_file(filename)

        timesteps[it] = timestep

        # Sort onto regular y,x grid
        order = np.lexsort((x, y))

        vx_sorted = vx[order]

        vx_grid = vx_sorted.reshape(ny, nx)

        # Remove global bulk velocity
        vx_grid = vx_grid - np.mean(vx_grid)

        # 2D spatial Fourier transform
        Vx = np.fft.fft2(vx_grid)

        # Extract kx = 0
        Vx_zonal_time[it, :] = Vx[:, idx_kx0]

        print(
            f"{it+1:4d}/{Nt}   timestep = {timestep}"
        )


    # ============================================================
    # OPTIONAL: REMOVE TEMPORAL MEAN
    # ============================================================
    #
    # This removes the strictly stationary omega=0 component.
    #
    # IMPORTANT:
    # If you specifically want to see whether the zonal flow
    # is stationary, DO NOT do this.
    #
    # Uncomment only if interested in fluctuations around the
    # time-averaged zonal flow.
    #
    # Vx_zonal_time -= np.mean(
    #     Vx_zonal_time,
    #     axis=0,
    #     keepdims=True
    # )


    # ============================================================
    # TEMPORAL WINDOW
    # ============================================================
    #
    # Finite time records cause spectral leakage.
    # Hann window reduces this.
    # ============================================================

    window = np.hanning(Nt)

    Vx_windowed = (
        Vx_zonal_time
        * window[:, None]
    )


    # ============================================================
    # TEMPORAL FOURIER TRANSFORM
    # ============================================================

    V_kw = np.fft.fft(
        Vx_windowed,
        axis=0
    )


    # ============================================================
    # FREQUENCY AXIS
    # ============================================================
    #
    # np.fft.fftfreq gives cycles / ps.
    #
    # omega = 2*pi*f gives rad / ps.
    # ============================================================

    frequency = np.fft.fftfreq(
        Nt,
        d=dt_frame_ps
    )

    omega = 2.0 * np.pi * frequency


    # ============================================================
    # SHIFT FREQUENCY AXIS
    # ============================================================

    omega = np.fft.fftshift(omega)

    V_kw = np.fft.fftshift(
        V_kw,
        axes=0
    )


    # ============================================================
    # POWER
    # ============================================================

    power = np.abs(V_kw)**2


    # ============================================================
    # USE POSITIVE ky
    # ============================================================

    positive_ky = ky > 0

    ky_plot = ky[positive_ky]

    power = power[:, positive_ky]


    # Sort ky from low -> high

    order = np.argsort(ky_plot)

    ky_plot = ky_plot[order]

    power = power[:, order]


    # ============================================================
    # OPTIONAL ky LIMIT
    # ============================================================

    mask = ky_plot <= ky_max

    ky_plot = ky_plot[mask]

    power = power[:, mask]


    # ============================================================
    # LOG POWER
    # ============================================================

    if use_log:

        P_plot = np.log10(
            power + 1e-30
        )

    else:

        P_plot = power


    # ============================================================
    # PLOT
    # ============================================================

    plt.figure(figsize=(9, 7))

    mesh = plt.pcolormesh(
        ky_plot,
        omega,
        P_plot,
        shading="auto"
    )

    cbar = plt.colorbar(mesh)

    if use_log:

        cbar.set_label(
            r"$\log_{10}|\tilde{v}_x(0,k_y,\omega)|^2$"
        )

    else:

        cbar.set_label(
            r"$|\tilde{v}_x(0,k_y,\omega)|^2$"
        )


    plt.xlabel(
        r"$k_y$ ($\AA^{-1}$)"
    )

    plt.ylabel(
        r"$\omega$ (rad ps$^{-1}$)"
    )

    plt.title(
        r"$k_y-\omega$ spectrum of $v_x$ at $k_x=0$"
    )

    plt.tight_layout()

    plt.savefig(
        "zonal_ky_omega_spectrum.png",
        dpi=600
    )

    plt.show()


reader()