import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, sqrt, exp
import datetime


# Yukawa with short range repulsion correction 
def yukawa_SRR(r, Z, kappa, b, Z_C = 3.5, A = 14.400778):
    return ( Z**2 + (Z_C**2 - Z**2)*exp(-b*r))*(A/r)*exp(-kappa*r)
    
# Calculates semi-classical pair-potential  
def calculator(b,Z, kappa):

        r_array = np.arange(0.001, 10, 0.001)

        y_SRR_array = np.array([])

        for r in r_array:
            y_SRR = yukawa_SRR(r, Z, kappa, b)
            y_SRR_array = np.append(y_SRR_array, y_SRR)
        
        # plots various potentials, use for comparison 

        """
        plt.plot(r_array, TF_array, label="Thomas-Fermi", color="orange") 
        plt.plot(r_array, TF_SRR_array, linestyle = "--", color="orange", label="Thomas-Fermi + SRR") 
        plt.plot(r_array, PW_array, color = "green", label = "Perrot-Dharma-Wardana")
        

        plt.plot(r_array, PW_SRR_array,label=f"Z= {Z},b={b}")
        plt.suptitle(r"Pair potential, ionisation (Z) and short-range-repulsion wavevector (b) parameter scan (C-C, 5000 K, 0.914 Mbar, 2.429 $\rho$/$\rho_0$)")
        
        plt.ylim(0, 10)
        plt.xlabel("r (angstrom)")
        plt.ylabel("Energy (eV)")
        plt.legend()
        """
        
        dr = r_array[1] - r_array[0]
        F = -np.gradient(y_SRR_array, dr) # determines gradient of potential energy (Force)

        print("\n....................\n")
        print(f"Generating r, V and F for b={b}, Z={Z}, kappa={kappa}")
        print("\n....................\n")

        return r_array, y_SRR_array,  F


# writes potential to a file LAMMPS can read
def potential_writer( b, Z, kappa):
    filename = "/home/lcv510/Documents/cdt/york/lammps/LAMMPS/python_scripts/gaussian_process/veff.table"

    r, V, F = calculator(
                    b,
                    Z,
                    kappa
                    )
    
    with open(filename, "w") as f:

        f.write(f"# C-C, Z = {Z}, b={b}, kappa={kappa} - {datetime.datetime.now()} \n")
        f.write("MY_POTENTIAL\n")        # <-- table name used by pair_coeff
        f.write(f"N {len(r)} R 0.001 7.999 \n\n")

        for i in range(len(r)):
            f.write("%d %f %f %f\n" % (i+1, r[i], V[i], F[i]))
    
    print("\n....................\n")
    print("LAMMPS table written to:", filename)
    print("\n....................\n")

