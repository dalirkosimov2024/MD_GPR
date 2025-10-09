from astropy import units as u
from plasmapy.formulary.collisions import impact_parameter, Coulomb_logarithm
from plasmapy.formulary.lengths import Debye_length
import numpy as np
import math
from astropy.constants import m_p, m_e, k_B , eps0, e, hbar
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# constants 
q_e = e.to(u.C)

# Laser parameters
#I_l = 1 #1e15 W/cm^3
laser_lambda = 0.35 # microns

# Target parameters
rho_0 = 1e3  * u.kg / u.m**3 # kg/m3, CH
P_0 = 1e5 * u.Pa # Pa
gamma = 5/3 # ideal gas
fractions = {"H": 0.5, "C":0.5}
Z = {"H":1, "C":6}
atomic_mass = {"H": 1, "C": 12}
Zbar = sum(fractions[i] * Z[i] for i in fractions)
m_i = sum(fractions[i] * atomic_mass[i] for i in fractions) * m_p

# Ablation pressure 
def ablation_pressure(I_l, laser_lambda):
    P_ablation = 57 * (I_l / laser_lambda) ** (2/3) # Mbar
    P_ablation = P_ablation * 1e11 #Pa
    return P_ablation * u.Pa

def shock_pressure(P_ablation):
    P_1 = P_ablation 
    return P_1

def hugoniot(P_1, P_0, rho_0, gamma):
    rho_1 = (P_1 * (gamma +1) + P_0 * (gamma -1) ) / (P_0 * (gamma +1) + P_1 * (gamma -1) ) * rho_0 
    return rho_1

def number_density(rho_1, m_i):
    n_CH = rho_1 / m_i  
    n_e = Zbar * n_CH
    n_total = n_CH + n_e
    return n_CH, n_e, n_total

def temperature(P_1,n_total):
    return (P_1 / (k_B * n_total)).to(u.K)  # kelvin

def kelvin_to_eV(T):
    return (k_B * T).to(u.eV)

def wigner_seitz_radius(n):
    return (3 / (4 * np.pi * n))**(1/3)

def coupling_parameter(r, T):
    z_prod = math.prod(Z[i] for i in Z)
    return (z_prod * q_e**2)/(4 * np.pi * eps0 * r * k_B * T)
    
def debye_length(T, n_e):
    return Debye_length(T, n_e)

def plasma_parameter(lambda_e, n_e):
    return 4 * np.pi * n_e* lambda_e**3

def fermi_energy(n_e):
    return ((hbar)**2 / (2 * m_e)) * (3 * np.pi**2 * n_e) ** (2/3)

def fermi_temperature(E_f):
    return (E_f / k_B)

def degeneracy_parameter(T, T_f):
    return T / T_f

def thomas_fermi_screening(E_f, n_e):
    k_s =  ( (3 * q_e**2) / (2 * eps0 * E_f) * n_e ) **0.5 # screening paramter
    return 1 / k_s

def tokamak_emulator(): 

    T_T_list = np.array(np.arange(5e5, 2e6, 5e4))
    Gamma_list = np.array([])
    theta_F_list = np.array([])
    P_ablation_list = np.array([])
    lambda_e_list = np.array([])
    ion_density_list = np.array([])
    n_e_list =  np.array([])
    Lambda_list = np.array([])
    T_eV_list = np.array([])
    E_f_list = np.array([])
    wsr_list = np.array([])

    n_eT = 1e21 /u.m**3 # assume n_e = n_i
    #T_T = 2e6 * u.K
    for T_T in T_T_list:
        T_T = T_T*u.K
        P_T = (k_B * n_eT * T_T)
        lambda_T = debye_length(T_T, n_eT)
        wg_radius_T = wigner_seitz_radius(n_eT)
        E_fT = fermi_energy(n_eT) 
        T_fT = fermi_temperature(E_fT) 
        Gamma_T = coupling_parameter(wg_radius_T, T_T)
        Lambda_T = plasma_parameter(lambda_T, n_eT)
        theta_fT = degeneracy_parameter(T_T, T_fT)

        Gamma_list = np.append(Gamma_list, Gamma_T)
        theta_F_list = np.append(theta_F_list, theta_fT)
        lambda_e_list = np.append(lambda_e_list, lambda_T.to_value(u.pm))
        n_e_list = np.append(n_e_list, (n_eT).to_value(1/u.m**3))
        wsr_list = np.append(wsr_list, wg_radius_T.to_value(u.pm))
        Lambda_list = np.append(Lambda_list, Lambda_T)
        E_f_list = np.append(E_f_list, E_fT.to_value(u.eV))
    
    plotter(T_T_list, n_e_list,
             wsr_list, lambda_e_list,
               Lambda_list, Gamma_list, theta_F_list,
               T_T_list, E_f_list)

    print(f"Electron density: {n_eT}\n"
          f"Electron temperature: {T_T}\n"
          f"Plasma pressure: {P_T}\n"
          f"Debye length: {lambda_T}\n"
          f"Wigner Seitz radius: {wg_radius_T}\n"
          f"Plasma parameter: {Lambda_T}\n"
          f"Coupling parameter: {Gamma_T}\n"
          f"Fermi temperature: {T_fT}\n"
          f"Degeneracy parameter: {theta_fT}")

def plotter(P_ablation_list, n_e_list,
             wg_radius_ion_list, tf_length_list,
               Lambda_list, Gamma_list, theta_F_list,
               T_eV_list, E_f_list):
    
    fmt = ticker.FuncFormatter(lambda x, _: f"{x:.4f}")

    fig, axs = plt.subplots(4,2)
    axs[0,0].plot(P_ablation_list, n_e_list/1e30)
    axs[0,0].set_ylabel(r"$n_e$ ($10^{30} m^{-3}$)")
    axs[0,0].yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    axs[0,1].plot(P_ablation_list, wg_radius_ion_list)
    axs[0,1].set_ylabel("Wigner-Seitzs radius (pm)")
    axs[0,1].yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    axs[1,0].plot(P_ablation_list, tf_length_list)
    axs[1,0].set_ylabel("Thomas-Fermi screening length (pm)")
    #axs[1,0].set_yscale("log")
    axs[1,0].yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    axs[1,1].plot(P_ablation_list, Lambda_list)
    axs[1,1].set_ylabel(r"Plasma parameter $\Lambda$")
    axs[1,1].set_yscale("log")
    axs[2,0].plot(P_ablation_list, Gamma_list)
    axs[2,0].set_yscale("log")
    axs[2,0].set_ylabel(r"Coupling parameter $\Gamma$ ($C^2$ / FJ)")
    axs[2,1].plot(P_ablation_list, theta_F_list)
    axs[2,1].set_yscale("log")
    axs[2,1].set_ylabel(r"Degeneracy parameter $\theta_F$")
    axs[3,0].plot(P_ablation_list, T_eV_list)
    axs[3,0].set_yscale("log")
    axs[3,0].set_ylabel("Temperature (eV)")
    axs[3,1].plot(P_ablation_list, E_f_list)
    axs[3,1].set_ylabel("Fermi energy (eV)")
    axs[3,1].yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))

    for ax in axs.flat:
        ax.set_xscale("log")
        ax.set(xlabel ='Ablation Pressure (Pa)')

    plt.tight_layout()
    plt.show()

def yukawa_potential_plot(tf_length, wg_radius_ion):
    
    print(tf_length)
    kappa = 1 / (tf_length )
    kappa = kappa
    print(kappa)
    A0 = 14.3996 #eV*Angstrom
    A_cc = 36 * A0
    A_hh = A0
    A_ch = 6 * A0
    
    r = np.arange(0, 2, 0.005)

    
    def V_yukawa(r, A, kappa):
        return A / r * np.exp(-kappa * r)
    
    V_cc = V_yukawa(r, A_cc, kappa)
    V_hh = V_yukawa(r, A_hh, kappa)
    V_ch = V_yukawa(r, A_ch, kappa)



    plt.plot(r, V_cc, label = "C-C")    
    plt.plot(r, V_ch, label = "C-H")    
    plt.plot(r, V_hh, label = "H-H")   
    plt.axvline(x = tf_length, color="red", linestyle = "--", label = r"$\lambda_{TF}$")
    plt.axvline(x = wg_radius_ion, color="blue", linestyle = "--", label = "Wigner-Seitz radius")
    plt.legend()
    plt.ylim(0,200)
    plt.xlabel(r"r ($\AA$)", fontsize=16)
    plt.ylabel(r"$V_\text{yukawa}$ (eV)", fontsize=16)
    plt.show()


    

if __name__ == "__main__":



    if False:
        #tokamak_emulator()
        value = (3**(1/3) * m_e * q_e**2) / (np.pi**(4/3) *hbar**2 *eps0)
        print(value/1e9)

    else:

        I_l_list = np.array(np.arange(0.001, 0.1, 0.001))
        I_l_list = list(reversed(I_l_list))
        Gamma_list = np.array([])
        theta_F_list = np.array([])
        P_ablation_list = np.array([])
        lambda_e_list = np.array([])
        ion_density_list = np.array([])
        n_e_list =  np.array([])
        Lambda_list = np.array([])
        T_eV_list = np.array([])
        E_f_list = np.array([])
        wsr_list = np.array([])
        tf_length_list = np.array([])

        for I_l in I_l_list:
            P_ablation = ablation_pressure(I_l,laser_lambda)
            P_1 = shock_pressure(P_ablation)
            rho_1 = hugoniot(P_1, P_0, rho_0, gamma)
            n_CH, n_e, n_total = number_density(rho_1, m_i) 
            T = temperature(P_1, n_total) 
            T_eV = kelvin_to_eV(T)
            wg_radius_ion = wigner_seitz_radius(n_CH)
            wg_radius_e = wigner_seitz_radius(n_e)
            E_f = fermi_energy(n_e)
            T_f = fermi_temperature(E_f)
            theta_F = degeneracy_parameter(T.to_value(u.K), T_f.to_value(u.K))
            lambda_e = debye_length(T, n_e)
            Lambda = plasma_parameter(lambda_e, n_e)
            Gamma = coupling_parameter(wg_radius_ion, T)
            tf_length = thomas_fermi_screening(E_f, n_e)

            Gamma_list = np.append(Gamma_list, Gamma)
            theta_F_list = np.append(theta_F_list, theta_F)
            P_ablation_list = np.append(P_ablation_list, P_ablation.to_value(u.Pa))
            lambda_e_list = np.append(lambda_e_list, lambda_e.to_value(u.pm))
            ion_density_list = np.append(ion_density_list, (n_CH/1e30).to_value(1/u.m**3))
            n_e_list = np.append(n_e_list, (n_e).to_value(1/u.m**3))
            wsr_list = np.append(wsr_list, wg_radius_ion.to_value(u.pm))
            Lambda_list = np.append(Lambda_list, Lambda)
            T_eV_list = np.append(T_eV_list, T_eV.to_value(u.eV))
            E_f_list = np.append(E_f_list, E_f.to_value(u.eV))
            tf_length_list = np.append(tf_length_list, tf_length.to_value(u.Angstrom))

            

        #plotter(P_ablation_list, n_e_list, 
        #        wsr_list, tf_length_list, 
        #        Lambda_list, Gamma_list, 
        #        theta_F_list, T_eV_list, E_f_list)

        print(f"\n  ---------COLLISIONALITY SCRIPT------------ \n \n \n"
            f"Abltion pressure: {round(P_ablation.to(u.GPa), 3)} \n"
            f"Shock pressure: {round(P_1.to(u.TPa),3)} \n"
            f"Density behind shock: {round(rho_1,3)}\n"
            f"Ion number denisty: {round((n_CH/1e30),3)}\n"
            f"Electron number denisty: {round((n_e/1e30),3)}\n"
            f"Temperature behind shock: {round(T_eV,3)}\n\n"
            f"Thomas-Fermi screening length: {round(tf_length.to(u.AA),3)}\n"
            f"Wigner-Seitz radius: {round(wg_radius_ion.to(u.pm),3)} \n"
            f"Coupling parameter: {round(Gamma,3)}\n"
            f"Fermi temperature: {round(T_f.to_value(u.K)/11604.52521,3)} eV \n"
            f"Degeneracy parameter: {round(theta_F,3)} \n"
            "\n \n \n")
        
        yukawa_potential_plot(tf_length.to_value(u.Angstrom), wg_radius_ion.to_value(u.Angstrom))