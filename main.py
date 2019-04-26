from scenarios import *

########################################################################################################################
# Parameters
########################################################################################################################

# Cluster Size (1 to 4 works)
C = 2
# Base Stations (1 per cell)
BS = chn(C)
# Cell Radius
R = 100
# Users per Cell
K = 10
# Min Transmission Frequency
Fmin = 9E8
# Max Transmission Frequency
Fmax = 5E9
# Shadowing variance (in dB)
S = 4
# Path Loss Exponent
gamma = 2
# Number of Transmission Bands
W = 1
# Maximum Bandwidth per Carrier
Bmax = 20E6
# Noise Power Spectrum Density
N0 = 4.11E-21
# Reference distance (1 m for indoor and 10 m for outdoor)
d0 = 10

k_x, k_y, c_x, c_y, d, tmp, beta = create_scenario(C, R, K, Fmin, Fmax, S, gamma, W, Bmax, d0)
drawScenario(C, R, K, c_x, c_y, k_x, k_y)
