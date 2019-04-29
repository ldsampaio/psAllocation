from scenarios import *

########################################################################################################################
# Parameters
########################################################################################################################

# Cluster Size (1 to 4 works)
C = 2
# Base Stations (1 per cell)
L = chn(C)
# Cell Radius
R = 100
# Users per Cell
K = 10
# Available Pilot Sequences
Tp = K
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
# PSO Population Size
psize = 25
# PSO Max Iteration
maxit = 1000

########################################################################################################################
# Auxiliary Functions
########################################################################################################################
def fitness(phi,beta,sigma):
    f = np.zeros((len(phi), len(phi[0][0])))
    for ell in range(0, len(phi[0][0])):
        for k in range(0, len(phi)):
            f[k, ell] = beta[k, ell, ell]
            deno = 0
            for j in range(0, len(phi[0][0])):
                for kline in range(0, len(phi)):
                    if j != ell:
                        deno += np.inner(phi[k, :, ell], phi[kline, :, j])*beta[kline, j, ell]
            deno += sigma
            f[k, ell] /= deno
    return np.sum(f)

def firstGeneration(phi):
    for ell in range(0, len(phi[0][0])):
        q = list(range(0, len(phi[0])))
        np.random.shuffle(q)
        phi[range(0, len(phi[0])), q, ell] = 1
    return phi

def pso(beta,sigma,psize,maxit):
    pop = np.zeros((len(beta), len(beta), len(beta[0]), int(psize)))
    v = pop
    f = np.zeros((int(psize), 1))
    for p in range(0, psize):
        pop[:, :, :, p] = firstGeneration(pop[:, :, :, p])
        f[p] = fitness(pop[:, :, :, p], beta, sigma)
    # Individual Best Fitness Values
    lbest = f
    # Individual Bests
    plbest = pop
    # Global Best Fitness Values
    gbest = np.zeros((int(maxit)+1, 1))
    gbest[0] = f.max()
    # Global Best
    pgbest = pop[:, :, :, f.argmax()]

    for i in range(1, maxit+1):
        for p in range(0, psize):

            # Calculating particle velocity
            v[:, :, :, p] = v[:, :, :, p] + np.random.uniform(0, 1) * (pop[:, :, :, p] - plbest[:, :, :, p])\
                            + np.random.uniform(0, 1) * (pop[:, :, :, p] - pgbest)

            # Calculating the probability of binary change according to velocity for each dimension in a single particle
            cprob = 1/(1 + np.exp(-v[:, :, :, p]))

            for d1 in range(0, len(v)):
                for d2 in range(0, len(v[0])):
                    for d3 in range(0, len(v[0][0])):
                        # Tmp is a random value between 0 and 1
                        tmp = np.random.uniform(0,1)
                        # Test for changing
                        if tmp < cprob[d1, d2, d3]:
                            pop[d1, d2, d3, p] = 1
                        else:
                            pop[d1, d2, d3, p] = 0

            # Discarding unfeasible candidates
            if np.sum(pop[:, :, :, p].ravel()) > len(pop) * len(pop[0][0]):
                pop[:, :, :, p] = np.zeros((len(pop), len(pop[0]), len(pop[0][0])))
                pop[:, :, :, p] = firstGeneration(pop[:, :, :, p])

            # Recalculate the Fitness
            f[p] = fitness(pop[:, :, :, p], beta, sigma)
            # Update Individual Bests
            if f[p] > lbest[p]:
                lbest[p] = f[p]
                plbest[:, :, :, p] = pop[:, :, :, p]
            # Update Global Best
            if f[p] > gbest[i-1]:
                gbest[i] = f[p]
                pgbest = pop[:, :, :, p]
            else:
                gbest[i] = gbest[i-1]
        print(i)
    return pgbest, gbest



########################################################################################################################
# Main Script
########################################################################################################################

k_x, k_y, c_x, c_y, d, tmp, beta = create_scenario(C, R, K, Fmin, Fmax, S, gamma, W, Bmax, d0)
drawScenario(C, R, K, c_x, c_y, k_x, k_y)

phi = np.zeros((int(K), int(Tp), int(L)))
phi = firstGeneration(phi)

f = fitness(phi, beta, N0*Bmax)
print(f)
phi, gbest = pso(beta, N0*Bmax, psize, maxit)

plt.figure()
plt.plot(range(0, maxit+1), gbest, '--r')
plt.show()