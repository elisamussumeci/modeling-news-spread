import numpy as np
import csv
from scipy.stats import bernoulli

G = np.loadtxt(open('/home/elisa/Projetos/modeling-news-spread/charlie_results/graph_complete.csv'), delimiter=",")
nodes = np.loadtxt(open('/home/elisa/Projetos/modeling-news-spread/charlie_results/empirical_graph_nodes.csv'), delimiter=",")

# load list of domains in data
with open('/home/elisa/Projetos/modeling-news-spread/charlie_results/graph_original_domains_each_node.txt') as f:
    domains = f.read().splitlines()

# create dictionary, one colour to each domain
total_d = list(set(domains))
cols = colors.keys()
color_dict = {'a': 1}

for pos, i in enumerate(total_d):
    color_dict[i] = cols[pos]


#initial conditions
eig = np.linalg.eig(G)[0].max()

i0 = np.loadtxt(open('/home/elisa/Projetos/modeling-news-spread/charlie_results/i0.csv'), delimiter=",")
s0 = 1-i0
total_articles = len(i0)

def fun(t, y, pars):
    y = np.array(y)
    i,s = y[:total_articles],y[total_articles:]
    A, lamb = pars

    M =  (i * A).sum(axis=1)
    N = lamb * s
    Q = N*M

    dI = -i + Q
    dS = -Q

    return np.hstack((dI,dS))




def plot_sol(sol, color_dict, domains):
    plots = list_plot([(j[0],j[1][0]) for j in sol[:total_articles]], color=color_dict[domains[0]], plotjoined=True, alpha=.8, gridlines=true)
    for i in range(500):
        co = color_dict[domains[i]]
        plots += list_plot([(j[0], j[1][i]) for j in sol[:total_articles]], color=co, plotjoined=True, alpha=.2, gridlines=true)
    plots.save('/home/elisa/Projetos/modeling-news-spread/charlie_results/simulation.png')



## dI- matrix com a probabilidade do artigo ser infectado no tempo t. dI[0] - artigos infectados no tempo 0
## Infects - matrix boolean com os infectados no tempo t.
## recebe T.solution

def create_dI(sol):
    s = len(T.solution[0][1])/2
    dI = np.zeros((len(T.solution), s))
    dS = np.zeros((len(T.solution), s))
    c = 0
    for i,v in sol:
        dI[c:] = v[:s]
        dS[c:] = v[s:]
        c+=1
    return dI, dS

def create_Infects(dI,dS):
    dR = 1-(dS+dI)

    Infects = np.zeros(dI.shape)
    Infects[0] = dI[0]

    S0 = np.ones(dI.shape[1]) - Infects[0]
    R0 = np.zeros(dI.shape[1])
    I0 = Infects[0]

    for t in range(1,dI.shape[0]):

        I = bernoulli.rvs(dI[t]*S0)
        R = bernoulli.rvs(dR[t]*I0)

        Infects[t] = I0 - R + I
        a = Infects[t]

        if len(a[a<0]) > 0:
            b = I0-R
            if len(b[b<0])>0:
                print('ei')


        I0 = Infects[t]
        S0 = S0 - I

    return(Infects)


def create_infected_matrix(la, T):
    T.ode_solve(t_span=[0, 14], y_0=list(i0)+list(s0), num_points=16, params=[G, la])
    plot_sol(T.solution, color_dict, domains)

    dI, dS = create_dI(T.solution)
    Infects = create_infects(dI,dS)

    return dI, Infects


T = ode_solver()
T.algorithm = "rkf45"
T.function = fun
l = 0.0000215


dI, Infects = create_infected_matrix(l, T)
np.savetxt('/home/elisa/Projetos/modeling-news-spread/charlie_results/dI.csv', dI, delimiter=',')
np.savetxt('/home/elisa/Projetos/modeling-news-spread/charlie_results/Infects.csv', Infects, delimiter=',')