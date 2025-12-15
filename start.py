import torch
import numpy as np
import torch.nn as nn
torch.manual_seed(0)
import matplotlib.pyplot as plt


class Pbit(nn.Module):
    def __init__(self,beta=1.0,bias=0.0):
        super().__init__()
        self.beta=beta
        self.bias=bias
        self.state=torch.randint(0,2,(1,)).float()*2-1

    def forward(self,x):
        I=self.beta*x+self.bias
        p=torch.tanh(I)
        noise=torch.rand_like(I)*2-1
        m=torch.sign(p-noise)
        self.state=m.detach()
        return m,p


class PCircuit(nn.Module):
    def __init__(self, J, h=None, beta=1.0):
        super().__init__()
        self.num_pbits=J.shape[0]
        self.J=J
        if h is None:
            self.h=torch.zeros(self.num_pbits)
        else:
            self.h=h
        self.beta=beta
        self.m=torch.randint(0,2,(self.num_pbits,)).float()*2-1

    def forward(self,steps=1,noise_type="continuous"):
        for _ in range(steps):
            i=torch.randint(0, self.num_pbits, (1,)).item()
            I_i=torch.dot(self.J[i],self.m)+self.h[i]
            p_i=torch.tanh(self.beta*I_i)

            if noise_type == "continuous":
                r=torch.rand(1)*2-1
            else:
                r=torch.randint(0,2,(1,)).float()*2-1
            
            m_i=torch.sign(p_i-r)
            if m_i == 0:
                m_i = torch.tensor(1.0)

            self.m[i]=m_i.detach()
        return self.m

    def energy(self):
        return -0.5 * self.m @ self.J @ self.m - self.h @ self.m


N=10
beta=1.0
h=torch.zeros(N)

J=torch.randn(N, N)
J=(J+J.T)/2
J.fill_diagonal_(0)

T=5000

# ----- continuous noise -----
torch.manual_seed(0)
circuit_cont=PCircuit(J, h, beta)
energies_cont=[]

for t in range(T):
    circuit_cont.forward(steps=1, noise_type="continuous")
    energies_cont.append(circuit_cont.energy().item())

# ----- discrete noise -----
torch.manual_seed(0)
circuit_disc=PCircuit(J, h, beta)
energies_disc=[]

for t in range(T):
    circuit_disc.forward(steps=1, noise_type="discrete")
    energies_disc.append(circuit_disc.energy().item())

# ----- plot -----
plt.figure(figsize=(7,4))
plt.plot(energies_cont, label="Continuous noise", alpha=0.8)
plt.plot(energies_disc, label="Discrete noise", alpha=0.8)
plt.xlabel("Update steps")
plt.ylabel("Energy")
plt.title("Energy convergence: Continuous vs Discrete noise")
plt.legend()
plt.show()


#Mean and Variance comparision

import numpy as np
import matplotlib.pyplot as plt

# ---- discard burn-in ----
burn_in = int(0.3 * len(energies_cont))

E_cont_ss = np.array(energies_cont[burn_in:])
E_disc_ss = np.array(energies_disc[burn_in:])

# ---- statistics ----
mean_cont = E_cont_ss.mean()
std_cont  = E_cont_ss.std()

mean_disc = E_disc_ss.mean()
std_disc  = E_disc_ss.std()

# ---- bar plot ----
labels = ["Continuous noise", "Discrete noise"]
means  = [mean_cont, mean_disc]
stds   = [std_cont, std_disc]

plt.figure(figsize=(6,4))
plt.bar(labels, means, yerr=stds, capsize=8)
plt.ylabel("Energy")
plt.title("Steady-state energy statistics")
plt.show()

#Energy Histogram

plt.figure(figsize=(7,4))

plt.hist(E_cont_ss, bins=30, alpha=0.7, label="Continuous noise")
plt.hist(E_disc_ss, bins=30, alpha=0.7, label="Discrete noise")

plt.xlabel("Energy")
plt.ylabel("Frequency")
plt.title("Distribution of visited energies")
plt.legend()
plt.show()


def random_graph(n, p=0.5, weight_scale=1.0):
    W=torch.zeros(n, n)
    for i in range(n):
        for j in range(i+1, n):
            if torch.rand(1)<p:
                w=weight_scale*torch.rand(1)
                W[i, j]=w
                W[j, i]=w
    return W

def maxcut_J(W):
    return -W

def cut_value(m, W):
    cut=0.0
    n=len(m)
    for i in range(n):
        for j in range(i+1, n):
            cut+=0.5* W[i, j]*(1-m[i]*m[j])
    return cut

##thanks gpt for the helpful functions !
# ---- problem setup ----
n = 12
W = random_graph(n)
J = maxcut_J(W)
h = torch.zeros(n)

circuit = PCircuit(J, h, beta=1.0)

# ---- run dynamics ----
T = 4000
energies = []
cuts = []

best_cut = -1
best_state = None
beta_min = 0.1
beta_max = 3.0

for t in range(T):
    # --- annealing schedule ---
    beta_t = beta_min + (beta_max - beta_min) * t / T
    circuit.beta = beta_t

    circuit.forward(steps=1, noise_type="continuous")
    
    E = circuit.energy().item()
    C = cut_value(circuit.m, W)
    
    energies.append(E)
    cuts.append(C)
    
    if C > best_cut:
        best_cut = C
        best_state = circuit.m.clone()
plt.figure(figsize=(6,4))
plt.plot(energies)
plt.xlabel("Time step")
plt.ylabel("Ising energy")
plt.title("MAXCUT via p-bit dynamics")
plt.show()

plt.figure(figsize=(6,4))
plt.plot(cuts)
plt.xlabel("Time step")
plt.ylabel("Cut value")
plt.title("Cut improvement over time")
plt.show()

def random_cut(W, trials=1000):
    n = W.shape[0]
    best = 0
    for _ in range(trials):
        m = torch.randint(0,2,(n,)).float()*2 - 1
        C = cut_value(m, W)
        best = max(best, C)
    return best

rand_best = random_cut(W)
print("Best p-bit cut :", best_cut)
print("Best random cut:", rand_best)
max_possible = W.sum().item() / 2
print("Cut fraction:", best_cut / max_possible)
print("Random cut fraction:", rand_best / max_possible)
