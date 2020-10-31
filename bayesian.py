import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataframe = pd.read_pickle("./data/dataset.pkl")

likes = dataframe["likes"].to_numpy()

p, bin_edges = np.histogram(likes, density=False, bins=50)

fig, ax = plt.subplots()

plt.title("marginal probability distribution")
plt.hist(bin_edges[:-1],bin_edges,weights=p)
plt.xlabel("likes")
plt.ylabel("count")

plt.savefig("./data/figures/marginal_prob_distr.png")
plt.show()

