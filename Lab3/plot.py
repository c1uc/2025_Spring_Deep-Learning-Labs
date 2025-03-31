from models.VQGAN_Transformer import MaskGit
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

xs = np.linspace(0, 1, 20)

gamma_funcs = [
    MaskGit.gamma_func(mode="linear"),
    MaskGit.gamma_func(mode="cosine"),
    MaskGit.gamma_func(mode="square"),
]

ys = [[gamma_func(x) for x in xs] for gamma_func in gamma_funcs]

sns.lineplot(x=xs, y=ys[0], label="linear")
sns.lineplot(x=xs, y=ys[1], label="cosine")
sns.lineplot(x=xs, y=ys[2], label="square")
plt.legend()
plt.savefig("gamma_funcs.png")
plt.show()
