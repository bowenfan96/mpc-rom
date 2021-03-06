import matplotlib.pyplot as plt
import numpy as np

bpod_u0 = [273.000001,
           373,
           360.2492412,
           296.1008372,
           294.5995899,
           286.5524599,
           283.5063943,
           281.7249347,
           281.0282384,
           279.6974883,
           275.8649638]

bpod_u1 = [273.000001,
           373,
           373,
           373,
           373,
           373,
           373,
           373,
           372.7735169,
           362.835462,
           296.7658734]

fom_u0 = [272.998579,
          373,
          340.9911429,
          294.699677,
          291.4684802,
          285.172999,
          282.4592974,
          281.6310748,
          281.6306936,
          281.331481,
          277.6932722]

fom_u1 = [272.998579,
          373,
          373,
          373,
          373,
          369.0772557,
          356.0310018,
          357.928045,
          357.1750968,
          360.5028503,
          312.2894653]

t = np.linspace(0, 11, 11)
plt.step(t, fom_u0)
plt.step(t, bpod_u0)
plt.step(t, fom_u1)
plt.step(t, bpod_u1)

plt.show()