from matplotlib import pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.transforms import Affine2D

obs_center = np.array([
    -3.361066383373184863e+00,
    -1.618362447483236366e+01,
    1.754212158819116496e+01,
    2.327441671685363644e+01,
    1.601334192118773814e+01,
    -2.177019678231786415e+01,
    -5.782483885379246402e+00,
    1.975485323546972438e+01,
    -2.041836004624516931e+01,
    5.669245497190956939e+00,
]).reshape(5, 2)
obstacles = [Rectangle((xy[0]-4, xy[1]-4), 8, 8) for xy in obs_center]
obstacles = PatchCollection(obstacles, facecolor="gray", edgecolor="black")


fig, ax = plt.subplots()

t_start = ax.transData
t = Affine2D().rotate_deg(-45)
t_end = t + t_start

car = Rectangle((-1, -0.5), 2, 1)
car.set_transform(t_end)
ax.add_patch(car)
ax.add_collection(obstacles)

ax.set_aspect('equal', 'box')
ax.set(xlim=[-25, 25], ylim=[-35, 35])
plt.savefig("temp.png")




# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import matplotlib as mpl
# fig = plt.figure()
# ax = fig.add_subplot(111)

# rect = patches.Rectangle((0.0120,0),0.1,1000)

# t_start = ax.transData
# t = mpl.transforms.Affine2D().rotate_deg(-45)
# t_end = t_start + t

# rect.set_transform(t_end)

# ax.add_patch(rect)

# plt.savefig("temp.png")
