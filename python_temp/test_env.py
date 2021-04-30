from KinoDynSys import Car1OrderSystem
import numpy as np
from matplotlib import pyplot as plt

# np.set_printoptions(threshold=np.inf)

state = np.array([11.54, 23.27, 0], dtype=np.float32)
state2 = np.array([10.54, 23.27, 0], dtype=np.float32)
control = np.array([-1.0, 0.0], dtype=np.float32)
for i in range(10):
    system = Car1OrderSystem()
    system.propagate(state, control, state2, 1.0)
    state = state2
    local_map = system.get_local_map(state)
    local_map = np.transpose(np.broadcast_to(local_map, (3, 64, 64)), [1, 2, 0]).astype(np.uint8)*255
    plt.imshow(local_map, cmap="gray", vmin=0, vmax=1)
    plt.savefig("data/local_map/{}.png".format(i))
