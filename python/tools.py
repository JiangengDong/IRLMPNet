# This script contains many auxilliary functions that only used once in this project.
# They are mainly used to convert files between different types.

import pickle
import numpy as np
import os
import tqdm
from matplotlib import pyplot as plt


def load_pickle(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        return pickle.load(f)


def move_obstacle_files():
    for env_name in ("acrobot", "car", "cartpole", "quadrotor"):
        print("moving ", env_name)
        work_dir = os.path.join("data", "obstacle", "%s_obs" % env_name)
        input_dir = os.path.join(work_dir, "pkl")
        output_dir = os.path.join(work_dir, "csv")
        os.makedirs(input_dir)
        os.makedirs(output_dir)

        # move pickle file into pkl folder
        for i in range(10):
            os.rename(os.path.join(work_dir, "obc_%d.pkl" % i), os.path.join(input_dir, "obc_%d.pkl" % i))
            os.rename(os.path.join(work_dir, "obs_%d.pkl" % i), os.path.join(input_dir, "obs_%d.pkl" % i))


def convert_obstacle_pkl_to_csv():
    for env_name in ("acrobot", "car", "cartpole", "quadrotor"):
        print("converting ", env_name)
        work_dir = os.path.join("data", "obstacle", "%s_obs" % env_name)
        input_dir = os.path.join(work_dir, "pkl")
        output_dir = os.path.join(work_dir, "csv")

        # convert to csv
        for i in tqdm.tqdm(range(10)):
            input_filename = os.path.join(input_dir, "obs_%d.pkl" % i)
            output_filename = os.path.join(output_dir, "obs_%d.csv" % i)
            data = load_pickle(input_filename)
            np.savetxt(output_filename, data, delimiter=",")


def convert_obstacle_pkl_to_npy():
    for env_name in ("acrobot", "car", "cartpole", "quadrotor"):
        print("converting ", env_name)
        work_dir = os.path.join("data", "point_cloud", "%s" % env_name)

        # read pickles
        voxels = []
        for i in tqdm.tqdm(range(10)):
            input_filename = os.path.join(work_dir, "obc_%d.pkl" % i)
            data = load_pickle(input_filename)
            voxels.append(data)
        voxels = np.stack(voxels)
        np.save(os.path.join(work_dir, "obc_all.npy"), voxels)
        print(voxels.shape)


def plot_pcd():
    input_filename = os.path.join("data", "point_cloud", "car", "obc_0.pkl")
    data = load_pickle(input_filename)
    plt.scatter(data[:, :, 0].flatten(), data[:, :, 1].flatten())
    plt.savefig("data/temp_visualize/point_cloud.pdf")
    plt.cla()


def convert_point_cloud_to_voxel():
    env_name = "car"
    work_dir = os.path.join("data", "point_cloud", "%s" % env_name)

    env_id = 0
    # pcd = load_pickle(os.path.join(work_dir, "obc_%d.pkl" % env_id))
    pcd = np.load(os.path.join(work_dir, "car_obs_env_vox.npy"))
    # TODO: complete this function later
    pass


def split_voxel_npy_to_voxel_csv():
    work_dir = os.path.join("data", "voxel", "car")
    data = np.load(os.path.join(work_dir, "car_voxel.npy"))
    for env_id in tqdm.tqdm(range(10)):
        np.savetxt(os.path.join(work_dir, "voxel_%d.csv" % env_id), data[env_id, 0], delimiter=",")

    
def view_path_data():
    path_data = np.load("data/training_path/cartpole/cartpole_obs_path_data.npy")
    gt_data = np.load("data/training_path/cartpole/cartpole_obs_gt.npy")
    pass # use debugger here


if __name__ == "__main__":
    data = load_pickle("./data/traj/car/0/path_0.pkl")
    pass
