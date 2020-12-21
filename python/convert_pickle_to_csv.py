import pickle
import numpy as np
import os
import tqdm

def load(path: str) -> np.ndarray: 
    with open(path, 'rb') as f:
        return pickle.load(f)

def move_files():
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

def main():
    for env_name in ("acrobot", "car", "cartpole", "quadrotor"):
        print("converting ", env_name)
        work_dir = os.path.join("data", "obstacle", "%s_obs" % env_name)
        input_dir = os.path.join(work_dir, "pkl")
        output_dir = os.path.join(work_dir, "csv")
        
        # convert to csv 
        for i in tqdm.tqdm(range(10)):
            input_filename = os.path.join(input_dir, "obs_%d.pkl" % i)
            output_filename = os.path.join(output_dir, "obs_%d.csv" % i)
            data = load(input_filename)
            np.savetxt(output_filename, data, delimiter=",")

if __name__ == "__main__":
    main()