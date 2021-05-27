import torch
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvas
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.patches import Arrow, Rectangle
from matplotlib.transforms import Affine2D
from functools import lru_cache
import ffmpeg
import os


@lru_cache(maxsize=10)
def load_obstacles(idx: int) -> np.ndarray:
    return np.load("./data/car1order/obstacle_center/obs_{}.npy".format(idx))


def h264_converter(filename: str):
    ffmpeg.input(filename).output(filename+".new", vcodec="libopenh264", format="mp4").overwrite_output().run()
    os.remove(filename)
    os.rename(filename+".new", filename)


def visualize_local_map(filename: str, observations: torch.Tensor):
    observations = observations[:, :3*64*64] + 0.5
    observations = observations.view(-1, 3, 64, 64)
    observations = torch.movedim(observations, 1, 3).cpu().numpy().astype(np.uint8)*255
    _, H, W, _ = observations.shape
    writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 30., (W, H), True)
    for observation in observations:
        writer.write(observation)
    writer.release()
    h264_converter(filename)


def visualize_global_map(filename: str, obstacle_idx: int, observations: torch.Tensor, predictions: torch.Tensor):
    observations = observations[:, 3*64*64:].cpu().numpy()
    observations = observations * np.array([25.0, 35.0, np.pi, 25.0, 35.0, np.pi], dtype=np.float32) * 2.0

    predictions = predictions[:, 3*64*64:].cpu().numpy()
    predictions = predictions * np.array([25.0, 35.0, np.pi, 25.0, 35.0, np.pi], dtype=np.float32) * 2.0

    obs_center = load_obstacles(obstacle_idx)
    obstacles = [Rectangle((xy[0]-4, xy[1]-4), 8, 8) for xy in obs_center]
    obstacles = PatchCollection(obstacles, facecolor="gray", edgecolor="black")

    car_observation = PatchCollection([Rectangle((-1, -0.5), 2, 1), Arrow(0, 0, 2, 0, width=1.0)], facecolor="blue", edgecolor="blue")
    car_prediction = PatchCollection([Rectangle((-1, -0.5), 2, 1), Arrow(0, 0, 2, 0, width=1.0)], facecolor="yellow", edgecolor="yellow")

    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.subplots()
    ax.set_aspect('equal', 'box')
    ax.set_xlim([-25, 25])
    ax.set_ylim([-35, 35])

    canvas.draw()
    image = np.array(canvas.buffer_rgba())
    H, W, _ = image.shape
    writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 30., (W, H), True)

    T = observations.shape[0]
    for t in range(T):
        # obstacles
        ax.add_collection(obstacles)
        # target
        ax.scatter(observations[-1, 3], observations[-1, 4], marker='*', c="red")
        # current position
        transform = Affine2D().rotate(observations[t, 2]).translate(observations[t, 0], observations[t, 1])
        car_observation.set_transform(transform+ax.transData)
        ax.add_collection(car_observation)
        # predicted position
        transform = Affine2D().rotate(predictions[t, 2]).translate(predictions[t, 0], predictions[t, 1])
        car_prediction.set_transform(transform+ax.transData)
        ax.add_collection(car_prediction)
        # plot
        ax.set_xlim([-25, 25])
        ax.set_ylim([-35, 35])
        canvas.draw()
        image = np.array(canvas.buffer_rgba())[:, :, :3]
        writer.write(image)
        ax.clear()

    writer.release()
    h264_converter(filename)


def visualize_planning_result(filename: str, obstacle_idx: int, states: np.ndarray, goal: np.ndarray):
    obs_center = load_obstacles(obstacle_idx)
    obstacles = [Rectangle((xy[0]-4, xy[1]-4), 8, 8) for xy in obs_center]
    obstacles = PatchCollection(obstacles, facecolor="gray", edgecolor="black")

    car = PatchCollection([Rectangle((-1, -0.5), 2, 1), Arrow(0, 0, 2, 0, width=1.0)], facecolor="blue", edgecolor="blue")

    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.subplots()
    ax.set_aspect('equal', 'box')
    ax.set_xlim([-25, 25])
    ax.set_ylim([-35, 35])

    canvas.draw()
    image = np.array(canvas.buffer_rgba())
    H, W, _ = image.shape
    writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 30., (W, H), True)

    T = states.shape[0]
    for t in range(T):
        # obstacles
        ax.add_collection(obstacles)
        # target
        ax.scatter(goal[0], goal[1], marker='*', c="red")
        # current position
        transform = Affine2D().rotate(states[t, 2]).translate(states[t, 0], states[t, 1])
        car.set_transform(transform+ax.transData)
        ax.add_collection(car)
        # plot
        canvas.draw()
        image = np.array(canvas.buffer_rgba())[:, :, :3]
        writer.write(image)
        ax.clear()

    writer.release()
    h264_converter(filename)


if __name__ == "__main__":
    states = np.loadtxt("data/car1order/test_traj/path0.csv", delimiter=" ")
    goal = np.array([-14.6383, 10.3227, - 0.185641])
    visualize_planning_result("./data/car1order/test_traj/path0.mp4", 0, states, goal)
