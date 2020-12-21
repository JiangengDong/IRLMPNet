from PIL import Image, ImageDraw
import numpy as np
from functools import lru_cache

default_param = {
    "vehicle_radius": 0.3,
    "wheel_base": 0.12,
    "wheel_radius": 0.1,
    "wheel_width": 0.05
}


def plot_differential_drive(img: Image.Image, pos: np.ndarray, color: list, dx: float=0.01, param: dict=default_param):
    car_image = get_differential_drive_image(**param, dx=dx)
    rotated_car_image = car_image.rotate(-pos[2]/np.pi*180, resample=Image.BILINEAR)

    vehicle_radius = int(param["vehicle_radius"]//dx)
    center = np.rint(pos[0:2]/dx).astype(np.int) + np.array(img.size)//2
    d = ImageDraw.Draw(img)
    d.bitmap(tuple(center - vehicle_radius), rotated_car_image, fill=color)
    return img


@lru_cache(maxsize=3)
def get_differential_drive_image(vehicle_radius: float, wheel_base: float, wheel_radius: float, wheel_width: float, dx: float):
    # convert to pixel coordinate
    vehicle_radius = int(vehicle_radius//dx)
    wheel_base = int(wheel_base//dx)
    wheel_radius = int(wheel_radius//dx)
    wheel_width = int(wheel_width//dx)

    # create image
    img_width = 2*vehicle_radius
    img = Image.new("RGBA", (img_width, img_width))

    # plot frame
    d = ImageDraw.Draw(img)
    d.chord(xy=[2, 2, img_width-2, img_width-2], start=30, end=330, width=5)

    # plot wheels
    d.rectangle(
        xy=[
            vehicle_radius-wheel_radius, vehicle_radius-wheel_base-wheel_width,
            vehicle_radius+wheel_radius, vehicle_radius-wheel_base+wheel_width
        ],
        width=5)
    d.rectangle(
        xy=[
            vehicle_radius-wheel_radius, vehicle_radius+wheel_base-wheel_width,
            vehicle_radius+wheel_radius, vehicle_radius+wheel_base+wheel_width
        ],
        width=5)

    return img


if __name__ == "__main__":
    img = Image.new("RGB", (400, 400), color=(255, 255, 255))

    pos = np.array([0.0, 0.0, np.pi])
    plot_differential_drive(img, pos, (0, 255, 0))

    img.save("data/img/car.png")
