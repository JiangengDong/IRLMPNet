import os
import pickle

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import utils


class VoxelEncoder(nn.Module):
    def __init__(self, input_size, output_size, in_channels):
        super(VoxelEncoder, self).__init__()
        input_size = [input_size, input_size]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3)),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3, 3)),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        with torch.no_grad():
            x = self.encoder(torch.autograd.Variable(torch.rand([1, in_channels] + input_size)))
        first_fc_in_features = 1
        for n in x.size()[1:]:
            first_fc_in_features *= n
        self.head = nn.Sequential(
            nn.Linear(first_fc_in_features, 64),
            nn.PReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


class PNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(PNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512), nn.PReLU(),
            nn.Linear(512, 256), nn.PReLU(),  # nn.Dropout(),
            nn.Linear(256, 128), nn.PReLU(),  # nn.Dropout(),
            nn.Linear(128, 64), nn.PReLU(), nn.Dropout(),
            nn.Linear(64, 32), nn.PReLU(), nn.Dropout(),
            nn.Linear(32, output_size))

    def forward(self, x):
        out = self.fc(x)
        return out


class MPNet(torch.jit.ScriptModule):
    def __init__(self, ae_input_size=32, ae_output_size=64,
                 in_channels=1,
                 state_size=3,
                 control_size=0):
        super(MPNet, self).__init__()
        self.state_size = state_size
        self.encoder = VoxelEncoder(input_size=ae_input_size,
                                    output_size=ae_output_size,
                                    in_channels=in_channels)
        self.pnet = PNet(input_size=ae_output_size + state_size * 2,
                         output_size=state_size + control_size)

    def forward(self, x, obs):
        if obs is not None:
            z = self.encoder(obs)
            z_x = torch.cat((z, x), 1)
        else:
            z_x = x
        return self.pnet(z_x)


# TODO: move loaders to a separate script
def load_voxels(data_dir, system_name):
    # return a numpy array stacked by voxels
    voxel_filename = os.path.join(data_dir, "voxel", system_name, "voxel.npy")

    if not os.path.exists(voxel_filename):
        # TODO: convert point cloud to voxel
        # TODO: save with hdf5 file format
        print("Cannot find {}. Converting from point clouds. ".format(voxel_filename))
        raise NotImplementedError

    return np.load(voxel_filename)


def load_trajs(data_dir, system_name):
    # return a HDF5 file object with the structure "env->traj".
    # Note: Remembder to close the HDF5 file after loading.
    traj_dir = os.path.join(data_dir, "traj", system_name)
    traj_filename = os.path.join(traj_dir, "traj.hdf5")

    if not os.path.exists(traj_filename):
        print("Cannot find {}. Converting from pickles. ".format(traj_filename))
        h5_file = h5py.File(traj_filename, 'a')
        for env_id in range(10):
            env_dir = os.path.join(traj_dir, "{}".format(env_id))
            h5_env_group = h5_file.create_group("{}".format(env_id))
            for traj_id in range(1000):
                traj_filename = os.path.join(env_dir, "path_{}.pkl".format(traj_id))
                with open(traj_filename, "rb") as f:
                    traj_data = pickle.load(f)
                h5_env_group.create_dataset("{}".format(traj_id), data=traj_data)
        h5_file.flush()
        h5_file.close()

    return h5py.File(traj_filename, 'r')


def convert_traj_to_mpnet_format(trajs):
    # iterate over all paths to determine datasize
    total_length = 0
    state_dim = np.prod(trajs["0"]["0"].shape[1:]).astype(np.int)
    for env_id in range(10):
        for traj_id in range(1000):
            total_length += trajs[str(env_id)][str(traj_id)].len() - 1

    # allocate memory and load
    all_input = np.zeros((total_length, 2 * state_dim))
    all_output = np.zeros((total_length, state_dim))
    all_voxel_id = np.zeros((total_length,), dtype=np.int)
    offset = 0
    for env_id in range(10):
        for traj_id in range(1000):
            traj = trajs[str(env_id)][str(traj_id)]
            N = traj.len() - 1
            all_input[offset:offset + N, :state_dim] = traj[:-1]
            all_input[offset:offset + N, state_dim:] = traj[-1]
            all_output[offset:offset + N] = traj[1:]
            all_voxel_id[offset:offset + N] = env_id
            offset += N

    return all_input, all_output, all_voxel_id


def load_data(data_dir, system_name):
    voxels = load_voxels(data_dir, system_name)

    # trajs = load_trajs(data_dir, system_name)
    # input_array, output_array, voxel_id_array = convert_traj_to_mpnet_format(trajs)
    # trajs.close()

    traj = np.load("./data/traj/car/car_obs_path_data.npy")
    voxel_id_array, input_array = np.split(traj, [1], axis=1)
    voxel_id_array = voxel_id_array.astype(np.int).flatten()
    output_array = np.load("./data/traj/car/car_obs_gt.npy")

    return input_array, output_array, voxel_id_array, voxels


class DataLoader:
    def __init__(self, input_array, output_array, voxel_id_array, voxel_array, *args, **kwargs):
        input_tensor = torch.from_numpy(input_array.astype(np.float32)).cuda()
        output_tensor = torch.from_numpy(output_array.astype(np.float32)).cuda()
        voxel_id_tensor = torch.from_numpy(voxel_id_array.astype(np.int)).cuda()
        dataset = torch.utils.data.TensorDataset(input_tensor, output_tensor, voxel_id_tensor)
        self.data_loader = torch.utils.data.DataLoader(dataset, *args, **kwargs)
        self.data_iterator = iter(self.data_loader)
        self.voxel_tensor = torch.from_numpy(voxel_array.astype(np.float32)).cuda()

    def __iter__(self, *args, **kwargs):
        self.data_iterator = iter(self.data_loader)
        return self

    def __next__(self, *args, **kwargs):
        input_sample, output_sample, voxel_id_sample = next(self.data_iterator)
        return input_sample, output_sample, self.voxel_tensor[voxel_id_sample.long()]

    def __len__(self, *args, **kwargs):
        return len(self.data_loader)


def main(data_dir="./data", system_name="car"):
    input_array, output_array, voxel_id_array, voxel_array = load_data(data_dir, system_name)

    # split into test loader and train loader
    N = input_array.shape[0]
    train_indices = np.random.choice(N, 9 * N // 10)
    train_mask = np.zeros((N,), dtype=np.bool)
    train_mask[train_indices] = True
    train_loader = DataLoader(input_array[train_mask],
                              output_array[train_mask],
                              voxel_id_array[train_mask],
                              voxel_array,
                              batch_size=128,
                              shuffle=True,
                              drop_last=True)
    test_mask = ~train_mask
    test_loader = DataLoader(input_array[test_mask],
                             output_array[test_mask],
                             voxel_id_array[test_mask],
                             voxel_array,
                             batch_size=128,
                             shuffle=True,
                             drop_last=True)

    # construct network
    mpnet = MPNet(ae_input_size=32, ae_output_size=64, in_channels=1, state_size=3, control_size=0).cuda()
    optimizer = torch.optim.Adam(mpnet.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    log_dir = os.path.join(data_dir, "log", "mpnet", system_name)
    utils.ensure_folder_exist(log_dir)
    model_script_filename = os.path.join(data_dir, "pytorch_model", "mpnet", system_name, "mpnet_script.pt")
    utils.ensure_folder_exist(model_script_filename)
    logger = SummaryWriter(log_dir)

    # main loop
    global_step = 0
    progress_bar = tqdm(total=1000, position=0)
    for i_epoch in range(1000):
        for input_tensor, output_tensor, voxel_tensor in tqdm(train_loader, "train", position=1, leave=False):
            optimizer.zero_grad()
            loss = F.mse_loss(mpnet.forward(input_tensor, voxel_tensor), output_tensor)
            loss.backward()
            optimizer.step()

            if global_step % 10 == 0:
                logger.add_scalar("training_loss", loss, global_step)

            global_step += 1
        scheduler.step()

        mpnet.eval()
        with torch.no_grad():
            count_temp = 0
            loss = torch.zeros(()).cuda()
            for input_tensor, output_tensor, voxel_tensor in tqdm(test_loader, "test", position=1, leave=False):
                loss += F.mse_loss(mpnet.forward(input_tensor, voxel_tensor), output_tensor)
                count_temp += 1
            logger.add_scalar("testing_loss", loss/count_temp, global_step)
        mpnet.train()

        mpnet.save(model_script_filename+".{}".format(i_epoch//10))
        progress_bar.set_postfix({'eval_loss': '{0:1.5f}'.format(loss.cpu().numpy()/count_temp)})
        progress_bar.update()


if __name__ == '__main__':
    main()
