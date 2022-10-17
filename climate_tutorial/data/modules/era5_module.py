import os
import glob
import torch
import numpy as np
import xarray as xr

from tqdm.notebook import tqdm
from torch.utils.data import Dataset
from torchvision.transforms import transforms

NAME_TO_VAR = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "msl",
    "surface_pressure": "sp",
    "toa_incident_solar_radiation": "tisr",
    "total_precipitation": "tp",
    "land_sea_mask": "lsm",
    "orography": "orography",
    "lattitude": "lat2d",
    "geopotential": "z",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "temperature": "t",
    "relative_humidity": "r",
    "specific_humidity": "q",
}

VAR_TO_NAME = {v: k for k, v in NAME_TO_VAR.items()}

class ERA5(Dataset):
    def __init__(self, root_dir, root_highres_dir, variables, years, split = 'train'):
        super().__init__()
        self.root_dir = root_dir
        self.root_highres_dir = root_highres_dir
        self.variables = variables
        self.years = years
        self.split = split

        self.data_dict = self.load_from_nc(self.root_dir)
        if self.root_highres_dir is not None:
            self.data_highres_dict = self.load_from_nc(self.root_highres_dir)

        self.get_lat_lon()

    def load_from_nc(self, data_dir):
        data_dict = {k: [] for k in self.variables}

        for year in tqdm(self.years):
            for var in self.variables:
                dir_var = os.path.join(data_dir, var)
                ps = glob.glob(os.path.join(dir_var, f'*{year}*.nc'))
                xr_data = xr.open_mfdataset(ps, combine='by_coords')
                lc = xr_data.coords["lon"]
                la = xr_data.coords["lat"]
                xr_data = xr_data.loc[
                    dict(lon=lc[(lc > 190) & (lc < 310)], lat=la[(la > 10) & (la < 75)])
                ]
                xr_data = xr_data[NAME_TO_VAR[var]]
                # np_data = xr_data.to_numpy()
                if len(xr_data.shape) == 3: # 8760, 32, 64
                    xr_data = xr_data.expand_dims(dim='level', axis=1)
                data_dict[var].append(xr_data)
        
        data_dict = {k: xr.concat(data_dict[k], dim='time') for k in self.variables}
        
        return data_dict

    def get_lat_lon(self):
        # lat lon is stored in each of the nc files, just need to load one and extract
        dir_var = os.path.join(self.root_dir, self.variables[0])
        year = self.years[0]
        ps = glob.glob(os.path.join(dir_var, f'*{year}*.nc'))
        xr_data = xr.open_mfdataset(ps, combine='by_coords')
        self.lat = xr_data['lat'].to_numpy()
        self.lon = xr_data['lon'].to_numpy()

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

class ERA5Forecasting(ERA5):
    def __init__(self, root_dir, root_highres_dir, in_vars, out_vars, pred_range, years, subsample=1, split='train'):
        print (f'Creating {split} dataset')
        super().__init__(root_dir, root_highres_dir, in_vars, years, split)
        
        self.in_vars = in_vars
        self.out_vars = out_vars
        self.pred_range = pred_range

        inp_data = xr.concat([self.data_dict[k] for k in in_vars], dim='level')
        out_data = xr.concat([self.data_dict[k] for k in out_vars], dim='level')

        self.inp_data = inp_data[0:-pred_range:subsample].to_numpy().astype(np.float32)
        self.out_data = out_data[pred_range::subsample].to_numpy().astype(np.float32)

        assert len(self.inp_data) == len(self.out_data)

        self.downscale_ratio = 1

        if split == 'train':
            self.inp_transform = self.get_normalize(self.inp_data)
            self.out_transform = self.get_normalize(self.out_data)
        else:
            self.inp_transform = None
            self.out_transform = None

        self.time = self.data_dict[in_vars[0]].time.to_numpy()[:-pred_range:subsample].copy()
        self.inp_lon = self.data_dict[in_vars[0]].lon.to_numpy().copy()
        self.inp_lat = self.data_dict[in_vars[0]].lat.to_numpy().copy()
        self.out_lon = self.data_dict[out_vars[0]].lon.to_numpy().copy()
        self.out_lat = self.data_dict[out_vars[0]].lat.to_numpy().copy()

        del self.data_dict

    def get_normalize(self, data):
        mean = np.mean(data, axis=(0, 2, 3))
        std = np.std(data, axis=(0, 2, 3))
        return transforms.Normalize(mean, std)

    def set_normalize(self, inp_normalize, out_normalize): # for val and test
        self.inp_transform = inp_normalize
        self.out_transform = out_normalize

    def get_climatology(self):
        return torch.from_numpy(self.out_data.mean(axis=0))

    def __getitem__(self, index):
        inp = torch.from_numpy(self.inp_data[index])
        out = torch.from_numpy(self.out_data[index])
        return self.inp_transform(inp), self.out_transform(out), self.in_vars, self.out_vars

    def __len__(self):
        return len(self.inp_data)

class ERA5ForecastingCustom(ERA5):
    def __init__(self, root_dir, root_highres_dir, in_vars, out_vars, pred_range, years, subsample=1, split='train'):
        print (f'Creating {split} dataset')
        super().__init__(root_dir, root_highres_dir, in_vars, years, split)
        
        self.in_vars = in_vars
        self.out_vars = out_vars
        self.pred_range = pred_range

        inp_data = xr.concat([self.data_dict[k] for k in in_vars], dim='level')
        out_data = xr.concat([self.data_dict[k] for k in out_vars], dim='level')

        self.inp_data = inp_data[0:-pred_range:subsample].to_numpy().astype(np.float32)
        self.out_data = out_data[pred_range::subsample].to_numpy().astype(np.float32)

        assert len(self.inp_data) == len(self.out_data)

        self.downscale_ratio = 1

        if split == 'train':
            self.inp_transform = self.get_normalize(self.inp_data)
            self.out_transform = self.get_normalize(self.out_data)
        else:
            self.inp_transform = None
            self.out_transform = None

        self.time = self.data_dict[in_vars[0]].time.to_numpy()[:-pred_range:subsample].copy()
        self.inp_lon = self.data_dict[in_vars[0]].lon.to_numpy().copy()
        self.inp_lat = self.data_dict[in_vars[0]].lat.to_numpy().copy()
        self.out_lon = self.data_dict[out_vars[0]].lon.to_numpy().copy()
        self.out_lat = self.data_dict[out_vars[0]].lat.to_numpy().copy()

        del self.data_dict

    def get_normalize(self, data):
        mean = np.mean(data, axis=(0, 2, 3))
        std = np.std(data, axis=(0, 2, 3))
        return transforms.Normalize(mean, std)

    def set_normalize(self, inp_normalize, out_normalize): # for val and test
        self.inp_transform = inp_normalize
        self.out_transform = out_normalize

    def get_climatology(self):
        return torch.from_numpy(self.out_data.mean(axis=0))

    def __getitem__(self, index):
        inp = torch.from_numpy(self.inp_data[index])
        out = torch.from_numpy(self.out_data[index])
        return self.inp_transform(inp), self.out_transform(out), self.in_vars, self.out_vars

    def __len__(self):
        return len(self.inp_data)

class ERA5Downscaling(ERA5):
    def __init__(self, root_dir, root_highres_dir, in_vars, out_vars, pred_range, years, subsample=1, split='train'):
        print (f'Creating {split} dataset')
        super().__init__(root_dir, root_highres_dir, in_vars, years, split)
        
        self.in_vars = in_vars
        self.out_vars = out_vars
        self.pred_range = pred_range

        inp_data = xr.concat([self.data_dict[k] for k in in_vars], dim='level')
        out_data = xr.concat([self.data_highres_dict[k] for k in out_vars], dim='level')

        self.inp_data = inp_data[::subsample].to_numpy().astype(np.float32)
        self.out_data = out_data[::subsample].to_numpy().astype(np.float32)

        assert len(self.inp_data) == len(self.out_data)

        self.downscale_ratio = self.out_data.shape[-1] // self.inp_data.shape[-1]

        if split == 'train':
            self.inp_transform = self.get_normalize(self.inp_data)
            self.out_transform = self.get_normalize(self.out_data)
        else:
            self.inp_transform = None
            self.out_transform = None

        self.time = self.data_dict[in_vars[0]].time.to_numpy()[::subsample].copy()
        self.inp_lon = self.data_dict[in_vars[0]].lon.to_numpy().copy()
        self.inp_lat = self.data_dict[in_vars[0]].lat.to_numpy().copy()
        self.out_lon = self.data_highres_dict[out_vars[0]].lon.to_numpy().copy()
        self.out_lat = self.data_highres_dict[out_vars[0]].lat.to_numpy().copy()

        del self.data_dict
        del self.data_highres_dict

    def get_normalize(self, data):
        mean = np.mean(data, axis=(0, 2, 3))
        std = np.std(data, axis=(0, 2, 3))
        return transforms.Normalize(mean, std)

    def set_normalize(self, inp_normalize, out_normalize): # for val and test
        self.inp_transform = inp_normalize
        self.out_transform = out_normalize

    def get_climatology(self):
        return torch.from_numpy(self.out_data.mean(axis=0))

    def __getitem__(self, index):
        inp = torch.from_numpy(self.inp_data[index])
        out = torch.from_numpy(self.out_data[index])
        return self.inp_transform(inp), self.out_transform(out), self.in_vars, self.out_vars

    def __len__(self):
        return len(self.inp_data)