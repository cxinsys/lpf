import json

import numpy as np
import PIL
from PIL import Image
 
from lpf.models import ReactionDiffusionModel
from lpf.utils import get_template_fpath
from lpf.utils import get_mask_fpath


def laplacian2d(a, dx):
    a_top = a[:, 0:-2, 1:-1]
    a_left = a[:, 1:-1, 0:-2]
    a_bottom = a[:, 2:, 1:-1]
    a_right = a[:, 1:-1, 2:]
    a_center = a[:, 1:-1, 1:-1]
    return (a_top + a_left + a_bottom + a_right - 4*a_center) / dx**2


def pde_u(dt, dx, u, v, u_c, v_c, Du, ru, k, su, mu):
    return dt * (Du * laplacian2d(u, dx) \
                  + (ru*((u_c**2 * v_c)/(1 + k*u_c**2)) + su - mu*u_c))


def pde_v(dt, dx, u, v, u_c, v_c, Dv, rv, k, sv):
    return dt * (Dv * laplacian2d(v, dx) \
                 + (-rv*((u_c**2 * v_c)/(1 + k*u_c**2)) + sv))


class LiawModel(ReactionDiffusionModel):
    def __init__(self,
                 width,
                 height,
                 dx,
                 dt,
                 n_iters,
                 thr=0.5,
                 num_init_pts=25,
                 rtol_early_stop=None,
                 initializer=None,
                 device=None,
                 ladybird=None):

        self._name = "LiawModel"
        self._width = width
        self._height = height
        self.shape = (width, height)
        self._dx = dx
        self._dt = dt
        self._n_iters = n_iters
        self._thr = thr
        self._num_init_pts = num_init_pts
        self._rtol_early_stop = rtol_early_stop
        self._initializer = initializer

        if ladybird is None:
            ladybird = "haxyridis"

        self._fpath_template = get_template_fpath(ladybird)
        self._fpath_mask = get_mask_fpath(ladybird)

        super().__init__(device)
        
    def update(self, i, param_batch):

        with self.am:
            batch_size = param_batch.shape[0]

            dt = self.am.array(self._dt, dtype=np.float64)
            dx = self.am.array(self._dx, dtype=np.float64)

            u = self.u
            v = self.v

            Du = param_batch[:, 0].reshape(batch_size, 1, 1)
            Dv = param_batch[:, 1].reshape(batch_size, 1, 1)

            ru = param_batch[:, 2].reshape(batch_size, 1, 1)
            rv = param_batch[:, 3].reshape(batch_size, 1, 1)

            k = param_batch[:, 4].reshape(batch_size, 1, 1)

            su = param_batch[:, 5].reshape(batch_size, 1, 1)
            sv = param_batch[:, 6].reshape(batch_size, 1, 1)
            mu = param_batch[:, 7].reshape(batch_size, 1, 1)

            u_c = u[:, 1:-1, 1:-1]
            v_c = v[:, 1:-1, 1:-1]

            self._delta_u = pde_u(dt, dx, u, v, u_c, v_c, Du, ru, k, su, mu)
            self._delta_v = pde_v(dt, dx, u, v, u_c, v_c, Dv, rv, k, sv)

            # Boundary conditions
            # delta_u[0, :] = 0   # Top
            # delta_u[-1, :] = 0  # Bottom
            # delta_u[:, 0] = 0   # Left
            # delta_u[:, -1] = 0  # Right

            # delta_v[0, :] = 0   # Top
            # delta_v[-1, :] = 0  # Bottom
            # delta_v[:, 0] = 0   # Left
            # delta_v[:, -1] = 0  # Right

            u[:, 1:-1, 1:-1] = u_c + self._delta_u
            v[:, 1:-1, 1:-1] = v_c + self._delta_v

    def check_invalid_values(self):
        if self.am.any(self.am.isnan(self.u)) or self.am.any(self.am.isnan(self.v)):
            raise ValueError("Invalid value occurs!")

    def is_early_stopping(self, rtol):
                
        adu = self.am.abs(self._delta_u)
        adv = self.am.abs(self._delta_v)
        
        au = self.am.abs(self.u[:, 1:-1, 1:-1])
        av = self.am.abs(self.v[:, 1:-1, 1:-1])
        
        # max_rc = max((adu/au).max(), (adv/av).max())
        
        return (adu <= (rtol * au)).all() and (adv <= (rtol * av)).all()

    def colorize(self, thr=None):
        if not thr:
            thr = self._thr
            
        batch_size = self.u.shape[0]
        color = np.zeros((batch_size, self._height, self._width, 3),
                         dtype=np.uint8)

        color[:, :, :, 0] = 231
        color[:, :, :, 1] = 79
        color[:, :, :, 2] = 3
        
        idx = self.am.get(self.u) > thr  # self.u.get() > thr
        color[idx, 0] = 5  # self.u[idx]
        color[idx, 1] = 5  # self.u[idx]
        color[idx, 2] = 5  # self.u[idx]
        
        return color
    
    def create_image(self, i=0, arr_color=None):
        
        if arr_color is None:
            arr_color = self.colorize()

        # Load template images.
        template = Image.open(self._fpath_template)
        mask = Image.open(self._fpath_mask).convert('L')

        wings = Image.fromarray(arr_color[i, :, :])
        wings = wings.resize((128, 128))
        wings_crop = wings.crop((36, 12, 36 + 54, 12 + 104))
        img_wings = Image.new('RGBA', (template.width, template.height))
        img_wings.paste(wings_crop, (1, 20))
        
        img_canvas = Image.new('RGBA', (template.width, template.height), "WHITE")
        img_canvas.paste(template, mask=template)
        
        img_left = Image.composite(img_canvas, img_wings, mask)
        img_right = img_left.transpose(PIL.Image.FLIP_LEFT_RIGHT)
  
        arr_left = np.array(img_left)
        arr_right = np.array(img_right)

        arr_left = arr_left[:, :-4, :]
        arr_right = arr_right[:, 4:, :]

        arr_merged = np.hstack([arr_left, arr_right])
        img = Image.fromarray(arr_merged)

        return img

    def save_image(self, fpath, i=0, arr_color=None):
        img = self.create_image(i, arr_color)
        img.save(fpath)
        return img
    
    def save_states(self, fpath_states):
        raise NotImplementedError()

    def save_model(self,
                   fpath,
                   i=None,
                   init_states=None,
                   init_pts=None,
                   params=None,
                   generation=None,
                   fitness=None):

        if i is None:
            i = 0
        else:
            batch_size = params.shape[0]
            if i < 0 or i >= batch_size:
                raise ValueError("i should be non-negative and less than the batch size.")

        if init_states is None:
            raise ValueError("init_states should be given.")

        if params is None:
            raise ValueError("params should be given.")

        with open(fpath, "wt") as fout:   
            n2v = {}
           
            n2v["generation"] = generation
            n2v["fitness"] = fitness

            # Model parameters
            n2v["u0"] = init_states[i, 0]
            n2v["v0"] = init_states[i, 1]
            
            n2v["Du"] = params[i, 0]
            n2v["Dv"] = params[i, 1]
            n2v["ru"] = params[i, 2]
            n2v["rv"] = params[i, 3]
            n2v["k"]  = params[i, 4]
            n2v["su"] = params[i, 5]
            n2v["sv"] = params[i, 6]
            n2v["mu"] = params[i, 7]
            
            for i, (ir, ic) in enumerate(zip(*init_pts)):
                # Convert int to str due to JSON format.
                n2v["init_pts_%d"%(i)] = (str(ir), str(ic))
            # end of for
            
            # Hyper-parameters and etc.
            n2v["width"] = self._width
            n2v["height"] =self._height
            n2v["dt"] = self._dt
            n2v["dx"] = self._dx
            n2v["n_iters"] = self._n_iters
            n2v["thr"] = self._thr
            n2v["initializer"] = self._initializer.name
            
            json.dump(n2v, fout)
    
        return n2v

    def get_param_bounds(self):
        
        if not hasattr(self, "bounds_min"):
            self.bounds_min = self.am.zeros((10 + 2 * self._num_init_pts),
                                            dtype=np.float64)
            
        if not hasattr(self, "bounds_max"):
            self.bounds_max = self.am.zeros((10 + 2 * self._num_init_pts),
                                            dtype=np.float64)
        
        # Du
        self.bounds_min[0] = -4
        self.bounds_max[0] = 0
        
        # Dv
        self.bounds_min[1] = -4
        self.bounds_max[1] = 0
        
        # ru
        self.bounds_min[2] = -2
        self.bounds_max[2] = 2
        
        # rv
        self.bounds_min[3] = -2
        self.bounds_max[3] = 2        
        
        # k
        self.bounds_min[4] = -4
        self.bounds_max[4] = 0
        
        # su
        self.bounds_min[5] = -4
        self.bounds_max[5] = 0
        
        # sv
        self.bounds_min[6] = -4
        self.bounds_max[6] = 0
        
        # mu
        self.bounds_min[7] = -3
        self.bounds_max[7] = -1
        
        # u0
        self.bounds_min[8] = 0
        self.bounds_max[8] = 1.5

        # v0
        self.bounds_min[9] = 0
        self.bounds_max[9] = 1.5
        
        # init coords (25 points).     
        for i in range(10, 2*self._num_init_pts, 2):
            self.bounds_min[i] = 0
            self.bounds_max[i] = self._height - 1
        # end of for

        for i in range(11, 2*self._num_init_pts, 2):
            self.bounds_min[i] = 0
            self.bounds_max[i] = self._width - 1
        # end of for
        
        return self.bounds_min, self.bounds_max

# end of class
