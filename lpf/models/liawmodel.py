import json

import numpy as np
# from numba import jit, njit
import PIL
from PIL import Image
 
from lpf.models import ReactionDiffusionModel


# def laplacian2d(a, dx):
#     return (
#         - 4 * a
#         + np.roll(a, 1, 0) 
#         + np.roll(a, -1, 0)
#         + np.roll(a, +1, 1)
#         + np.roll(a, -1, 1)
#     ) / (dx ** 2)


# @njit(fastmath=True, cache=True, nogil=True, parallel=True)
def laplacian2d(a, dx):
    a_top = a[:, 0:-2, 1:-1]
    a_left = a[:, 1:-1, 0:-2]
    a_bottom = a[:, 2:, 1:-1]
    a_right = a[:, 1:-1, 2:]
    a_center = a[:, 1:-1, 1:-1]
    return (a_top + a_left + a_bottom + a_right - 4*a_center) / dx**2

# @njit(fastmath=True, cache=True, nogil=True, parallel=True)
def pde_u(dt, dx, u, v, u_c, v_c, Du, ru, k, su, mu):    
    return dt * (Du * laplacian2d(u, dx) \
                  + (ru*((u_c**2 * v_c)/(1 + k*u_c**2)) + su - mu*u_c))
        
# @njit(fastmath=True, cache=True, nogil=True, parallel=True)
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
                 fpath_template=None,
                 fpath_mask=None):
    
        self.width = width
        self.height = height
        self.shape = (width, height)
        self.dx = dx
        self.dt = dt
        self.n_iters = n_iters
        self.thr = thr
        self.num_init_pts = num_init_pts
        self.rtol_early_stop = rtol_early_stop
        self.initializer = initializer
        self.fpath_template = fpath_template
        self.fpath_mask = fpath_mask
        
        
    # @jit(fastmath=True)
    def update(self, i, param_batch):
        
        batch_size = param_batch.shape[0]
        
        dt = self.dt #.reshape(batch_size, 1, 1)
        dx = self.dx #.reshase(batch_size, 1, 1)
        
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

        self.delta_u = pde_u(dt, dx, u, v, u_c, v_c, Du, ru, k, su, mu)
        self.delta_v = pde_v(dt, dx, u, v, u_c, v_c, Dv, rv, k, sv)

        # Boundary conditions
        # delta_u[0, :] = 0   # Top
        # delta_u[-1, :] = 0  # Bottom
        # delta_u[:, 0] = 0   # Left
        # delta_u[:, -1] = 0  # Right
        
        # delta_v[0, :] = 0   # Top
        # delta_v[-1, :] = 0  # Bottom
        # delta_v[:, 0] = 0   # Left
        # delta_v[:, -1] = 0  # Right

        u[:, 1:-1, 1:-1] = u_c + self.delta_u
        v[:, 1:-1, 1:-1] = v_c + self.delta_v

    def is_early_stopping(self, rtol):       
                
        adu = np.abs(self.delta_u)
        adv = np.abs(self.delta_v)
        
        au = np.abs(self.u[:, 1:-1, 1:-1])
        av = np.abs(self.v[:, 1:-1, 1:-1])
        
        # max_rc = max((adu/au).max(), (adv/av).max())
        
        return (adu <= (rtol * au)).all() and (adv <= (rtol * av)).all()

    def colorize(self, thr=None):
        if not thr:
            thr = self.thr
            
        batch_size = self.u.shape[0]
        color = np.zeros((batch_size, self.height, self.width, 3),
                         dtype=np.uint8)
        color[:, :, :, 0] = 231
        color[:, :, :, 1] = 79
        color[:, :, :, 2] = 3
        
        idx = self.u > thr
        color[idx, 0] = 5 # self.u[idx]
        color[idx, 1] = 5 # self.u[idx]
        color[idx, 2] = 5 # self.u[idx]
        
        return color
    
    def create_image(self, 
                     i=0,
                     arr_color=None,
                     fpath_template=None,
                     fpath_mask=None):        
        
        if arr_color is None:
            arr_color = self.colorize()
            
        if not fpath_template:
            fpath_template = self.fpath_template
            
        if not fpath_mask:
            fpath_mask = self.fpath_mask

        # Load template images.
        template = Image.open(fpath_template)
        #template = template.resize(shape)
        
        mask = Image.open(fpath_mask).convert('L')
        #mask = mask.resize(shape).convert('L')
        
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

    def save_image(self,
                   fpath_image,
                   i=0,
                   arr_color=None,
                   fpath_template=None,
                   fpath_mask=None):                

        img = self.create_image(i,
                                arr_color,
                                fpath_template,
                                fpath_mask)
        

        img.save(fpath_image)
            
        return img
    
    
    def save_states(self, fpath_states):
        raise NotImplementedError()
    
    
    def save_model(self,
                   fpath,
                   init_states,
                   init_pts,
                   params,
                   generation=None,
                   fitness=None):

        if params.ndim > 1:
            params = params[0, :]

        with open(fpath, "wt") as fout:   
            n2v = {}
           
            n2v["generation"] = generation
            n2v["fitness"] = fitness

            # Model parameters
            n2v["u0"] = init_states[0]
            n2v["v0"] = init_states[1]
            
            n2v["Du"] = params[0]
            n2v["Dv"] = params[1]
            n2v["ru"] = params[2]
            n2v["rv"] = params[3]
            n2v["k"]  = params[4]
            n2v["su"] = params[5]
            n2v["sv"] = params[6]
            n2v["mu"] = params[7]           
            
            
            for i, (ir, ic) in enumerate(zip(*init_pts)):
                n2v["init_pts_%d"%(i)] = (str(ir), str(ic))
            
            # Hyper-parameters and etc.
            n2v["width"] = self.width
            n2v["height"] =self.height
            n2v["dt"] = self.dt
            n2v["dx"] = self.dx  
            n2v["n_iters"] = self.n_iters
            n2v["thr"] = self.thr
            n2v["initializer"] = self.initializer.__class__.__name__
            
            json.dump(n2v, fout)
    
        return n2v

    def get_param_bounds(self):
        
        if not hasattr(self, "bounds_min"):
            self.bounds_min = np.zeros((10 + 2*self.num_init_pts),
                                       dtype=np.float64)
            
        if not hasattr(self, "bounds_max"):
            self.bounds_max = np.zeros((10 + 2*self.num_init_pts),
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
        for i in range(10, 2*self.num_init_pts, 2):
            self.bounds_min[i] = 0
            self.bounds_max[i] = self.height - 1
            
        for i in range(11, 2*self.num_init_pts, 2):
            self.bounds_min[i] = 0
            self.bounds_max[i] = self.width - 1
        
        
        return self.bounds_min, self.bounds_max

# end of class
