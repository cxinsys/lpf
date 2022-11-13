import json
from collections.abc import Sequence

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
    return dt * (Du * laplacian2d(u, dx) + (ru*((u_c**2 * v_c)/(1 + k*u_c**2)) + su - mu*u_c))
    # return dt * (Du * laplacian2d(u, dx) + (ru * ((u_c*u_c * v_c) / (1 + k * u_c*u_c)) + su - mu * u_c))


def pde_v(dt, dx, u, v, u_c, v_c, Dv, rv, k, sv):
    return dt * (Dv * laplacian2d(v, dx) + (-rv*((u_c**2 * v_c)/(1 + k*u_c**2)) + sv))
    # return dt * (Dv * laplacian2d(v, dx) + (-rv*((u_c*u_c * v_c)/(1 + k * u_c*u_c)) + sv))


class LiawModel(ReactionDiffusionModel):
    def __init__(self,
                 width,
                 height,
                 dx,
                 dt,
                 n_iters,
                 thr=0.5,
                 n_init_pts=25,
                 rtol_early_stop=None,
                 color_u=None,
                 color_v=None,
                 initializer=None,
                 device=None,
                 ladybird=None,
                 ):

        self._name = "LiawModel"
        self._width = width
        self._height = height
        self.shape = (width, height)
        self._dx = dx
        self._dt = dt
        self._n_iters = n_iters
        self._thr = thr
        self._n_init_pts = n_init_pts
        self._rtol_early_stop = rtol_early_stop
        self._initializer = initializer

        if not color_u:
            color_u =  np.array([5, 5, 5], dtype=np.uint8)

        if not color_v:
            color_v = np.array([231, 79, 3], dtype=np.uint8)

        self._color_u = color_u
        self._color_v = color_v

        if ladybird is None:
            ladybird = "haxyridis"

        self._fpath_template = get_template_fpath(ladybird)
        self._fpath_mask = get_mask_fpath(ladybird)

        super().__init__(device)
        
    def update(self, params):

        with self.am:
            batch_size = params.shape[0]

            dt = self._dt
            dx = self._dx

            u = self.u
            v = self.v

            Du = params[:, 0].reshape(batch_size, 1, 1)
            Dv = params[:, 1].reshape(batch_size, 1, 1)

            ru = params[:, 2].reshape(batch_size, 1, 1)
            rv = params[:, 3].reshape(batch_size, 1, 1)

            k = params[:, 4].reshape(batch_size, 1, 1)

            su = params[:, 5].reshape(batch_size, 1, 1)
            sv = params[:, 6].reshape(batch_size, 1, 1)
            mu = params[:, 7].reshape(batch_size, 1, 1)

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

        color[:, :, :, 0] = self._color_v[0]
        color[:, :, :, 1] = self._color_v[1]
        color[:, :, :, 2] = self._color_v[2]
        
        idx = self.am.get(self.u) > thr
        color[idx, 0] = self._color_u[0]
        color[idx, 1] = self._color_u[1]
        color[idx, 2] = self._color_u[2]
        
        return color
    
    def create_image(self, index=0, arr_color=None):
        if arr_color is None:
            arr_color = self.colorize()

        # Load template images.
        template = Image.open(self._fpath_template)
        mask = Image.open(self._fpath_mask).convert('L')

        pattern = Image.fromarray(arr_color[index, :, :])
        pattern = pattern.resize((128, 128))

        # crop(left, upper, right, lower)
        pattern_crop = pattern.crop((36, 12, 36 + 54, 12 + 104))
        img_wing = Image.new('RGBA', (template.width, template.height))
        img_wing.paste(pattern_crop, (1, 20))
        
        img_canvas = Image.new('RGBA', (template.width, template.height), "WHITE")
        img_canvas.paste(template, mask=template)
        

        """
        <Understanding the compoiste function>
        
        Image.paste(im, box=None, mask=None)
            - Where the mask is 255, the given image is copied as is.
            - Where the mask is 0, the current value is preserved.
            - Intermediate values will mix the two images together,
              including their alpha channels if they have them.
            - [REF] https://pillow.readthedocs.io/en/stable/reference/Image.html

        The following is the implementation of compoiste function.

        def composite(image1, image2, mask):
            image = image2.copy()
            image.paste(image1, None, mask)  # without the box
            return image


        The following code basically pastes the img_template to the img_wing with the mask.
        """
        img_left = Image.composite(img_canvas, img_wing, mask)
        img_right = img_left.transpose(PIL.Image.FLIP_LEFT_RIGHT)
  
        arr_left = np.array(img_left)
        arr_right = np.array(img_right)

        arr_left = arr_left[:, :-4, :]
        arr_right = arr_right[:, 4:, :]

        arr_merged = np.hstack([arr_left, arr_right])
        ladybird = Image.fromarray(arr_merged)

        return ladybird, pattern

    def save_image(self,
                   index=0,
                   fpath_ladybird=None,
                   fpath_pattern=None,
                   arr_color=None):
        ladybird, pattern = self.create_image(index, arr_color)
        ladybird.save(fpath_ladybird)
        if fpath_pattern:
            pattern.save(fpath_pattern)
    
    def save_states(self, index=0, fpath=None, states=None):
        raise NotImplementedError()

    def save_model(self,
                   index=None,
                   fpath=None,
                   init_states=None,
                   init_pts=None,
                   params=None,
                   generation=None,
                   fitness=None):
        
        if not fpath:
            raise FileNotFoundError("Invalid file path: %s"%(fpath))

        if index is None:
            index = 0
        else:
            batch_size = params.shape[0]
            if index < 0 or index >= batch_size:
                raise ValueError("index should be non-negative and less than the batch size.")

        if init_states is None:
            raise ValueError("init_states should be given.")
            
        if init_pts is None:
            raise ValueError("init_pts should be given.")

        if params is None:
            raise ValueError("params should be given.")

        with open(fpath, "wt") as fout:   
            n2v = {}

            n2v["index"] = index
           
            n2v["generation"] = generation
            n2v["fitness"] = fitness

            # Save kinetic parameters
            n2v["u0"] = float(init_states[index, 0])
            n2v["v0"] = float(init_states[index, 1])
            
            n2v["Du"] = float(params[index, 0])
            n2v["Dv"] = float(params[index, 1])
            n2v["ru"] = float(params[index, 2])
            n2v["rv"] = float(params[index, 3])
            n2v["k"]  = float(params[index, 4])
            n2v["su"] = float(params[index, 5])
            n2v["sv"] = float(params[index, 6])
            n2v["mu"] = float(params[index, 7])

            # Save init points
            n2v["n_init_pts"] = self._n_init_pts

            for i, (ir, ic) in enumerate(init_pts[index, :]):
                # Convert int to str due to JSON format.
                n2v["init_pts_%d"%(i)] = [int(ir), int(ic)]
            # end of for
            
            # Save hyper-parameters and etc.
            n2v["width"] = self._width
            n2v["height"] =self._height
            n2v["dt"] = self._dt
            n2v["dx"] = self._dx
            n2v["n_iters"] = self._n_iters
            n2v["thr"] = self._thr
            n2v["initializer"] = self._initializer.name if self._initializer else None

            n2v["color_u"] = self._color_u.tolist()
            n2v["color_v"] = self._color_v.tolist()

            json.dump(n2v, fout)
    
        return n2v


    @staticmethod
    def parse_params(model_dicts):
        """Parse the parameters from the model dictionaries.
           A model knows how to parse its parameters.
        """
        if not isinstance(model_dicts, Sequence):
            raise TypeError("model_dicts should be a sequence of model dictionary.")

        batch_size = len(model_dicts)
        params = np.zeros((batch_size, 8), dtype=np.float64)

        for index, n2v in enumerate(model_dicts):
            params[index, 0] = n2v["Du"]
            params[index, 1] = n2v["Dv"]
            params[index, 2] = n2v["ru"]
            params[index, 3] = n2v["rv"]
            params[index, 4] = n2v["k"]
            params[index, 5] = n2v["su"]
            params[index, 6] = n2v["sv"]
            params[index, 7] = n2v["mu"]

        return params

    @staticmethod
    def parse_init_states(self, model_dicts):
        """Parse the initial states from the model dictionaries.
           A model knows how to parse its initial states.
        """
        if not isinstance(model_dicts, Sequence):
            raise TypeError("model_dicts should be a sequence of model dictionary.")

        batch_size = len(model_dicts)
        init_states = np.zeros((batch_size, 2), dtype=np.float64)

        for index, n2v in enumerate(model_dicts):
            init_states[index, 0] = n2v["u0"]
            init_states[index, 1] = n2v["v0"]
        # end of for

        return init_states

    def get_param_bounds(self):
        
        if not hasattr(self, "bounds_min"):
            self.bounds_min = self.am.zeros((10 + 2 * self._n_init_pts),
                                            dtype=np.float64)
            
        if not hasattr(self, "bounds_max"):
            self.bounds_max = self.am.zeros((10 + 2 * self._n_init_pts),
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
        for index in range(10, 2 * self._n_init_pts, 2):
            self.bounds_min[index] = 0
            self.bounds_max[index] = self._height - 1
        # end of for

        for index in range(11, 2 * self._n_init_pts, 2):
            self.bounds_min[index] = 0
            self.bounds_max[index] = self._width - 1
        # end of for
        
        return self.bounds_min, self.bounds_max

    def get_len_dv(self):  # length of the decision vector in PyGMO
        return 10 + 2 * self._n_init_pts




# end of class
