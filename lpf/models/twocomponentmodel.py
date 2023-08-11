import json
from collections.abc import Sequence
from collections.abc import Mapping

import numpy as np
import PIL
from PIL import Image
 
from lpf.models import ReactionDiffusionModel
from lpf.initializers import Initializer
from lpf.solvers import Solver
from lpf.utils import get_template_fpath
from lpf.utils import get_mask_fpath


class TwoComponentModel(ReactionDiffusionModel):

    def __init__(self,
                 initializer=None,
                 n_init_pts=None,
                 params=None,
                 width=None,
                 height=None,
                 dx=None,
                 thr_color=None,
                 color_u=None,
                 color_v=None,
                 device=None,
                 ladybird=None,
                 dtype=None):

        # Set the device.
        super().__init__(device, dtype)

        # Set constant members.
        self._name = "TwoComponentModel"
        self._n_states = 2

        # Set initializer.
        self._initializer = initializer

        if n_init_pts:  # used for search
            self._n_init_pts = n_init_pts

        # Set kinetic parameters.
        if params is not None:
            with self.am:
                self._params = self.am.array(params, dtype=self._dtype)
                self._batch_size = self._params.shape[0]

        # Set the size of space (2D grid).
        if not width:
            width = 128

        if not height:
            height = 128

        self._width = width
        self._height = height
        self.shape = (height, width)

        if not dx:
            dx = 0.1

        self._dx = dx

        # Set the threshold and colors for coloring.
        if not thr_color:
            thr_color = 0.5

        self._thr_color = thr_color

        if not color_u:
            color_u = (5, 5, 5)

        if not color_v:
            color_v = (231, 79, 3) 

        self._color_u = np.array(color_u, dtype=np.uint8)
        self._color_v = np.array(color_v, dtype=np.uint8)

        # Set the template and mask for visualization.
        if ladybird is None:
            ladybird = "haxyridis"

        self._fpath_template = get_template_fpath(ladybird)
        self._fpath_mask = get_mask_fpath(ladybird)

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, arr):
        self._u = arr

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, arr):
        self._v = arr

    def initialize(self):
        with self.am:
            self._shape_grid = (self._n_states,
                                self._batch_size,
                                self._height,
                                self._width)

            self._y_mesh = self.am.zeros(self._shape_grid,
                                         dtype=self._dtype)

            self._u = self._y_mesh[0, :, :, :]
            self._v = self._y_mesh[1, :, :, :]

            # self._y_linear = self._y_mesh.ravel()

            self._dydt_mesh = self.am.zeros(self._shape_grid,
                                            dtype=self._dtype)
        # end of with

        self._initializer.initialize(self)

    def laplacian2d(self, a, dx):
        a_top = a[:, 0:-2, 1:-1]
        a_left = a[:, 1:-1, 0:-2]
        a_bottom = a[:, 2:, 1:-1]
        a_right = a[:, 1:-1, 2:]
        a_center = a[:, 1:-1, 1:-1]
        return (a_top + a_left + a_bottom + a_right - 4 * a_center) / dx ** 2

    def reactions(self, t, u_c, v_c):
        raise NotImplementedError()

    def pdefunc(self, t, y_mesh=None, y_linear=None):
        """Equation function for integration.
        """

        batch_size = self.params.shape[0]
            
        dydt_mesh = self._dydt_mesh

        u = y_mesh[0, :, :, :]
        v = y_mesh[1, :, :, :]

        # Model must update its states.
        self._u = u
        self._v = v
        dx = self._dx

        # Get the kinetic parameters.
        Du = self.params[:, 0].reshape(batch_size, 1, 1)
        Dv = self.params[:, 1].reshape(batch_size, 1, 1)

        u_c = u[:, 1:-1, 1:-1]
        v_c = v[:, 1:-1, 1:-1]

        # Reactions
        f, g = self.reactions(t, u_c, v_c)

        # Diffusions + Reactions
        # dydt_mesh[0, :, 1:-1, 1:-1] = Du * self.laplacian2d(u, dx) + f
        # dydt_mesh[1, :, 1:-1, 1:-1] = Dv * self.laplacian2d(v, dx) + g

        self.am.set(
            dydt_mesh,
            (0, slice(None), slice(1, -1), slice(1, -1)),
            Du * self.laplacian2d(u, dx) + f
        )
        
        self.am.set(
            dydt_mesh,
            (1, slice(None), slice(1, -1), slice(1, -1)),
            Dv * self.laplacian2d(v, dx) + g
        )

        # Neumann boundary condition: dydt = 0
        # dydt_mesh[:, :, 0, :] = 0.0
        # dydt_mesh[:, :, -1, :] = 0.0
        # dydt_mesh[:, :, :, 0] = 0.0
        # dydt_mesh[:, :, :, -1] = 0.0

        self.am.set(dydt_mesh, (slice(None), slice(None), 0, slice(None)),  0.0)
        self.am.set(dydt_mesh, (slice(None), slice(None), -1, slice(None)),  0.0)
        self.am.set(dydt_mesh, (slice(None), slice(None), slice(None), 0),  0.0)
        self.am.set(dydt_mesh, (slice(None), slice(None), slice(None), -1),  0.0)

        return self._dydt_mesh

    def is_numerically_invalid(self, index, arr_u=None, arr_v=None):
        if arr_u is None:
            arr_u = self.u

        if arr_v is None:
            arr_v = self.v

        with self.am:
            arr_u = arr_u[index, ...].astype(np.float16)
            arr_v = arr_v[index, ...].astype(np.float16)
            
            abs_u = self.am.abs(arr_u)
            abs_v = self.am.abs(arr_v)

            return (arr_u < 0).any() or (arr_v < 0).any() \
                   or self.am.isnan(abs_u.min()) or self.am.isnan(abs_v.min()) \
                   or self.am.isinf(abs_u.max()) or self.am.isinf(abs_v.max())

    def is_early_stopping(self, rtol):
                
        adu = self.am.abs(self._f)
        adv = self.am.abs(self._g)
        
        au = self.am.abs(self.u[:, 1:-1, 1:-1])
        av = self.am.abs(self.v[:, 1:-1, 1:-1])

        return (adu <= (rtol * au)).all() and (adv <= (rtol * av)).all()

    def colorize(self, thr_color=None):
        if not thr_color:
            thr_color = self._thr_color
            
        batch_size = self.u.shape[0]
        color = np.zeros((batch_size, self._height, self._width, 3),
                         dtype=np.uint8)

        color[:, :, :, 0] = self._color_v[0]
        color[:, :, :, 1] = self._color_v[1]
        color[:, :, :, 2] = self._color_v[2]
        
        idx = self.am.get(self.u) > thr_color
        color[idx, 0] = self._color_u[0]
        color[idx, 1] = self._color_u[1]
        color[idx, 2] = self._color_u[2]
        
        return color
    
    def create_image(self, index=0, arr_color=None, pattern=None):
        if arr_color is None and pattern is None:
            arr_color = self.colorize()

        # Load template images.
        template = Image.open(self._fpath_template)
        mask = Image.open(self._fpath_mask).convert('L')

        if pattern is None:
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
        img_right = img_left.transpose(PIL.Image.Transpose.FLIP_LEFT_RIGHT)
  
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
    
    def save_states(self, index=0, fpath=None):
        np.savez(fpath, u=self.u[index, ...], v=self.v[index, ...])
            
    def to_dict(self,
                index=0,
                initializer=None,
                params=None,
                solver=None,
                generation=None,
                fitness=None):

        n2v = {}
        
        n2v["model"] = self._name

        if index:
            n2v["index"] = index
            
        if generation:             
            n2v["generation"] = generation    
             
        if fitness:
            n2v["fitness"] = fitness
       
        # Save parameters for space and colorization.
        n2v["width"] = self._width
        n2v["height"] =self._height
        n2v["dx"] = self._dx
        n2v["thr_color"] = self._thr_color
        n2v["color_u"] = self._color_u.tolist()
        n2v["color_v"] = self._color_v.tolist()
       
        # Get the members of initializer: n_init_pts, init_pts, init_states
        if not initializer and self._initializer:
            initializer = self._initializer

        if initializer is None:
            pass
        elif isinstance(initializer, dict):
            n2v.update(initializer)
        elif isinstance(initializer, Initializer):
            n2v.update(initializer.to_dict(index))
        else:
             raise TypeError("initializer should be dict or a subclass of Initializer.")
       
        # Get the members of solver
        if not solver:
            n2v["solver"] = None
        elif isinstance(solver, dict):
            n2v.update(solver)
        elif isinstance(solver, Solver):
            n2v.update(solver.to_dict())
        
        # else:
        #     raise TypeError("solver should be dict or a subclass of Solver.")
             
        return n2v

    def save_model(self,
                   index=None,
                   fpath=None,
                   initializer=None,
                   params=None,
                   solver=None,
                   generation=None,
                   fitness=None):
        
        if not fpath:
            raise FileNotFoundError("Invalid file path: %s"%(fpath))

        if index is None:
            index = 0
        else:
            batch_size = self._batch_size
            if index < 0 or index >= batch_size:
                raise ValueError("index should be non-negative and less than the batch size.")

        if initializer is None:
            # if self.initializer is None:
            #     raise ValueError("initializer should be defined in model or given for save_model function.")

            initializer = self.initializer

        # if params is None:
        #     raise ValueError("params should be given.")

        with open(fpath, "wt") as fout:   
            model_dict = self.to_dict(index=index,
                                      initializer=initializer,
                                      params=params,
                                      solver=solver,
                                      generation=generation,
                                      fitness=fitness)

            json.dump(model_dict, fout)
    
        return model_dict

    @classmethod
    def parse_params(self, model_dicts):
        """Parse the parameters from the model dictionaries.
           A model knows how to parse its parameters.
        """
        
        if not isinstance(model_dicts, Sequence) and isinstance(model_dicts, Mapping):
            model_dicts = [model_dicts]
        elif isinstance(model_dicts, Sequence):
            pass
        else:
            raise TypeError("model_dicts should be a sequence of model dictionary or a mappable type like dict.")

        return model_dicts

    @classmethod
    def parse_init_conds(self, model_dicts):
        """Parse the initial conditions (initial points and states) from the model dictionaries.
           A model knows how to parse its initial points and states.
        """
        if not isinstance(model_dicts, Sequence) and isinstance(model_dicts, Mapping):
            model_dicts = [model_dicts]
        elif isinstance(model_dicts, Sequence):
            pass
        else:
            raise TypeError("model_dicts should be a sequence of model dictionary or a mappable type like dict.")

        # batch_size = len(model_dicts)
        # init_states = np.zeros((batch_size, 2), dtype=np.float64)
        #
        # for index, n2v in enumerate(model_dicts):
        #     init_states[index, 0] = n2v["u0"]
        #     init_states[index, 1] = n2v["v0"]
        # # end of for
        #
        # return init_states

        batch_size = len(model_dicts)
        init_states = np.zeros((batch_size, 2), dtype=self._dtype)
        init_pts = []

        for i, n2v in enumerate(model_dicts):
            init_states[i, 0] = n2v["u0"]
            init_states[i, 1] = n2v["v0"]

            n_init_pts = 0
            dict_init_pts = {}
            for name, val in n2v.items():
                if name.startswith("init_pts"):
                    # val = list(val)
                    dict_init_pts[name] = (int(val[0]), int(val[1]))
                    n_init_pts += 1
            # end of for

            coords = []
            for j, (name, coord) in enumerate(dict_init_pts.items()):
                coords.append((coord[0], coord[1]))
            # end of for
            init_pts.append(coords)
        # end of for

        init_pts = np.array(init_pts, dtype=np.uint32)

        return init_pts, init_states

    def get_param_bounds(self):
        raise NotImplementedError()

    def len_decision_vector(self):  # length of the decision vector in PyGMO
        raise NotImplementedError()

# end of class TwoComponentModel
