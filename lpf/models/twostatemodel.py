import json
from collections.abc import Sequence

import numpy as np
import PIL
from PIL import Image
 
from lpf.models import ReactionDiffusionModel
from lpf.initializers import Initializer
from lpf.solvers import Solver
from lpf.utils import get_template_fpath
from lpf.utils import get_mask_fpath


class TwoStateModel(ReactionDiffusionModel):

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
                 ladybird=None):

        # Set the device.
        super().__init__(device)

        # Set constant members.
        self._name = "TwoStateModel"
        self._n_states = 2

        # Set initializer.
        self._initializer = initializer

        if n_init_pts:  # used for search
            self._n_init_pts = n_init_pts

        # Set kinetic parameters.
        if params is not None:
            with self.am:
                self._params = self.am.array(params, dtype=params.dtype)                
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

    @property
    def v(self):
        return self._v

    def laplacian2d(self, a, dx):
        a_top = a[:, 0:-2, 1:-1]
        a_left = a[:, 1:-1, 0:-2]
        a_bottom = a[:, 2:, 1:-1]
        a_right = a[:, 1:-1, 2:]
        a_center = a[:, 1:-1, 1:-1]
        return (a_top + a_left + a_bottom + a_right - 4 * a_center) / dx ** 2

    def reactions(self, t, u_c, v_c):
        raise NotImplementedError()

    def pdefunc(self, t, y_linear):
        """Equation function for integration.
        """

        batch_size = self.params.shape[0]

        y_mesh = y_linear.reshape(self.n_states, batch_size, self.height, self.width)
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
        dydt_mesh[0, :, 1:-1, 1:-1] = Du * self.laplacian2d(u, dx) + f
        dydt_mesh[1, :, 1:-1, 1:-1] = Dv * self.laplacian2d(v, dx) + g

        # Neumann boundary condition: dydt = 0
        dydt_mesh[:, :, 0, :] = 0.0
        dydt_mesh[:, :, -1, :] = 0.0
        dydt_mesh[:, :, :, 0] = 0.0
        dydt_mesh[:, :, :, -1] = 0.0

        return self._dydt_linear  # It is the same as dydt.ravel()


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
        with open(fpath, "wt") as fout:
            self.u
            self.v
            
            
    def to_dict(self,
                index=None,
                initializer=None,
                params=None,
                solver=None,
                generation=None,
                fitness=None):
        
        n2v = {}
        
        n2v["model_name"] = self._name

        
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
        if isinstance(initializer, dict):
            n2v.update(initializer)
        elif isinstance(initializer, Initializer):
            n2v.update(initializer.to_dict(index))
        # else:
        #     raise TypeError("initializer should be dict or a subclass of Initializer.")
       
        # Get the members of solver
        if isinstance(solver, dict):
            n2v.update(solver)
        elif isinstance(solver, Solver):
            n2v.update(solver.to_dict())
        else:
            raise TypeError("solver should be dict or a subclass of Solver.")
             
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


    @staticmethod
    def parse_params(model_dicts):
        raise NotImplementedError()


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
        raise NotImplementedError()


    def len_decision_vector(self):  # length of the decision vector in PyGMO
        raise NotImplementedError()

# end of class TwoStateModel
