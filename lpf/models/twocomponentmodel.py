import json
import os.path as osp
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
from lpf.conditioners import Dirichlet2C
from lpf.conditioners import Neumann2C
from lpf.geometry import make_geometry
from lpf.geometry import geometry_kwargs
from lpf.geometry.grid import grid_laplacian


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
                 species=None,
                 dtype=None,
                 domain=None,
                 mesh=None):

        # Set the device.
        super().__init__(device, dtype)

        # Set constant members.
        self._name = "TwoComponentModel"
        self._n_states = 2

        # Set initializer.
        self._initializer = initializer

        if n_init_pts is not None:  # used for search
            self._n_init_pts = n_init_pts

        # Set kinetic parameters.
        if params is not None:
            with self.am:
                self._params = self.am.array(params,
                                             dtype=self._dtype)
                self._batch_size = self._params.shape[0]

        # Set the size of space (2D grid).
        if width is None:
            width = 128

        if height is None:
            height = 128

        self._width = width
        self._height = height
        self.shape = (height, width)

        if dx is None:
            dx = 0.1

        self._dx = dx

        # Set the threshold and colors for coloring.
        if thr_color is None:  # [!TODO] when params and batch_size is None 
            # Handle case where _batch_size might not be set yet (e.g., in diploid models)
            if hasattr(self, '_batch_size') and self._batch_size is not None:
                thr_color = 0.5 * np.ones((self._batch_size, 1, 1))
            else:
                # Default to batch size of 1 if not set
                thr_color = 0.5 * np.ones((1, 1, 1))

        self._thr_color = thr_color

        if color_u is None:
            color_u = (5, 5, 5)

        if color_v is None:
            color_v = (231, 79, 3)

        self._color_u = np.array(color_u, dtype=np.uint8)
        self._color_v = np.array(color_v, dtype=np.uint8)

        # Set the template and mask for visualization.
        if species is None:
            species = "haxyridis"

        self._fpath_template = get_template_fpath(species)
        self._fpath_mask = get_mask_fpath(species)

        # Space: the full grid (default), a domain on it (domain=, an
        # lpf.domains.Domain or one per batch element) or the vertices of a
        # mesh (mesh=, an lpf.surfaces.SurfaceMesh; width, height and dx are
        # then not used). See lpf.geometry.
        self._geometry = make_geometry(height, width, dx,
                                       domain=domain, mesh=mesh)
        self._color_bg = np.array((255, 255, 255), dtype=np.uint8)

    @property
    def u(self):
        if self._y_mesh is not None:
            return self._y_mesh[0]
        return self._u

    @u.setter
    def u(self, obj):
        with self.am:
            if hasattr(self, '_y_mesh') and self._y_mesh is not None:
                self._y_mesh = self.am.set(
                    self._y_mesh, (0, ...),
                    self.am.array(obj, dtype=self._dtype))
            else:
                self._u = obj

    @property
    def v(self):
        if self._y_mesh is not None:
            return self._y_mesh[1]
        return self._v

    @v.setter
    def v(self, obj):
        with self.am:
            if hasattr(self, '_y_mesh') and self._y_mesh is not None:
                self._y_mesh = self.am.set(
                    self._y_mesh, (1, ...),
                    self.am.array(obj, dtype=self._dtype))
            else:
                self._v = obj
                
    def initialize(self):
        with self.am:
            self._shape_grid = (self._n_states,
                                self._batch_size,
                                *self.spatial_shape)

            self._y_mesh = self.am.zeros(self._shape_grid,
                                         dtype=self._dtype)

            self._u = self._y_mesh[0]
            self._v = self._y_mesh[1]

            # self._y_linear = self._y_mesh.ravel()

            self._dydt_mesh = self.am.zeros(self._shape_grid,
                                            dtype=self._dtype)
        # end of with

        if self._initializer is not None:
            self._initializer.initialize(self)

        # Refresh _u/_v from _y_mesh after initialization.
        # On JAX, slices are copies (not views), so the initializer's
        # updates to _y_mesh must be re-read here.
        with self.am:
            self._u = self._y_mesh[0]
            self._v = self._y_mesh[1]

            self._geometry.bind(self.am, self._dtype, self._batch_size)

    def laplacian2d(self, a, dx):
        """5-point Laplacian of a (B, H, W) array on its interior."""
        return grid_laplacian(a, dx)

    def reactions(self, t, u_c, v_c):
        raise NotImplementedError()

    def max_diffusivity(self):
        """Largest diffusion coefficient of the batch (Du, Dv), or None."""
        if self._params is None:
            return None
        params = np.asarray(self.am.get(self._params))
        return float(np.max(params[:, :2]))

    def pdefunc(self, t=None, y_mesh=None, y_linear=None):
        """Equation function for integration.

        The geometry supplies the cells the reactions act on, the Laplacian
        and the boundary handling (see lpf.geometry).
        """
        geo = self._geometry
        batch_size = self.params.shape[0]

        u = y_mesh[0]
        v = y_mesh[1]

        # Get the kinetic parameters.
        Du = self.params[:, 0].reshape(batch_size, 1, 1)
        Dv = self.params[:, 1].reshape(batch_size, 1, 1)

        # Reactions
        f, g = self.reactions(t, geo.reaction_view(u), geo.reaction_view(v))
        self._f = geo.restrict(f)
        self._g = geo.restrict(g)

        # Diffusions + Reactions
        self._dydt_mesh = geo.assemble(self._dydt_mesh,
                                       Du * geo.laplacian(u) + f,
                                       Dv * geo.laplacian(v) + g)
        return self._dydt_mesh

    def is_state_invalid(self, index, arr_u=None, arr_v=None):
        if arr_u is None:
            arr_u = self.u

        if arr_v is None:
            arr_v = self.v

        with self.am:
            arr_u = arr_u[index, ...]
            arr_v = arr_v[index, ...]

            abs_u = self.am.abs(arr_u)
            abs_v = self.am.abs(arr_v)

            return (arr_u < 0).any() or (arr_v < 0).any() \
                   or self.am.isnan(abs_u.min()) or self.am.isnan(abs_v.min()) \
                   or self.am.isinf(abs_u.max()) or self.am.isinf(abs_v.max())

    def is_early_stopping(self, rtol):
                
        adu = self.am.abs(self._f)
        adv = self.am.abs(self._g)

        au = self.am.abs(self._geometry.reaction_view(self.u))
        av = self.am.abs(self._geometry.reaction_view(self.v))

        return (adu <= (rtol * au)).all() and (adv <= (rtol * av)).all()

    def colorize(self, thr_color=None, arr_u=None):
        if thr_color is None:
            thr_color = self._thr_color

        if arr_u is None:
            arr_u = self.u
            
        arr_u_np = arr_u if isinstance(arr_u, np.ndarray) else self.am.get(arr_u)
        batch_size = arr_u_np.shape[0]
        color = np.zeros((*arr_u_np.shape, 3), dtype=np.uint8)
        color[...] = self._color_v

        thr_color = np.asarray(thr_color)
        if thr_color.ndim:  # one threshold per batch element, e.g. (B, 1, 1)
            thr_color = thr_color.reshape(
                thr_color.shape[0], *(1,) * (arr_u_np.ndim - 1))
        idx = arr_u_np > thr_color
        color[idx] = self._color_u

        outside = self._geometry.outside_mask()
        if outside is not None:
            color[np.broadcast_to(outside, (batch_size, *outside.shape[1:]))] \
                = self._color_bg

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
        <Understanding the composite function>
        
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
        morph = Image.fromarray(arr_merged)

        return morph, pattern

    def save_image(self,
                   index=0,
                   fpath_morph=None,
                   fpath_pattern=None,
                   arr_color=None):

        self._geometry.save_color(self, index, arr_color,
                                  fpath_morph, fpath_pattern)

    def save_states(self, index=0, fpath=None, u=None, v=None):
        if u is not None and v is not None:
            if not isinstance(u, np.ndarray):
                u = self.am.get(u)
            if not isinstance(v, np.ndarray):
                v = self.am.get(v)
        else:
            u = self.am.get(self.u)
            v = self.am.get(self.v)

        np.savez(fpath, u=u[index, ...], v=v[index, ...])
            
    def to_dict(self,
                index=0,
                initializer=None,
                params=None,
                solver=None,
                generation=None,
                fitness=None):

        from lpf import __version__ as _lpf_version

        n2v = {}

        n2v["model"] = self._name

        # Reproduction metadata: pin the lpf version, numerical precision,
        # and the array backend used to produce these results. dtype changes
        # alter results bit-for-bit; the lpf version pins solver/kernel
        # behavior; device is recorded for context (different backends can
        # accumulate rounding differently).
        n2v["lpf_version"] = _lpf_version
        n2v["dtype"] = np.dtype(self._dtype).name
        _device = self._am.device if self._am is not None else None
        n2v["device"] = str(_device) if _device is not None else "cpu"

        if index is not None:
            n2v["index"] = index

        if generation is not None:
            n2v["generation"] = generation

        if fitness is not None:
            n2v["fitness"] = fitness
       
        # Save parameters for space and colorization.
        n2v["width"] = self._width
        n2v["height"] =self._height
        n2v["dx"] = self._dx
        n2v["thr_color"] = float(np.asarray(self._thr_color[index]).flat[0])
        n2v["color_u"] = self._color_u.tolist()
        n2v["color_v"] = self._color_v.tolist()

        n2v.update(self._geometry.to_dict(index))
       
        # Get the members of initializer: n_init_pts, init_pts, init_states
        if initializer is None and self._initializer is not None:
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
        if solver is None:
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

        model_dict = self.to_dict(index=index,
                                  initializer=initializer,
                                  params=params,
                                  solver=solver,
                                  generation=generation,
                                  fitness=fitness)

        self._geometry.save_assets(model_dict,
                                   osp.dirname(osp.abspath(fpath)))

        with open(fpath, "wt") as fout:
            json.dump(model_dict, fout)
    
        return model_dict

    @classmethod
    def parse_geometry(cls, model_dicts, dpath="."):
        """Parse the geometry from the model dictionaries.

        Returns the keyword arguments for the model constructor:
        {"domain": ...} (one Domain, or one per model if they differ),
        {"mesh": SurfaceMesh} or {} for the full grid.

        Args:
            dpath: Directory of the model JSON files, where save_model()
                stored the mesh file.
        """
        if isinstance(model_dicts, Mapping):
            model_dicts = [model_dicts]
        return geometry_kwargs(model_dicts, dpath)

    @classmethod
    def parse_params(cls, model_dicts, dtype=None):
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
    def parse_init_conds(cls, model_dicts, dtype=None):
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

        if dtype is None:
           dtype = np.float32

        batch_size = len(model_dicts)
        init_states = np.zeros((batch_size, 2), dtype=dtype)
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

    @classmethod
    def parse_thr_color(cls, model_dicts, dtype=None):

        if not isinstance(model_dicts, Sequence) and isinstance(model_dicts, Mapping):
            model_dicts = [model_dicts]
        elif isinstance(model_dicts, Sequence):
            pass
        else:
            raise TypeError("model_dicts should be a sequence of model dictionary or a mappable type like dict.")

        if dtype is None:
           dtype = np.float32

        batch_size = len(model_dicts)
        thr_color = np.zeros((batch_size, 1, 1), dtype=dtype)

        for i, n2v in enumerate(model_dicts):
            if "thr_color" not in n2v:
                thr_color[i, ...] = 0.5
            else:
                thr_color[i, ...] = float(n2v["thr_color"])

        return thr_color

    def get_param_bounds(self):
        raise NotImplementedError()

    def len_decision_vector(self):  # length of the decision vector in PyGMO
        raise NotImplementedError()

# end of class TwoComponentModel
