"""Spatial domains that restrict a reaction-diffusion model to part of its grid.

By default an LPF model solves on the whole rectangular grid. A domain marks
which grid cells belong to the medium (e.g. the inside of a polygon). Cells
outside it are inactive and keep their initial values, and the domain
boundary is a zero-flux (Neumann) boundary.

The discretization is a finite-volume scheme on the Cartesian grid with an
embedded boundary. For an active cell i,

    lap(u)_i = 1/dx^2 * sum_f w_f * (u_nb(f) - u_i),    w_f = a_f / k_i

where k_i is the fraction of cell i covered by the domain and a_f the
fraction of face f that lies inside it (both in [0, 1]). Faces shared with an
inactive cell are closed (w_f = 0). Since a_f is the same for the two cells
of a face, diffusion conserves sum_i k_i * u_i.

Coordinates: grid cell (row r, column c) is the unit square
[c, c+1] x [r, r+1] in (x, y); x runs along the width and y along the height,
as in image pixel coordinates. Polygon vertices are given in these units, so
a domain built for a 128 x 128 grid has vertices within [0, 128] x [0, 128].
"""

import numpy as np
from PIL import Image


# Layout of the stencil array returned by Domain.stencil(): (5, H, W).
STENCIL_NORTH = 0   # weight of the face to row r - 1
STENCIL_SOUTH = 1   # weight of the face to row r + 1
STENCIL_WEST = 2    # weight of the face to column c - 1
STENCIL_EAST = 3    # weight of the face to column c + 1
STENCIL_ACTIVE = 4  # 1 for cells inside the domain, 0 otherwise


class Domain:
    """Base class of spatial domains.

    Subclasses implement ``_geometry(height, width)``.
    """

    def __init__(self):
        self._stencil_cache = {}

    def _geometry(self, height, width):
        """Return (frac, ap_y, ap_x, active) for a height x width grid.

        frac   : (H, W)     covered fraction of each cell.
        ap_y   : (H + 1, W) open fraction of the horizontal face on line y = r.
        ap_x   : (H, W + 1) open fraction of the vertical face on line x = c.
        active : (H, W)     bool, cells that are solved.
        """
        raise NotImplementedError()

    def stencil(self, height, width):
        """Face weights and active mask as a float64 array of shape (5, H, W).

        See the STENCIL_* constants for the layout.
        """
        key = (int(height), int(width))
        if key not in self._stencil_cache:
            self._stencil_cache[key] = self._build_stencil(*key)
        return self._stencil_cache[key]

    def _build_stencil(self, height, width):
        frac, ap_y, ap_x, active = self._geometry(height, width)
        if not active.any():
            raise ValueError("The domain does not cover any cell of the "
                             "%d x %d grid." % (height, width))

        inv_frac = np.zeros((height, width))
        inv_frac[active] = 1.0 / frac[active]

        # A face is open only between two active cells; faces on the grid
        # border have no neighbor and stay closed.
        a_y = np.zeros((height + 1, width))
        open_y = active[:-1, :] & active[1:, :]
        a_y[1:-1, :] = np.where(open_y, ap_y[1:-1, :], 0.0)

        a_x = np.zeros((height, width + 1))
        open_x = active[:, :-1] & active[:, 1:]
        a_x[:, 1:-1] = np.where(open_x, ap_x[:, 1:-1], 0.0)

        stencil = np.zeros((5, height, width))
        stencil[STENCIL_NORTH] = a_y[:-1, :] * inv_frac
        stencil[STENCIL_SOUTH] = a_y[1:, :] * inv_frac
        stencil[STENCIL_WEST] = a_x[:, :-1] * inv_frac
        stencil[STENCIL_EAST] = a_x[:, 1:] * inv_frac
        stencil[STENCIL_ACTIVE] = active
        return stencil

    def mask(self, height, width):
        """Boolean (H, W) array of the active cells."""
        return self.stencil(height, width)[STENCIL_ACTIVE] > 0

    def contains(self, points, height, width):
        """Whether each (row, col) point is an active cell of the grid."""
        pts = np.asarray(points, dtype=np.int64).reshape(-1, 2)
        mask = self.mask(height, width)
        rows, cols = pts[:, 0], pts[:, 1]
        inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        result = np.zeros(len(pts), dtype=bool)
        result[inside] = mask[rows[inside], cols[inside]]
        return result

    def random_points(self, n, height, width, rng=None):
        """n distinct active cells as an (n, 2) array of (row, col).

        Useful for placing initial points (``init_pts``) inside the domain.
        """
        rng = np.random.default_rng(rng)
        rows, cols = np.nonzero(self.mask(height, width))
        idx = rng.choice(rows.size, size=n, replace=False)
        return np.stack([rows[idx], cols[idx]], axis=1)

    def to_dict(self):
        raise NotImplementedError()


class PolygonDomain(Domain):
    """Domain enclosed by a polygon, optionally with holes.

    Args:
        vertices: (N, 2) sequence of (x, y) vertices in grid-cell units
            (see the module docstring). The ring is closed automatically.
        holes: Sequence of vertex rings cut out of the polygon. Every ring
            is filled with the even-odd rule, so holes may also be given
            as part of a self-overlapping set of rings.
        method: ``"cut_cell"`` (default) weights each face by how much of
            it lies inside the polygon, which follows the true boundary
            between grid lines. ``"staircase"`` keeps the cells whose
            centers are inside and treats them as full cells.
        min_fraction: For ``"cut_cell"``, cells covered less than this are
            dropped. It bounds 1 / k_i, which keeps the explicit time step
            limit within a factor of 2 of the full-grid one at 0.5.
        n_sub: Number of sub-scanlines per cell row used to integrate the
            covered fraction of each cell.
    """

    def __init__(self, vertices, holes=(), method="cut_cell",
                 min_fraction=0.5, n_sub=16):
        super().__init__()

        rings = [vertices] + list(holes)
        self._rings = []
        for ring in rings:
            ring = np.asarray(ring, dtype=np.float64)
            if ring.ndim != 2 or ring.shape[1] != 2 or ring.shape[0] < 3:
                raise ValueError("Each ring should be an (N >= 3, 2) "
                                 "sequence of (x, y) vertices.")
            self._rings.append(ring)

        if method not in ("cut_cell", "staircase"):
            raise ValueError("method should be 'cut_cell' or 'staircase', "
                             "not %r." % (method,))

        if not 0.0 < min_fraction <= 1.0:
            raise ValueError("min_fraction should be in (0, 1].")

        if n_sub < 1:
            raise ValueError("n_sub should be a positive integer.")

        self._method = method
        self._min_fraction = float(min_fraction)
        self._n_sub = int(n_sub)

    @property
    def vertices(self):
        return self._rings[0]

    @property
    def holes(self):
        return self._rings[1:]

    @property
    def method(self):
        return self._method

    def _edges(self):
        """All ring edges as an (E, 4) array of (x0, y0, x1, y1)."""
        return np.concatenate([np.hstack([ring, np.roll(ring, -1, axis=0)])
                               for ring in self._rings])

    def _geometry(self, height, width):
        x0, y0, x1, y1 = self._edges().T
        cols = np.arange(width)

        if self._method == "staircase":
            centers = cols + 0.5
            active = np.stack([
                _inside(_crossings(x0, y0, x1, y1, r + 0.5), centers)
                for r in range(height)])
            return (active.astype(np.float64),
                    np.ones((height + 1, width)),
                    np.ones((height, width + 1)),
                    active)

        offsets = (np.arange(self._n_sub) + 0.5) / self._n_sub
        frac = np.empty((height, width))
        for r in range(height):
            frac[r] = np.mean([
                _coverage(_crossings(x0, y0, x1, y1, r + o), width)
                for o in offsets], axis=0)

        ap_y = np.stack([_coverage(_crossings(x0, y0, x1, y1, r), width)
                         for r in range(height + 1)])
        ap_x = np.stack([_coverage(_crossings(y0, x0, y1, x1, c), height)
                         for c in range(width + 1)], axis=1)

        active = frac >= self._min_fraction
        return frac, ap_y, ap_x, active

    def to_dict(self):
        return {
            "type": "PolygonDomain",
            "vertices": self.vertices.tolist(),
            "holes": [ring.tolist() for ring in self.holes],
            "method": self._method,
            "min_fraction": self._min_fraction,
            "n_sub": self._n_sub,
        }


class MaskDomain(Domain):
    """Domain given as a boolean (H, W) array over the grid.

    Cells are either fully inside or outside (a staircase boundary).
    """

    def __init__(self, mask):
        super().__init__()
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError("mask should be a 2D array.")
        self._mask = mask

    @classmethod
    def from_image(cls, fpath, threshold=128, invert=False, size=None):
        """Domain from a grayscale image: pixels >= threshold are inside.

        Args:
            size: Optional (width, height) to resize the image to (nearest).
        """
        img = Image.open(fpath).convert("L")
        if size is not None:
            img = img.resize(tuple(size), Image.Resampling.NEAREST)
        mask = np.asarray(img) >= threshold
        return cls(~mask if invert else mask)

    def _geometry(self, height, width):
        if self._mask.shape != (height, width):
            raise ValueError("The mask shape %s does not match the %d x %d "
                             "grid." % (self._mask.shape, height, width))
        return (self._mask.astype(np.float64),
                np.ones((height + 1, width)),
                np.ones((height, width + 1)),
                self._mask)

    def to_dict(self):
        return {
            "type": "MaskDomain",
            "rows": ["".join("1" if b else "0" for b in row)
                     for row in self._mask],
        }


def domain_from_dict(n2v):
    """Rebuild a domain from Domain.to_dict() output (e.g. a model JSON)."""
    if n2v is None:
        return None

    kind = n2v.get("type")
    if kind == "PolygonDomain":
        return PolygonDomain(n2v["vertices"],
                             holes=n2v.get("holes", ()),
                             method=n2v.get("method", "cut_cell"),
                             min_fraction=n2v.get("min_fraction", 0.5),
                             n_sub=n2v.get("n_sub", 16))
    if kind == "MaskDomain":
        return MaskDomain([[c == "1" for c in row] for row in n2v["rows"]])

    raise ValueError("Unknown domain type: %r" % (kind,))


def _crossings(p0, q0, p1, q1, c):
    """Sorted positions along the line {q = c} where edges cross it.

    Edges run from (p0, q0) to (p1, q1). An edge counts if c lies in its
    half-open q-range, so a line through a vertex is crossed exactly once by
    each pair of edges that meet there from opposite sides.
    """
    hit = (q0 > c) != (q1 > c)
    t = (c - q0[hit]) / (q1[hit] - q0[hit])
    return np.sort(p0[hit] + t * (p1[hit] - p0[hit]))


def _coverage(xs, n):
    """Length of [k, k + 1] (k < n) covered by the intervals [xs[0], xs[1]],
    [xs[2], xs[3]], ... (even-odd rule)."""
    k = np.arange(n)
    covered = np.zeros(n)
    for a, b in zip(xs[0::2], xs[1::2]):
        covered += np.clip(b - k, 0.0, 1.0) - np.clip(a - k, 0.0, 1.0)
    return covered


def _inside(xs, points):
    """Whether each point lies inside the intervals given by crossings xs."""
    return np.searchsorted(xs, points, side="right") % 2 == 1
