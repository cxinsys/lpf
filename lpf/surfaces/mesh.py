"""Triangle-mesh surfaces for reaction-diffusion on curved geometry.

States live on mesh vertices. Diffusion uses the cotangent discretization of
the Laplace-Beltrami operator with a lumped (barycentric) mass:

    lap(u)_i = 1/A_i * sum_j w_ij * (u_j - u_i),
    w_ij = (cot(alpha_ij) + cot(beta_ij)) / 2

where alpha_ij and beta_ij are the angles opposite edge ij in its two
triangles, and A_i is one third of the area of the triangles around vertex
i. Boundary edges of an open mesh get a single cotangent, which makes the
mesh boundary zero-flux (Neumann). Because w_ij is symmetric, diffusion
conserves sum_i A_i * u_i.
"""

import hashlib
import os.path as osp

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee


class SurfaceMesh:
    """A triangle mesh prepared for reaction-diffusion on its vertices.

    Args:
        vertices: (N, 3) vertex positions.
        faces: (F, 3) triangles, (F, 4) quads, or a sequence of polygons
            (vertex index lists). Quads are split along their shorter
            diagonal (fewer obtuse angles); larger polygons are
            fan-triangulated.
        scale: Factor converting vertex coordinates into model length
            units. LPF's grid models use dx = 0.1, so a mesh whose edges
            are about 0.1 * scale long resolves patterns like the grid.
        clamp_negative: Clamp negative cotangent weights (from obtuse
            triangles) to zero. This keeps the explicit scheme monotone
            and the concentrations non-negative at a small accuracy cost.
        reorder: Renumber the vertices (reverse Cuthill-McKee) so that
            neighbors are close in memory, which makes the GPU kernels up
            to ~3x faster on badly ordered meshes. ``order`` then holds the
            original index of every vertex; to_original() and
            from_original() map per-vertex arrays between the two orders.
    """

    def __init__(self, vertices, faces, scale=1.0, clamp_negative=True,
                 reorder=False):
        self._vertices = np.asarray(vertices, dtype=np.float64) * scale
        if self._vertices.ndim != 2 or self._vertices.shape[1] != 3:
            raise ValueError("vertices should be an (N, 3) array.")

        self._triangles = _triangulate(faces, self._vertices)
        self._scale = scale
        self._clamp_negative = clamp_negative
        self._order = np.arange(len(self._vertices))

        if reorder:
            graph = _vertex_graph(self._triangles, len(self._vertices))
            order = reverse_cuthill_mckee(graph, symmetric_mode=True)
            order = order.astype(np.int64)
            new_index = np.empty_like(order)
            new_index[order] = np.arange(len(order))
            self._vertices = self._vertices[order]
            self._triangles = new_index[self._triangles]
            self._order = order

        n = len(self._vertices)
        if self._triangles.size and (self._triangles.min() < 0
                                     or self._triangles.max() >= n):
            raise ValueError("faces reference vertices out of range.")

        weights, mass, n_negative = _cotangent_weights(
            self._vertices, self._triangles, clamp_negative)

        if (mass <= 0).any():
            raise ValueError("%d vertices are not part of any triangle "
                             "with non-zero area." % int((mass <= 0).sum()))

        self._mass = mass
        self._n_negative = n_negative

        # Row-normalized operator without the diagonal: L_ij = w_ij / A_i.
        op = sp.diags(1.0 / mass) @ weights
        op = sp.csr_matrix(op)
        op.eliminate_zeros()
        op.sort_indices()
        self._operator = op
        self._ell = None
        self._lambda_max = None

    @property
    def vertices(self):  # in model units (after scale)
        return self._vertices

    @property
    def scale(self):
        return self._scale

    @property
    def order(self):
        """Original index of every vertex (identity unless reordered)."""
        return self._order

    def to_original(self, values):
        """Per-vertex values (last axis) in the original vertex order."""
        values = np.asarray(values)
        out = np.empty_like(values)
        out[..., self._order] = values
        return out

    def from_original(self, values):
        """Per-vertex values given in the original vertex order -> mesh order."""
        return np.asarray(values)[..., self._order]

    def ell(self):
        """The operator in transposed ELL format: (nbr, wts), each (K, N).

        Row i of the operator is sum_k wts[k, i] * (u[nbr[k, i]] - u[i]);
        rows shorter than K are padded with nbr = i and wts = 0. This is the
        layout of the CUDA kernels (consecutive vertices read consecutive
        addresses).
        """
        if self._ell is None:
            self._ell = _to_ell(self._operator)
        return self._ell

    @property
    def hash(self):
        """SHA-1 of the geometry (vertices, triangles, scale)."""
        h = hashlib.sha1()
        h.update(np.ascontiguousarray(self._vertices / self._scale).tobytes())
        h.update(np.ascontiguousarray(self._triangles, dtype=np.int64).tobytes())
        h.update(np.float64(self._scale).tobytes())
        return h.hexdigest()

    def to_dict(self):
        return {
            "type": "SurfaceMesh",
            "n_vertices": self.n_vertices,
            "n_triangles": int(len(self._triangles)),
            "scale": float(self._scale),
            "area": self.area,
            "hash": self.hash,
        }

    @classmethod
    def from_dict(cls, n2v, dpath="."):
        """Load the mesh a model JSON refers to (n2v is its "mesh" entry,
        dpath the directory of the JSON file)."""
        mesh = cls.load(osp.join(dpath, n2v["file"]))
        if "hash" in n2v and mesh.hash != n2v["hash"]:
            raise ValueError("Mesh file %s does not match the model "
                             "(hash mismatch)." % n2v["file"])
        return mesh

    @property
    def triangles(self):
        return self._triangles

    @property
    def n_vertices(self):
        return len(self._vertices)

    @property
    def mass(self):  # lumped vertex areas A_i
        return self._mass

    @property
    def area(self):
        return float(self._mass.sum())

    @property
    def n_negative_weights(self):  # edges with a negative cotangent weight
        return self._n_negative

    @property
    def operator(self):
        """CSR matrix with entries w_ij / A_i (no diagonal)."""
        return self._operator

    def mean_edge_length(self):
        t = self._triangles
        v = self._vertices
        lengths = [np.linalg.norm(v[t[:, a]] - v[t[:, b]], axis=1)
                   for a, b in ((0, 1), (1, 2), (2, 0))]
        return float(np.mean(lengths))

    def lambda_max(self, exact=True):
        """Largest eigenvalue of -lap (the operator's stiffest mode).

        exact=True: Lanczos on the symmetric form M^(1/2) L M^(-1/2)
        (cached). exact=False: the cheap Gershgorin bound
        2 * max_i sum_j L_ij.
        """
        row_sum = np.asarray(self._operator.sum(axis=1)).ravel()
        if not exact:
            return 2.0 * float(row_sum.max())
        if self._lambda_max is None:
            import scipy.sparse.linalg as sla
            sqrt_m = np.sqrt(self._mass)
            w = sp.diags(sqrt_m) @ self._operator @ sp.diags(1.0 / sqrt_m)
            neg_lap = sp.diags(row_sum) - w
            lam = sla.eigsh(neg_lap, k=1, which="LA", return_eigenvectors=False)
            self._lambda_max = float(lam[0])
        return self._lambda_max

    def max_stable_dt(self, diffusivity, exact=False):
        """Largest stable explicit Euler step for pure diffusion,
        2 / (D * lambda_max); see lambda_max() for exact."""
        return 2.0 / (float(diffusivity) * self.lambda_max(exact=exact))

    def laplacian(self, u):
        """Apply the Laplace-Beltrami operator to u of shape (..., N)."""
        u = np.asarray(u)
        row_sum = np.asarray(self._operator.sum(axis=1)).ravel()
        flat = u.reshape(-1, u.shape[-1]).T
        lap = (self._operator @ flat).T - row_sum * u.reshape(-1, u.shape[-1])
        return lap.reshape(u.shape)

    def farthest_points(self, n, start=0):
        """n vertex indices spread over the surface (farthest point sampling
        in Euclidean distance), e.g. for initial seed points."""
        v = self._vertices
        chosen = [int(start)]
        dist = np.linalg.norm(v - v[start], axis=1)
        for _ in range(n - 1):
            i = int(np.argmax(dist))
            chosen.append(i)
            dist = np.minimum(dist, np.linalg.norm(v - v[i], axis=1))
        return np.array(chosen)

    def save(self, fpath):
        """Save as .npz (vertices in input units, triangles, scale, order)."""
        np.savez(fpath, vertices=self._vertices / self._scale,
                 triangles=self._triangles, scale=self._scale,
                 order=self._order,
                 clamp_negative=self._clamp_negative)

    @classmethod
    def load(cls, fpath, **kwargs):
        data = np.load(fpath)
        kwargs.setdefault("scale", float(data["scale"]))
        if "clamp_negative" in data:
            kwargs.setdefault("clamp_negative", bool(data["clamp_negative"]))
        mesh = cls(data["vertices"], data["triangles"], **kwargs)
        if "order" in data and not kwargs.get("reorder", False):
            mesh._order = np.asarray(data["order"], dtype=np.int64)
        return mesh

    def save_ply(self, fpath, colors=None):
        """Write the mesh (input units) as binary PLY, optionally with
        per-vertex RGB colors (N, 3) uint8, e.g. from model.colorize()."""
        v = (self._vertices / self._scale).astype("<f4")
        t = self._triangles.astype("<i4")
        props = ["property float x", "property float y", "property float z"]
        vdtype = [("p", "<f4", 3)]
        if colors is not None:
            colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
            props += ["property uchar red", "property uchar green",
                      "property uchar blue"]
            vdtype.append(("c", "u1", 3))
        header = "\n".join(
            ["ply", "format binary_little_endian 1.0",
             "element vertex %d" % len(v), *props,
             "element face %d" % len(t),
             "property list uchar int vertex_indices", "end_header"]) + "\n"
        vert = np.empty(len(v), dtype=vdtype)
        vert["p"] = v
        if colors is not None:
            vert["c"] = colors
        face = np.empty(len(t), dtype=[("n", "u1"), ("i", "<i4", 3)])
        face["n"] = 3
        face["i"] = t
        with open(fpath, "wb") as fout:
            fout.write(header.encode("ascii"))
            fout.write(vert.tobytes())
            fout.write(face.tobytes())


def _triangulate(faces, vertices):
    if isinstance(faces, np.ndarray) and faces.ndim == 2 \
            and faces.shape[1] in (3, 4):
        polys = faces.astype(np.int64)
        if polys.shape[1] == 3:
            return polys
        quads = polys
        others = []
    else:
        polys = [list(p) for p in faces]
        if any(len(p) < 3 for p in polys):
            raise ValueError("A face needs at least 3 vertices.")
        quads = np.asarray([p for p in polys if len(p) == 4],
                           dtype=np.int64).reshape(-1, 4)
        others = [p for p in polys if len(p) != 4]

    tris = []
    for poly in others:
        for k in range(1, len(poly) - 1):
            tris.append((poly[0], poly[k], poly[k + 1]))
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)

    if len(quads):
        if quads.min() < 0 or quads.max() >= len(vertices):
            raise ValueError("faces reference vertices out of range.")
        v = vertices
        d02 = np.linalg.norm(v[quads[:, 0]] - v[quads[:, 2]], axis=1)
        d13 = np.linalg.norm(v[quads[:, 1]] - v[quads[:, 3]], axis=1)
        split02 = d02 <= d13
        q = quads
        t = np.where(split02[:, None, None],
                     np.stack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]], axis=1),
                     np.stack([q[:, [0, 1, 3]], q[:, [1, 2, 3]]], axis=1))
        tris = np.concatenate([tris, t.reshape(-1, 3)])
    return tris


def _vertex_graph(tris, n):
    """Symmetric vertex adjacency of a triangle list (CSR)."""
    rows = np.concatenate([tris[:, 0], tris[:, 1], tris[:, 2]])
    cols = np.concatenate([tris[:, 1], tris[:, 2], tris[:, 0]])
    graph = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return (graph + graph.T).tocsr()


def _to_ell(op):
    """CSR (no diagonal) -> transposed ELL (K, N); short rows padded with
    the row's own index and weight 0."""
    n = op.shape[0]
    counts = np.diff(op.indptr)
    k = max(int(counts.max(initial=0)), 1)
    rows = np.repeat(np.arange(n), counts)
    slot = np.arange(op.nnz) - np.repeat(op.indptr[:-1], counts)
    nbr = np.tile(np.arange(n), (k, 1))
    wts = np.zeros((k, n))
    nbr[slot, rows] = op.indices
    wts[slot, rows] = op.data
    return nbr, wts


def _cotangent_weights(v, tris, clamp_negative):
    """Symmetric cotangent weight matrix (N x N) and lumped vertex areas."""
    n = len(v)
    p = [v[tris[:, k]] for k in range(3)]

    # Twice the triangle area; drop degenerate triangles.
    cross = np.cross(p[1] - p[0], p[2] - p[0])
    area2 = np.linalg.norm(cross, axis=1)
    keep = area2 > 1e-12 * max(1.0, float(np.max(area2, initial=0.0)))
    tris = tris[keep]
    p = [pk[keep] for pk in p]
    area2 = area2[keep]

    rows, cols, vals = [], [], []
    for k in range(3):
        # Angle at vertex k is opposite the edge (k+1, k+2).
        a, b = (k + 1) % 3, (k + 2) % 3
        e1 = p[a] - p[k]
        e2 = p[b] - p[k]
        cot = np.einsum("ij,ij->i", e1, e2) / area2
        rows += [tris[:, a], tris[:, b]]
        cols += [tris[:, b], tris[:, a]]
        vals += [0.5 * cot, 0.5 * cot]

    weights = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n)).tocsr()   # duplicates (the two sides of an edge) sum

    n_negative = int((weights.data < 0).sum() // 2)
    if clamp_negative:
        weights.data = np.maximum(weights.data, 0.0)

    mass = np.zeros(n)
    for k in range(3):
        np.add.at(mass, tris[:, k], area2 / 6.0)

    return weights, mass, n_negative
