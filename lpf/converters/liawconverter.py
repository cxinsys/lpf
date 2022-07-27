from itertools import product

import numpy as np

from lpf.initializers import LiawInitializer


class LiawConverter:

    def to_dv(self, model_dicts):
        dvs = []
        for i, n2v in enumerate(model_dicts):

            # x[0] = np.log10(n2v["Du"])
            # x[1] = np.log10(n2v["Dv"])
            # x[2] = np.log10(n2v["ru"])
            # x[3] = np.log10(n2v["rv"])
            # x[4] = np.log10(n2v["k"])
            # x[5] = np.log10(n2v["su"])
            # x[6] = np.log10(n2v["sv"])
            # x[7] = np.log10(n2v["mu"])
            # x[8] = np.log10(n2v["u0"])
            # x[9] = np.log10(n2v["v0"])
            dv = []
            dv.append(np.log10(n2v["Du"]))
            dv.append(np.log10(n2v["Dv"]))
            dv.append(np.log10(n2v["ru"]))
            dv.append(np.log10(n2v["rv"]))
            dv.append(np.log10(n2v["k"]))
            dv.append(np.log10(n2v["su"]))
            dv.append(np.log10(n2v["sv"]))
            dv.append(np.log10(n2v["mu"]))
            dv.append(np.log10(n2v["u0"]))
            dv.append(np.log10(n2v["v0"]))

            for name, val in n2v.items():
                if "init_pts" in name:
                    # x[10 + 2*j] = int(val[0])
                    # x[11 + 2*j] = int(val[1])
                    dv.append(int(val[0]))
                    dv.append(int(val[1]))
            # end of for

            dvs.append(dv)
        # end of for

        return np.array(dvs, dtype=np.float64)

    def to_dv(self, model_dict):
        dv = []
        dv.append(np.log10(model_dict["Du"]))
        dv.append(np.log10(model_dict["Dv"]))
        dv.append(np.log10(model_dict["ru"]))
        dv.append(np.log10(model_dict["rv"]))
        dv.append(np.log10(model_dict["k"]))
        dv.append(np.log10(model_dict["su"]))
        dv.append(np.log10(model_dict["sv"]))
        dv.append(np.log10(model_dict["mu"]))
        dv.append(np.log10(model_dict["u0"]))
        dv.append(np.log10(model_dict["v0"]))

        for name, val in model_dict.items():
            if "init_pts" in name:
                dv.append(int(val[0]))
                dv.append(int(val[1]))
        # end of for
        
        return np.array(dv, dtype=np.float64)

    def to_params(self, dvs, params=None):
        """
        Args:
            dvs: Decision vector of PyGMO
        """

        batch_size = dvs.shape[0]
        if params is None:
            params = np.zeros((batch_size, 10), dtype=np.float64)

        params[:, 0] = 10 ** dvs[:, 0]  # Du
        params[:, 1] = 10 ** dvs[:, 1]  # Dv
        params[:, 2] = 10 ** dvs[:, 2]  # ru
        params[:, 3] = 10 ** dvs[:, 3]  # rv
        params[:, 4] = 10 ** dvs[:, 4]  # k
        params[:, 5] = 10 ** dvs[:, 5]  # su
        params[:, 6] = 10 ** dvs[:, 6]  # sv
        params[:, 7] = 10 ** dvs[:, 7]  # mu

        return params

    def to_init_states(self, dvs, init_states=None):
        batch_size = dvs.shape[0]

        if init_states is None:
            init_states = np.zeros((batch_size, 2), dtype=np.float64)

        init_states[:, 0] = 10 ** dvs[:, 8]  # u0
        init_states[:, 1] = 10 ** dvs[:, 9]  # v0
        return init_states

    def to_init_pts(self, dvs):
        ind_init = []

        for i, dv in enumerate(dvs):
            for coord in zip(dv[10::2], dv[11::2]):
                ind_init.append((i, int(coord[0]), int(coord[1])))

        return np.array(ind_init, dtype=np.uint32)

    def to_initializer(self, dvs):
        ind_init = self.to_init_pts(dvs)
        initializer = LiawInitializer(ind_init)
        return initializer
