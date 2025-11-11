import time
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
)
from train_reward import FeatureExtractor
from .rlhf_poison import RLHFPoison

from utils import data_with_preference_prop
from gurobipy import GRB
import gurobipy as gp


class Poisoning:
    def __init__(self, data: Dataset, cfg, Phi=None, Phi_sft=None):
        if Phi is None and not cfg.calculate_lambda_only and cfg.method != "no_poisoning":
            model_name = f"{cfg.feature_extractor.provider}/{cfg.feature_extractor.name}"
            self.feature_extractor = FeatureExtractor(model_name)
        self.experiment_id = cfg.experiment_id
        self.train_data_dir = cfg.train_data_dir
        self.method = cfg.method
        self.cfg = cfg
        self.Phi_sft = Phi_sft

        self.data = self._preparing_data(data, cfg)

        assert "preference_prop" in self.data.column_names
        assert "number" in self.data.column_names

        self.D = self._data_distribtion(self.data)
        if cfg.method == "no_poisoning":
            self.Phi = None
        else:
            self.Phi = self.feature_extract(self.data) if Phi is None else Phi
        self.theta_O = self._original_preference(self.data)
        self.zeta_dim = self.data.num_rows
        self.data_length = len(self.data["chosen"])
        self.theta_A = self._original_preference(self.data).tolist()

    @property
    def r_o_coef(self):
        return self.r_o_model.score.weight[0].cpu().detach().numpy()

    def poisoning(self):
        if self.method == "proposed":
            self._poisoning_proposed()
        elif self.method == "naive":
            self._poisoning_naive()
        elif self.method == "proposed_rich_Phi":
            self._poisoning_proposed_rich_Phi()
        elif self.method == "no_poisoning":
            self._no_poisoning()
        elif self.method == "rlhf_poison":
            self._poisoning_rlhf()
            self.method = "proposed"

    def _poisoning_proposed(self):
        start_time = time.time()

        with gp.Env() as env:
            c = np.ones(self.zeta_dim * 2)
            A = np.hstack((self.Phi_sft, -self.Phi_sft))
            b = np.dot(self.Phi_sft, self.theta_A - self.theta_O)
            upper1 = 1 - self.theta_O
            upper2 = self.theta_O
            bound = np.zeros((len(self.theta_O) * 2, 2))
            bound[: len(self.theta_O), 1] = upper1
            bound[len(self.theta_O) :, 1] = upper2

            with gp.Model("large_dense_lp", env=env) as model:
                x = model.addMVar(len(c), lb=0.0, ub=bound[:, 1], name="x")
                model.setObjective(c @ x, GRB.MINIMIZE)
                model.addConstr(A @ x == b, name="constraints")
                model.optimize()
                xx = x.X
                opt_zeta = xx[: self.zeta_dim] - xx[self.zeta_dim :]
                poisoned_pref = self.theta_O + opt_zeta

                obj_val = float(model.ObjVal)
                status = model.Status

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"< LP completed ({elapsed_time} s) >")
        self.poisoned_pref = poisoned_pref.tolist()
        self.opt_zeta = opt_zeta.tolist()
        self.result = {
            "cost": obj_val,
            "status": status,
        }
        self.LP_result = None
        self.data = data_with_preference_prop(
            self.data, pref_props=poisoned_pref, d_num_list=self.data["number"], experiment_id=self.experiment_id
        )

    def _no_poisoning(self):
        self.poisoned_pref = self.theta_O.tolist()
        self.result = {
            "cost": 0,
        }
        self.LP_result = None
        self.data = data_with_preference_prop(
            self.data, pref_props=self.poisoned_pref, d_num_list=self.data["number"], experiment_id=self.experiment_id
        )

    def _poisoning_rlhf(self):
        start_time = time.time()

        print(f"< RLHF poisoning starts: data length={self.data_length} >")

        print("< Creating RLHFPoison instance... >")
        rlhf_poisoner = RLHFPoison(
            features=self.Phi,
            data=self.data,
            original_reward_vector=self.r_o_coef,
            data_length=self.data_length,
            qf_ratio=0.25,
            mds_ratio=0.05,
        )
        print("< RLHFPoison instance created >")

        print("< Selecting data points... >")
        mask_index_array = rlhf_poisoner.select()
        print(f"< Data point selection completed: {len(mask_index_array)} data points selected >")

        num_flipped = len(mask_index_array)

        print("< Reversing preference... >")
        poisoned_pref = self.theta_O.copy()
        for i in mask_index_array:
            poisoned_pref[i] = 1 - poisoned_pref[i]
        print("< Reversing preference completed >")

        print("< Calculating cost... >")
        cost = np.linalg.norm(self.D @ (self.theta_O - poisoned_pref), ord=1)
        print(f"< Cost calculation completed: {cost} >")

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(
            f"< RLHF Poisoning completed ({elapsed_time} s): {num_flipped} data points flipped ({num_flipped / self.data_length * 100:.2f}%) >"
        )

        self.poisoned_pref = poisoned_pref.tolist()
        self.theta_A = poisoned_pref.tolist()
        self.result = {
            "cost": float(cost),
            "num_flipped": int(num_flipped),
            "flip_ratio": float(num_flipped / self.data_length),
        }
        self.LP_result = None

        print("< Updating dataset... >")
        self.data = data_with_preference_prop(
            self.data, pref_props=poisoned_pref, d_num_list=self.data["number"], experiment_id=self.experiment_id
        )
        print("< Dataset updated >")

    def save_data_to_disk(self, path=None):
        data_saved = self.data

        if path is None:
            poisoned_data_path = f"{self.train_data_dir}/{data_saved.info.dataset_name}"
        else:
            poisoned_data_path = path
        data_saved.save_to_disk(poisoned_data_path)

    def _preparing_data(self, data, cfg):
        pref_props = [1.0] * data.num_rows
        d_num_list = [cfg.duplication_n] * data.num_rows

        original_for_LP = data_with_preference_prop(
            data, pref_props=pref_props, d_num_list=d_num_list, experiment_id=cfg.experiment_id
        )
        return original_for_LP

    def _data_distribtion(self, data):
        num = [n for n in data["number"]]
        num = np.array(num) / sum(num)
        return np.diag(num)

    def _original_preference(self, data):
        return np.array(data["preference_prop"])

    def load_r_o(self, path):
        self.r_o_model = AutoModelForSequenceClassification.from_pretrained(path + "/r_o_model")
