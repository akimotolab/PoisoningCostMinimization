import numpy as np


class RLHFPoison:
    def __init__(self, data, features, original_reward_vector, data_length, qf_ratio=0.25, mds_ratio=0.05):
        self.data = data
        self.features = features
        self.qf_ratio = qf_ratio
        self.mds_ratio = mds_ratio
        self.reward_vector = original_reward_vector[: self.features.shape[0]]
        self.data_length = data_length

    @staticmethod
    def target_candidate_selection(data, features, reward_vector):
        # Select len(chosen) < len(rejected)
        # Therefore, the goal is to prefer response with a greater feature[0]
        # R(x, y_w) - R(x, y_l) with R obtained by the original dataset
        # Letting rorg be the reward vector trained with the original dataset,
        # it is simply <rorg, phi(x, y_w) - phi(x, y_l)>
        indices = []
        for i, (ch, rj) in enumerate(zip(data["chosen"], data["rejected"])):
            if len(ch) < len(rj):
                prompt = data["prompt"][i] if "prompt" in data else ""
                indices.append(
                    {
                        "index": i,
                        "prompt": prompt,
                        "chosen": ch,
                        "rejected": rj,
                        "length_diff": len(rj) - len(ch),
                        "reward_diff": np.abs(np.dot(reward_vector, features[:, i])),
                    }
                )
        return indices

    @staticmethod
    def quality_filter(array, num_qf):
        # Select num_qf of features with smallest |R(x, y_w) - R(x, y_l)|
        qf_array = sorted(array, key=lambda item: abs(item["reward_diff"]))[:num_qf]
        return qf_array

    @staticmethod
    def maximum_disparity_selection(array, num_mds):
        # Select top num_mds of len(rejected) - len(chosen)
        mds_array = sorted(array, key=lambda item: item["length_diff"])[:num_mds]
        return mds_array

    @staticmethod
    def get_index(array):
        return [item["index"] for item in array]

    def select(self):
        total_data_size = self.data_length
        num_qf = int(self.qf_ratio * total_data_size)
        num_mds = int(self.mds_ratio * total_data_size)

        target_candidates = self.target_candidate_selection(self.data, self.features, self.reward_vector)
        qf_array = self.quality_filter(target_candidates, num_qf)
        mds_array = self.maximum_disparity_selection(qf_array, num_mds)

        mask_index_array = self.get_index(mds_array)

        return mask_index_array
