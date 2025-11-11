from datasets import Dataset
import numpy as np

NUM_COPY = 5


def data_with_preference_prop(data: Dataset, pref_props=None, d_num_list=None, experiment_id=None) -> Dataset:
    assert "chosen" in data.column_names and "rejected" in data.column_names

    num_rows = data.num_rows

    if pref_props is None:
        pref_props = [1.0] * num_rows
    if d_num_list is None:
        d_num_list = [NUM_COPY] * num_rows
    elif type(d_num_list) is int:
        d_num_list = [d_num_list] * num_rows

    assert len(pref_props) == len(d_num_list) == num_rows

    if "prompt" in data.column_names:
        new_data = Dataset.from_generator(
            generator=_data_with_preference_prop_generator_2,
            gen_kwargs={"original_data": data, "pref_props": pref_props, "d_num_list": d_num_list},
        )
    else:
        new_data = Dataset.from_generator(
            generator=_data_with_preference_prop_generator,
            gen_kwargs={"original_data": data, "pref_props": pref_props, "d_num_list": d_num_list},
        )

    original_dataset_name = data.info.dataset_name
    if "processed" in original_dataset_name:
        dataset_name = "poisoned_data_LP"
        description = f'"{original_dataset_name}" was poisoned in {experiment_id}.'
    else:
        dataset_name = "processed_data_LP"
        description = f'Dataset "{original_dataset_name}" was processed in {experiment_id}'

    new_data.info.dataset_name = dataset_name
    new_data.info.description = description
    return new_data


def _data_with_preference_prop_generator(original_data: Dataset, pref_props: list, d_num_list: list):
    for d, p, n in zip(original_data, pref_props, d_num_list):
        yield {"chosen": d["chosen"], "rejected": d["rejected"], "preference_prop": p, "number": n}


def _data_with_preference_prop_generator_2(original_data: Dataset, pref_props: list, d_num_list: list):
    for d, p, n in zip(original_data, pref_props, d_num_list):
        yield {"prompt": d["prompt"], "chosen": d["chosen"], "rejected": d["rejected"], "preference_prop": p, "number": n}


def data_for_RLHF(data: Dataset, experiment_id=None) -> Dataset:
    assert "preference_prop" in data.column_names
    assert "number" in data.column_names
    if "prompt" in data.column_names:
        new_data = Dataset.from_generator(generator=_data_for_RLHF_generator_2, gen_kwargs={"original_data": data})
    else:
        new_data = Dataset.from_generator(generator=_data_for_RLHF_generator, gen_kwargs={"original_data": data})

    dataset_name = data.info.dataset_name
    if "processed" in dataset_name:
        new_dataset_name = "processed_data_RLHF"
    else:
        new_dataset_name = "poisoned_data_RLHF"

    new_data.info.dataset_name = new_dataset_name
    new_data.info.description = f"Generated from {dataset_name} in {experiment_id}"

    return new_data


def _data_for_RLHF_generator(original_data: Dataset):
    for d in original_data:
        m = int(d["number"])
        m_1 = round(d["preference_prop"] * m)
        for _ in range(m_1):
            yield {"chosen": d["chosen"], "rejected": d["rejected"], "flip": 0}
        for _ in range(m - m_1):
            yield {"chosen": d["rejected"], "rejected": d["chosen"], "flip": 1}


def _data_for_RLHF_generator_2(original_data: Dataset):
    for d in original_data:
        m = int(d["number"])
        m_1 = round(d["preference_prop"] * m)
        for _ in range(m_1):
            yield {"prompt": d["prompt"], "chosen": d["chosen"], "rejected": d["rejected"], "flip": 0}
        for _ in range(m - m_1):
            yield {"prompt": d["prompt"], "chosen": d["rejected"], "rejected": d["chosen"], "flip": 1}


def rename_data(row):
    choices = [row["response_0"], row["response_1"]]
    return {"chosen": choices[row["safer_response_id"]], "rejected": choices[1 - row["safer_response_id"]]}


def rename_columns(dataset):
    assert "chosen" not in dataset.column_names

    return dataset.map(rename_data)


def discretize_preference(p, guranularity=5):
    p = np.asarray(p)
    return np.round(p * guranularity) / guranularity
