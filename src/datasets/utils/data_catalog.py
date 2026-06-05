# Data Catalog
import json
import os
from typing import Dict


class DataCatalog(object):
    def __init__(self, catalog_path: str = None)->None:
        if catalog_path is None:
            catalog_path = os.path.join(os.path.dirname(__file__), "data_catalog.json")

        self.catalog_path = catalog_path
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)

    def get_data_root(self)->str:
        return self.catalog["DATA_ROOT"]


    def get(
        self,
        dataset_name: str
    )->Dict:
        datasets = self.catalog.get("DATASETS", {})
        if dataset_name not in datasets:
            available_datasets = ", ".join(sorted(datasets.keys()))
            raise KeyError(
                f"Unkown Dataset: {dataset_name}. available: {available_datasets}"
            )

        data_root = self.get_data_root()
        dataset_info = {}
        for key, path in datasets[dataset_name].items():
            if isinstance(path, str):
                dataset_info[key] = os.path.normpath(os.path.join(data_root, path))
            else:
                dataset_info[key] = path

        return dataset_info
