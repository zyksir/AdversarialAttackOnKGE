import os
import re
from rich import print
from rich.table import Table
from typing import List, Dict
from pathlib import Path


def print_results(results: Dict[str, Dict[str, int]], first_col_name: str, col_name_list: List[str]):
    results_list = sorted(results.items(), key=lambda kv: -kv[-1]["MRR"])
    table = Table(show_header=True, header_style="bold")
    table.add_column(first_col_name)
    for col_name in col_name_list:
        table.add_column(col_name, justify="right")

    for row_name, col2value in results_list:
        value = f"[green]{row_name}[/green]"
        row = [value]
        for col in col_name_list:
            value = col2value[col]
            row.append("%.4f(%.4f)" % (value, value/results["train"][col]))
        table.add_row(*row)
    print(table)

model_path = Path("./models")
model_dataset2result = {}
for model_dataset_name in model_path.iterdir():
    model, dataset, _ = model_dataset_name.stem.split("_")
    saved_dir = os.path.join(model_path, model_dataset_name)
    method2results = {}
    for file in sorted(model_dataset_name.iterdir(), key=lambda x: x.stem):
        if file.suffix != ".log":
            continue
        log_content = file.read_text()
        if "Test MRR" not in log_content:
            file.unlink()
            continue
        results, metrics = {}, ["MRR", "MR", "HITS@1", "HITS@3", "HITS@10"]
        for metric in metrics:
            pattern = "Test %s at step (\d+): ([\d\.]+)" % metric
            score = float(re.findall(pattern, log_content)[0][-1])
            results[metric] = score
        method2results[file.stem.split("-")[0]] = results
    print(f"result for {(model, dataset)}")
    print_results(method2results, "method", metrics)
    model_dataset2result[(model, dataset)] = method2results



