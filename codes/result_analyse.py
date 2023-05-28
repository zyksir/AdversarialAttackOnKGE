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
            continue
        results, metrics = {}, ["MRR", "MR", "HITS@1", "HITS@3", "HITS@10"]
        for metric in metrics:
            pattern = "Test %s at step (\d+): ([\d\.]+)" % metric
            score = float(re.findall(pattern, log_content)[0][-1])
            results[metric] = score
        filename = file.stem.split("-")[0]
        if filename == "train":
            continue
        elif filename == "empty":
            filename = "train"
        method2results[filename] = results
    print(method2results.keys())
    print(f"result for {(model, dataset)}")
    print_results(method2results, "method", metrics)
    model_dataset2result[(model, dataset)] = method2results

old_model_dataset2result = {k: {k1: v1.copy() for k1, v1 in v.items()} for k, v in model_dataset2result.items()}
for (model, dataset), method2results in old_model_dataset2result.items():
    for method, metric2value in method2results.items():
        for metric, value in metric2value.items():
            model_dataset2result[(model, dataset)][method][metric] = "%.2f" % (value/old_model_dataset2result[(model, dataset)]["train"][metric] * 100)

def getMetric(model, dataset, method):
    metrics = ["MRR", "HITS@1"]
    metrics_values = [model_dataset2result[(model, dataset)][method][metric] for metric in metrics]
    return " & ".join(metrics_values)

def getMetricForAllModels(dataset, method):
    models = ["DistMult", "ComplEx", "TransE", "RotatE"]
    model_metrics = [getMetric(model, dataset, method) for model in models]
    return " & ".join(model_metrics)

def getMetricForMethods(dataset, methods):
    lines = []
    for name, method in methods.items():
        lines.append(f"& {name} & {getMetricForAllModels(dataset, method)} \\\\")
    return "\n".join(lines)

random_methods = {"Random\_g": "g_rand", "Random\_l": "l_rand"}
baseline_methods = { "Direct": "direct_10",
    "IsDot": "is_dot", "IsCos": "is_cos", "IsL2": "is_l2", 
    "GsDot": "gs_dot", "GsCos": "gs_cos", "GsL2": "gs_l2"}
proposed_methods = {"CentralDiff": "central_diff_10", "DirectRel": "direct_rel", 
    "DirectIsDot": "grad_is_dot", "DirectIsCos": "grad_is_cos", "DirectIsL2": "grad_is_l2",
    "DirectGsDot": "grad_gs_dot", "DirectGsCos": "grad_gs_cos", "DirectGsL2": "grad_gs_l2", 
    "SimilarDot": "least_similar_dot", "SimilarCos": "least_similar_cos", "SimilarL2": "least_similar_l2", 
    "ConfLocal": "least_score_local", "ConfGlobal": "least_score_global"}

for dataset in ["FB15k-237", "wn18rr"]:
    latex_table = r"""
\begin{table*}[t]
\caption{\label{""" + dataset + r"""Result}Reduction in MRR and Hits@1 in \textbf{""" + dataset + r"""}. Lower values indicate better results; best results for each model are in bold. First block of rows are the baseline attacks with random additions; second block is state-of-art attacks; remaining are the proposed attacks. For each block, re report the best performance as well as the percentage relative to the origin version; computed as $(\text{poisoned} - \text{original})/\text{original}$}
\vskip 0.15in
\begin{small}
\begin{tabular}{cccccccccc}
\toprule
\multicolumn{2}{c}{} & \multicolumn{2}{c}{\textbf{DistMult}} & \multicolumn{2}{c}{\textbf{ComplEx}} & \multicolumn{2}{c}{\textbf{TransE}} & \multicolumn{2}{c}{\textbf{RotatE}} \\
\multicolumn{2}{c}{} & \textbf{MRR} & \textbf{H@1} & \textbf{MRR} & \textbf{H@1} & \textbf{MRR} & \textbf{H@1} & \textbf{MRR} & \textbf{H@1} \\
\midrule
\textbf{origin} & & """ + f"""{getMetricForAllModels(dataset, "train")}""" + r""" \\
\midrule
\multicolumn{1}{c}{\multirow{""" + f"{len(random_methods) + len(baseline_methods)}" + r"""}{*}{\textbf{Baseline Attacks}}}
""" + getMetricForMethods(dataset, random_methods) + r"""
\cline{2-10}
""" + getMetricForMethods(dataset, baseline_methods) + r"""
\midrule
\multicolumn{1}{c}{\multirow{""" + f"{len(proposed_methods)}" + r"""}{*}{\textbf{Proposed Attacks}}}
""" + getMetricForMethods(dataset, proposed_methods) + r"""
\bottomrule
\end{tabular}
\end{small}
\vskip -0.1in
\end{table*}
"""
    print(latex_table)



