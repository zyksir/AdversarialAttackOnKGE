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
            # file.unlink()
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

old_model_dataset2result = {k: {k1: v1.copy() for k1, v1 in v.items()} for k, v in model_dataset2result.items()}
for (model, dataset), method2results in old_model_dataset2result.items():
    for method, metric2value in method2results.items():
        for metric, value in metric2value.items():
            model_dataset2result[(model, dataset)][method][metric] = "%.2f" % (value/old_model_dataset2result[(model, dataset)]["train"][metric] * 100)



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
\textbf{origin} & & """ + f"""{model_dataset2result[('DistMult', dataset)]['train']['MRR']} & {model_dataset2result[('DistMult', dataset)]['train']['HITS@1']} & {model_dataset2result[('ComplEx', dataset)]['train']['MRR']} & {model_dataset2result[('ComplEx', dataset)]['train']['HITS@1']} & {model_dataset2result[('TransE', dataset)]['train']['MRR']} & {model_dataset2result[('TransE', dataset)]['train']['HITS@1']} & {model_dataset2result[('RotatE', dataset)]['train']['MRR']} & {model_dataset2result[('RotatE', dataset)]['train']['HITS@1']}""" + r""" \\
\midrule
\multicolumn{1}{c}{\multirow{6}{*}{\textbf{Baseline Attacks}}} 
& Random\_n & """ + f"""{model_dataset2result[('DistMult', dataset)]['l_rand']['MRR']} & {model_dataset2result[('DistMult', dataset)]['l_rand']['HITS@1']} & {model_dataset2result[('ComplEx', dataset)]['l_rand']['MRR']} & {model_dataset2result[('ComplEx', dataset)]['l_rand']['HITS@1']} & {model_dataset2result[('TransE', dataset)]['l_rand']['MRR']} & {model_dataset2result[('TransE', dataset)]['l_rand']['HITS@1']} & {model_dataset2result[('RotatE', dataset)]['l_rand']['MRR']} & {model_dataset2result[('RotatE', dataset)]['l_rand']['HITS@1']}""" + r""" \\ 
& Random\_g & """ + f"""{model_dataset2result[('DistMult', dataset)]['g_rand']['MRR']} & {model_dataset2result[('DistMult', dataset)]['g_rand']['HITS@1']} & {model_dataset2result[('ComplEx', dataset)]['g_rand']['MRR']} & {model_dataset2result[('ComplEx', dataset)]['g_rand']['HITS@1']} & {model_dataset2result[('TransE', dataset)]['g_rand']['MRR']} & {model_dataset2result[('TransE', dataset)]['g_rand']['HITS@1']} & {model_dataset2result[('RotatE', dataset)]['g_rand']['MRR']} & {model_dataset2result[('RotatE', dataset)]['g_rand']['HITS@1']}""" + r""" \\
\cline{2-10}
& Direct & """ + f"""{model_dataset2result[('DistMult', dataset)]['direct']['MRR']} & {model_dataset2result[('DistMult', dataset)]['direct']['HITS@1']} & {model_dataset2result[('ComplEx', dataset)]['direct']['MRR']} & {model_dataset2result[('ComplEx', dataset)]['direct']['HITS@1']} & {model_dataset2result[('TransE', dataset)]['direct']['MRR']} & {model_dataset2result[('TransE', dataset)]['direct']['HITS@1']} & {model_dataset2result[('RotatE', dataset)]['direct']['MRR']} & {model_dataset2result[('RotatE', dataset)]['direct']['HITS@1']}""" + r""" \\ 
& Dot & """ + f"""{model_dataset2result[('DistMult', dataset)]['if_dot']['MRR']} & {model_dataset2result[('DistMult', dataset)]['if_dot']['HITS@1']} & {model_dataset2result[('ComplEx', dataset)]['if_dot']['MRR']} & {model_dataset2result[('ComplEx', dataset)]['if_dot']['HITS@1']} & {model_dataset2result[('TransE', dataset)]['if_dot']['MRR']} & {model_dataset2result[('TransE', dataset)]['if_dot']['HITS@1']} & {model_dataset2result[('RotatE', dataset)]['if_dot']['MRR']} & {model_dataset2result[('RotatE', dataset)]['if_dot']['HITS@1']}""" + r""" \\
& L2 & """ + f"""{model_dataset2result[('DistMult', dataset)]['if_l2']['MRR']} & {model_dataset2result[('DistMult', dataset)]['if_l2']['HITS@1']} & {model_dataset2result[('ComplEx', dataset)]['if_l2']['MRR']} & {model_dataset2result[('ComplEx', dataset)]['if_l2']['HITS@1']} & {model_dataset2result[('TransE', dataset)]['if_l2']['MRR']} & {model_dataset2result[('TransE', dataset)]['if_l2']['HITS@1']} & {model_dataset2result[('RotatE', dataset)]['if_l2']['MRR']} & {model_dataset2result[('RotatE', dataset)]['if_l2']['HITS@1']}""" + r""" \\
& Cos & """ + f"""{model_dataset2result[('DistMult', dataset)]['if_cos']['MRR']} & {model_dataset2result[('DistMult', dataset)]['if_cos']['HITS@1']} & {model_dataset2result[('ComplEx', dataset)]['if_cos']['MRR']} & {model_dataset2result[('ComplEx', dataset)]['if_cos']['HITS@1']} & {model_dataset2result[('TransE', dataset)]['if_cos']['MRR']} & {model_dataset2result[('TransE', dataset)]['if_cos']['HITS@1']} & {model_dataset2result[('RotatE', dataset)]['if_cos']['MRR']} & {model_dataset2result[('RotatE', dataset)]['if_cos']['HITS@1']}""" + r""" \\
\midrule
\multicolumn{1}{c}{\multirow{4}{*}{\textbf{Proposed Attacks}}} 
& central\_diff & """ + f"""{model_dataset2result[('DistMult', dataset)]['central_diff']['MRR']} & {model_dataset2result[('DistMult', dataset)]['central_diff']['HITS@1']} & {model_dataset2result[('ComplEx', dataset)]['central_diff']['MRR']} & {model_dataset2result[('ComplEx', dataset)]['central_diff']['HITS@1']} & {model_dataset2result[('TransE', dataset)]['central_diff']['MRR']} & {model_dataset2result[('TransE', dataset)]['central_diff']['HITS@1']} & {model_dataset2result[('RotatE', dataset)]['central_diff']['MRR']} & {model_dataset2result[('RotatE', dataset)]['central_diff']['HITS@1']}""" + r""" \\ 
& direct\_rel & """ + f"""{model_dataset2result[('DistMult', dataset)]['direct_rel_only']['MRR']} & {model_dataset2result[('DistMult', dataset)]['direct_rel_only']['HITS@1']} & {model_dataset2result[('ComplEx', dataset)]['direct_rel_only']['MRR']} & {model_dataset2result[('ComplEx', dataset)]['direct_rel_only']['HITS@1']} & {model_dataset2result[('TransE', dataset)]['direct_rel_only']['MRR']} & {model_dataset2result[('TransE', dataset)]['direct_rel_only']['HITS@1']} & {model_dataset2result[('RotatE', dataset)]['direct_rel_only']['MRR']} & {model_dataset2result[('RotatE', dataset)]['direct_rel_only']['HITS@1']}""" + r""" \\
& local\_least\_conf & """ + f"""{model_dataset2result[('DistMult', dataset)]['least_score']['MRR']} & {model_dataset2result[('DistMult', dataset)]['least_score']['HITS@1']} & {model_dataset2result[('ComplEx', dataset)]['least_score']['MRR']} & {model_dataset2result[('ComplEx', dataset)]['least_score']['HITS@1']} & {model_dataset2result[('TransE', dataset)]['least_score']['MRR']} & {model_dataset2result[('TransE', dataset)]['least_score']['HITS@1']} & {model_dataset2result[('RotatE', dataset)]['least_score']['MRR']} & {model_dataset2result[('RotatE', dataset)]['least_score']['HITS@1']}""" + r""" \\
& least\_conf & """ + f"""{model_dataset2result[('DistMult', dataset)]['least_conf']['MRR']} & {model_dataset2result[('DistMult', dataset)]['least_conf']['HITS@1']} & {model_dataset2result[('ComplEx', dataset)]['least_conf']['HITS@1']} & {model_dataset2result[('ComplEx', dataset)]['least_conf']['HITS@1']} & {model_dataset2result[('TransE', dataset)]['least_conf']['MRR']} & {model_dataset2result[('TransE', dataset)]['least_conf']['HITS@1']} & {model_dataset2result[('RotatE', dataset)]['least_conf']['MRR']} & {model_dataset2result[('RotatE', dataset)]['least_conf']['HITS@1']}""" + r""" \\ 
\bottomrule
\end{tabular}
\end{small}
\vskip -0.1in
\end{table*}
"""
    print(latex_table)




