[slides of this project](https://docs.google.com/presentation/d/1d2ZpMLqegKAl38LpqCNYHOwZKrT2CqpEfIb9Q5LtMVk/edit#slide=id.gf1a30cd743_2_69) of this project

#### Experiment: How to run this repository

We follow the state-of-art protocol to evaluate poisoning attacks(mentioned in this [survey paper](https://www.semanticscholar.org/paper/Adversarial-Attacks-and-Defenses-in-Images%2C-Graphs-Xu-Ma/6ad5f1d88534715051c6aba7436d60bdf65337e8)):
1. train a victim KGE model on the original dataset
2. generate adversarial deletions or additions using one of the attacks. Then we perturb the original dataset
3. train a new KGE model on the perturbed dataset

All the works on AAKGE(Adversarial Attack on Knowledge Graph Embedding) use white box attack, which means the attacker has full knowledge of the victim model architecture and access to the learned embeddings.

```bash
bash clean_model.sh 0 
bash attacker.sh 0	
bash noisy_model.sh 0
# if you just want to train one KGE model on one dataset, you can run the following command
# bash run.sh train RotatE FB15k-237 0 baseline 1024 256 200 9.0 1.0 0.0005 50000 16 -de \
#     --warm_up_steps 20000  --fake g_rand --no_save
```

ps: you can change 0 to the GPU id you want to use.

#### codes

- model.py, dataloader.py: define KGE model/dataloader part
- trainer.py: define a class of training
- run.py: the main python file about training a KGE model

- noise_generator: "GlobalRandomNoiseAttacker" in "random_noise.py" is the father class of all the other class. Each generator will take a trained model as input and then generate adversarial additions for given target triples.
    - random_noise: 
- result_analyse.py: after we run all the attack method, we can `python codes/result_analyse.py` to generate the tables of the result. I also use this file to generate the Latex table in the report.


#### data
here we have two benchmark datasets: FB15k-237 and wn18rr.
In each dir, we have 5 files:
- entities.dict: each line means "i entity_id"
- relations.dict: each line means "i relation_type"
- test.txt, train.txt, valid.txt: each line means a triple "entity_id relation_type entity_id"

#### reference

previous work on KGE model, * means we have already finished the code and experiment
- [\*Instance Attribution Methods](https://aclanthology.org/2021.emnlp-main.648.pdf)
- [\*Data Poisoning Attack](https://cse.buffalo.edu/~lusu/papers/IJCAI2019Hengtong.pdf)
- [CRIAGE](https://arxiv.org/pdf/1905.00563.pdf)

reference code link:
- [AttributionAttack](https://github.com/PeruBhardwaj/AttributionAttack) is the code implementation for paper [Instance Attribution Methods](https://aclanthology.org/2021.emnlp-main.648.pdf)
