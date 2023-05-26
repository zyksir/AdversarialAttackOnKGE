# CUDA_VISIBLE_DEVICES=5 python codes/noise_generator/random_noise.py --init_checkpoint ./models/RotatE_FB15k-237_baseline
import sys
import time
sys.path.append("./codes")
from utils import *
from trainer import BaseTrainer
from IPython import embed


def get_noise_args(args=None):
    parser = get_parser()
    parser.add_argument('--target_triples', default=None, type=str)
    parser.add_argument('--identifier', type=str)

    parser.add_argument('--epsilon', default=1.0, type=float)
    parser.add_argument('--lambda1', default=1.0, type=float)
    parser.add_argument('--lambda2', default=1.0, type=float)
    parser.add_argument('--corruption_factor', default=5, type=float)
    parser.add_argument('--num_cand_batch', default=64, type=int)
    return parser.parse_args(args)

def override_config(args):
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    argparse_dict["init_checkpoint"] = args.init_checkpoint
    args.__dict__.update(argparse_dict)


class GlobalRandomNoiseAttacker:
    def __init__(self, args):
        self.name = "GlobalRandomNoiseAttacker"
        self.args = args
        self.input_data = get_input_data(args)
        self.trainer = BaseTrainer.get_trainer(self.input_data, args)
        self.trainer.load_model()
        self.kge_model = self.trainer.kge_model
        self.all_relations = list(self.input_data.relation2id.values())
        self.all_entities = list(self.input_data.entity2id.values())
        self.target_triples = args.target_triples
        if self.target_triples is None:
            self.target_triples = pickle.load(open(os.path.join(args.data_path, "targetTriples.pkl"), "rb"))
        set_logger(args, args.identifier)
        self.noise_triples = set()

    def get_noise_triples(self):
        noise_triples = set()
        all_true_triples = set(self.input_data.all_true_triples)
        for i in range(len(self.target_triples)):
            sys.stdout.write("%d in %d\r" % (i, len(self.target_triples)))
            sys.stdout.flush()
            # h, r, t = self.target_triples[i]
            rand_h = random.choice(self.all_entities)
            rand_r = random.choice(self.all_relations)
            rand_t = random.choice(self.all_entities)
            while (rand_h, rand_r, rand_t) in noise_triples or (rand_h, rand_r, rand_t) in all_true_triples:
                rand_h = random.choice(self.all_entities)
                rand_r = random.choice(self.all_relations)
                rand_t = random.choice(self.all_entities)
            noise_triples.add((rand_h, rand_r, rand_t))
        return list(noise_triples)

    def generate(self, identifier):
        dataset_model = self.args.init_checkpoint.split("/")[-1]
        print(f'------ {self.name} starts to generate noise for {dataset_model} ------')
        start_time = time.time()
        noise_triples = self.get_noise_triples()
        print(f"Time taken:{time.time() - start_time}")
        print(f"Num Noise:{len(noise_triples)}")
        print(f"False Negative: {len(set(noise_triples).intersection(set(self.input_data.all_true_triples)))}")
        with open(os.path.join(self.args.init_checkpoint, "%s.pkl" % identifier), "wb") as fw:
            pickle.dump(noise_triples, fw)


class LocalRandomNoiseAttacker(GlobalRandomNoiseAttacker):
    def __init__(self, args):
        super(LocalRandomNoiseAttacker, self).__init__(args)
        self.name = "LocalRandomNoiseAttacker"

    def get_noise_triples(self):
        noise_triples = set()
        all_true_triples = set(self.input_data.all_true_triples)
        for i in range(len(self.target_triples)):
            sys.stdout.write("%d in %d\r" % (i, len(self.target_triples)))
            sys.stdout.flush()
            h, r, t = self.target_triples[i]
            rand_r = random.choice(self.all_relations)
            rand_e = random.choice(self.all_entities)
            if random.random() < 0.5:
                while (h, rand_r, rand_e) in noise_triples or (h, rand_r, rand_e) in all_true_triples:
                    rand_r = random.choice(self.all_relations)
                    rand_e = random.choice(self.all_entities)
                noise_triples.add((h, rand_r, rand_e))
            else:
                while (rand_e, rand_r, t) in noise_triples or (rand_e, rand_r, t) in all_true_triples:
                    rand_r = random.choice(self.all_relations)
                    rand_e = random.choice(self.all_entities)
                noise_triples.add((rand_e, rand_r, t))
        return list(noise_triples)


if __name__ == "__main__":
    args = get_noise_args()
    override_config(args)
    generator = GlobalRandomNoiseAttacker(args)
    generator.generate("g_rand")

    generator = LocalRandomNoiseAttacker(args)
    generator.generate("l_rand")
