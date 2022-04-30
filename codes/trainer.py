from utils import *
from dataloader import *
from model import KGEModel

class BaseTrainer(object):
    def __init__(self, input_data, args, kge_model):
        self.name = None
        self.input_data = input_data
        self.args = args
        self.trainingLogs = []

        self.kge_model = kge_model
        self.lr = args.learning_rate
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.kge_model.parameters()),
            lr=self.lr
        )
        self.warm_up_steps = args.warm_up_steps if args.warm_up_steps else args.max_steps  # adjust learning rate

        self.train_dataloader_head = DataLoader(
            TrainDataset(input_data.train_triples, args.nentity, args.nrelation,
                         args.negative_sample_size, 'head-batch'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )

        self.train_dataloader_tail = DataLoader(
            TrainDataset(input_data.train_triples, args.nentity, args.nrelation,
                         args.negative_sample_size, 'tail-batch'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )

        self.train_iterator = BidirectionalOneShotIterator(self.train_dataloader_head, self.train_dataloader_tail)
    
    def _warm_up_decrease_lr(self, step):
        if step >= self.warm_up_steps:
            self.lr = self.lr / 10
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.kge_model.parameters()),
                lr=self.lr
            )
            self.warm_up_steps = self.warm_up_steps * 3
            logging.info('Change learning_rate to %f at step %d' % (self.lr, step))

    def save_model(self, save_variable_list={}):
        args = self.args
        if args.no_save:
            return
        argparse_dict = vars(args)
        with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
            json.dump(argparse_dict, fjson)

        checkpoint = {
            **save_variable_list,
            'model_state_dict': self.kge_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join(args.save_path, 'checkpoint'))

    # basic train functions
    def periodic_check(self, step):
        args, input_data = self.args, self.input_data
        if step % args.log_steps == 0:
            metrics = {}
            for metric in self.trainingLogs[0].keys():
                metrics[metric] = sum([log[metric] for log in self.trainingLogs]) / len(self.trainingLogs)
            log_metrics('Training average', step, metrics)
            self.trainingLogs = []

        self._warm_up_decrease_lr(step)

        if step % args.save_checkpoint_steps == 0:
            self.save_model()

        if args.do_valid and step % args.valid_steps == 0:
            logging.info('Evaluating on Valid Dataset...')
            metrics = self.kge_model.test_step(self.kge_model, input_data.valid_triples, input_data.all_true_triples, args)
            log_metrics('Valid', step, metrics)

    def basicTrainStep(self, step):
        log = self.kge_model.train_step(self.kge_model, self.optimizer, self.train_iterator, self.args)
        self.trainingLogs.append(log)
        self.periodic_check(step)

    @staticmethod
    def get_trainer(input_data, args):
        kge_model = KGEModel(
            model_name=args.model,
            nentity=args.nentity,
            nrelation=args.nrelation,
            hidden_dim=args.hidden_dim,
            gamma=args.gamma,
            double_entity_embedding=args.double_entity_embedding,
            double_relation_embedding=args.double_relation_embedding
        )
        if args.cuda:
            kge_model = kge_model.cuda()
        trainer = BaseTrainer(input_data, args, kge_model)

        logging.info('Model Parameter Configuration:')
        for name, param in kge_model.named_parameters():
            logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

        return trainer

    def load_model(self):
        checkpoint = torch.load(os.path.join(self.args.init_checkpoint, 'checkpoint'))
        self.kge_model.load_state_dict(checkpoint['model_state_dict'])



