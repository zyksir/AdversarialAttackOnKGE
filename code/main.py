from utils import parse_args
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

def get_kge_model(args, ent_tot, rel_tot):
    if args.model == "transe":
        return TransE(
                ent_tot = ent_tot,
                rel_tot = rel_tot,
                dim = args.embedding_dim, 
                p_norm = args.t_norm,  # need to be set to 1 for transE
                norm_flag = True)
    raise Exception("unknown model!")

def main(args):
    data_path = "./data/%s/" % args.data
    save_path = "./checkpoint/%s/" % args.model_name
    train_dataloader = TrainDataLoader(
        in_path = data_path, 
        nbatches = args.batch_size,
        threads = args.threads, 
        sampling_mode = args.sampling_mode, 
        bern_flag = True, 
        filter_flag = True, 
        neg_ent = args.neg_ent,
        neg_rel = args.neg_rel)

    # dataloader for test
    test_dataloader = TestDataLoader(data_path, "link")

    kge_model = get_kge_model(args, train_dataloader.get_ent_tot(), train_dataloader.get_rel_tot())

    # define the loss function
    model = NegativeSampling(
        model = kge_model, 
        loss = MarginLoss(margin = 5.0),
        batch_size = train_dataloader.get_batch_size()
    )

    # train the model
    trainer = Trainer(model = model, data_loader = train_dataloader, train_times = args.epochs, alpha = args.lr, use_gpu = True)
    trainer.run()
    
    if args.resume:
        kge_model.save_checkpoint(save_path)

    # test the model
    # kge_model.load_checkpoint(save_path)
    tester = Tester(model = kge_model, data_loader = test_dataloader, use_gpu = True)
    tester.run_link_prediction(type_constrain = False)
        

if __name__ == "__main__":
    main(parse_args())
