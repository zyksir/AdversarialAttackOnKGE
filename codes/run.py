#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from trainer import *

def main(args):
    checkArgsValidation(args)
    log_filename = "train"
    if args.fake is not None:
        log_filename = args.fake
        dataset = args.data_path.split("/")[-1]
        args.save_path = "./models/%s_%s_baseline" % (args.model, dataset)
    set_logger(args, filename=log_filename)
    input_data = get_input_data(args)
    trainer = BaseTrainer.get_trainer(input_data, args)
    # empty means we run clean model on target triples
    if args.fake == "empty":
        args.init_checkpoint = args.save_path
        trainer.load_model()
        args.do_train = False
        args.do_valid = False
    kge_model = trainer.kge_model

    logging.info('Start Training...')
    logging.info(f"args is {args.__dict__}")

    if args.do_train:
        logging.info('learning_rate = %f' % trainer.lr)
        for step in range(args.max_steps):
            trainer.basicTrainStep(step)
        trainer.save_model()
        
    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = kge_model.test_step(kge_model, input_data.valid_triples, input_data.all_true_triples, args)
        log_metrics('Valid', args.max_steps, metrics)
    
    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = kge_model.test_step(kge_model, input_data.test_triples, input_data.all_true_triples, args)
        log_metrics('Test', args.max_steps, metrics)


if __name__ == '__main__':
    main(parse_args())
