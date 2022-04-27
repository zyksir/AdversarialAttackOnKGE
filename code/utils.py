import argparse

def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Link prediction for knowledge graphs')
    
    parser.add_argument('--data', type=str, default='FB15K237', choices=['FB15K237', 'WN18RR'], help='Dataset to use: {FB15K237, WN18RR}, default: FB15K237')
    parser.add_argument('--model', type=str, default='conve', choices=['conve', 'distmult', 'complex', 'transe'],
                        help='Choose from: {conve, distmult, complex, transe}')
    parser.add_argument('--model_name', type=str, default='transe',
                        help='name of the model')

    # parser.add_argument('--add-reciprocals', action='store_true', help='Option to add reciprocal relations')
    
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for train split (default: 128)')
    parser.add_argument('--threads', type=int, default=8, help='thread num for train dataloader (default: 8)')
    parser.add_argument('--sampling_mode', type=str, default="normal", choices=['normal', 'cross'], help='sampling mode for train dataloader (default: normal)')
    # parser.add_argument('--test-batch-size', type=int, default=128, help='Batch size for test split (default: 128)')
    # parser.add_argument('--valid-batch-size', type=int, default=128, help='Batch size for valid split (default: 128)')
    parser.add_argument('--neg_ent', type=int, default=32, help='neg ent num for each triple (default: 32)')
    parser.add_argument('--neg_rel', type=int, default=0, help='neg rel num for each triple (default: 0)')
    

    parser.add_argument('--t_margin', type=float, default=0.0, help='Margin value for scoring function of Translating models. Default:0.0')
    parser.add_argument('--t_norm', type=int, default=2, help='P-norm value for scoring function of Translating models. Default:2')
    
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--lr-decay', type=float, default=0.0, help='Weight decay value to use in the optimizer. Default: 0.0')
    parser.add_argument('--embedding-dim', type=int, default=200, help='The embedding dimension (1D). Default: 200')
    parser.add_argument('--save-influence-map', action='store_true', help='Save the influence map during training for gradient rollback.')
    parser.add_argument('--resume', action='store_true', help='Restore a saved model.')
    
    
    parser.add_argument('--max-norm', action='store_true', help='Option to add unit max norm constraint to entity embeddings')
    
    # parser.add_argument('--save-influence-map', action='store_true', help='Save the influence map during training for gradient rollback.')
    
    parser.add_argument('--stack_width', type=int, default=20, help='The first dimension of the reshaped/stacked 2D embedding. Second dimension is inferred. Default: 20')
    #parser.add_argument('--stack_height', type=int, default=10, help='The second dimension of the reshaped/stacked 2D embedding. Default: 10')
    parser.add_argument('--hidden-drop', type=float, default=0.3, help='Dropout for the hidden layer. Default: 0.3.')
    parser.add_argument('--input-drop', type=float, default=0.2, help='Dropout for the input embeddings. Default: 0.2.')
    parser.add_argument('--feat-drop', type=float, default=0.3, help='Dropout for the convolutional features. Default: 0.2.')
    parser.add_argument('-num-filters', default=32,   type=int, help='Number of filters for convolution')
    parser.add_argument('-kernel-size', default=3, type=int, help='Kernel Size for convolution')
    
    parser.add_argument('--use-bias', action='store_true', help='Use a bias in the convolutional layer. Default: True')
    
    parser.add_argument('--reg-weight', type=float, default=5e-2, help='Weight for regularization. Default: 5e-2')#maybe 5e-2?
    parser.add_argument('--reg-norm', type=int, default=2, help='Norm for regularization. Default: 3')
    
    parser.add_argument('--resume-split', type=str, default='test', help='Split to evaluate a restored model')
    parser.add_argument('--seed', type=int, default=17, metavar='S', help='Random seed (default: 17)')
    
    parser.add_argument('--reproduce-results', action='store_true', help='Use the hyperparameters to reproduce the results.')
    parser.add_argument('--original-data', type=str, default='FB15k-237', help='Dataset to use; this option is needed to set the hyperparams to reproduce the results for training after attack, default: FB15k-237')
    
    return parser.parse_args(args)

