import torch

pickle_dir = 'results/experiment.pickle'

def show_metrics():
    d = torch.load(pickle_dir)
    print('==========================')
    
    print('Final Test Eval Accuracy')
    print(d['final_eval_test_accu'])
    print('==========================')
    
    print('Final Test Eval Loss')
    print(d['final_eval_test_loss'])
    print('==========================')
    
    print('Meta Genotype')
    print(d['meta_genotype'])
    print('==========================')
    
    print('Train Test Loss')
    print(d['train_test_loss'])
    print('==========================')
    
    print('Train Test Accuracy')
    print(d['train_test_accu'])
    print('==========================')
    
    print('Test Test Loss')
    print(d['test_test_loss'])
    print('==========================')
    
    print('Test Test Accuracy')
    print(d['test_test_accu'])
    print('==========================')
   
    print('Num Params')
    print(d['sparse_params_logger'])
    print('==========================')
    
if __name__ == "__main__":
    show_metrics()
