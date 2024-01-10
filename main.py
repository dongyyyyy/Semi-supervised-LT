from train.ABC_contrastive.train import train_fixmatch_withABC_contrastive
from train.ABC_contrastive.labeled_only.train import train_fixmatch_withABC_contrastive_labeled
from train.ABC_contrastive.unlabeled_only.train import train_fixmatch_withABC_contrastive_unlabeled

from train.Vanilla_semi.train_fixmatch import train_fixmatch
from train.ABC.train_fixmatch_withABC import train_fixmatch_ABC

if __name__ == '__main__':
    args = args_setting()
    if args.training_method == 'fixmatch':
        train_fixmatch(args)
    elif args.training_method == 'fixmatchABC':
        train_fixmatch_ABC(args)
    elif args.training_method == 'train_fixmatch_withABC_contrastive':
        train_fixmatch_withABC_contrastive(args)
    elif args.training_method == 'train_fixmatch_withABC_contrastive_labeled':
        train_fixmatch_withABC_contrastive_labeled(args)
    elif args.training_method == 'train_fixmatch_withABC_contrastive_unlabeled':
        train_fixmatch_withABC_contrastive_unlabeled(args)


        