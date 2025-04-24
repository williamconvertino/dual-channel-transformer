import util.cache # Initializes cache in the data directory, to avoid home directory issues on cloud environments

from argparse import ArgumentParser
from util.trainer import Trainer
from util.evaluator import Evaluator
from dataset.tokenizer import Tokenizer
from dataset.tinystories_dataset import TinyStoriesDataset
from models.language_model import LanguageModel
from util.loading import load_most_recent_checkpoint, load_config

def main():
    parser = ArgumentParser()
    parser.add_argument("--train", type=str)
    parser.add_argument("--eval", type=str, nargs="+")
    args = parser.parse_args()

    assert args.train or args.eval, "Must specify either training, evaluation, or dictionary learning"
    
    if args.train:
        model_name = args.train
    elif args.eval:
        model_name = args.eval[0]
    
    config = load_config(model_name)
    
    tokenizer = Tokenizer()
    config.vocab_size = tokenizer.vocab_size
    
    model = LanguageModel(config)
    checkpoint = load_most_recent_checkpoint(model)
    
    splits = TinyStoriesDataset.get_splits(tokenizer, config.max_seq_len)
    
    if args.train:
        trainer = Trainer(model, splits, tokenizer, checkpoint)
        trainer.train()
    elif args.eval:
        assert checkpoint is not None, "No checkpoint found for model, cannot evaluate"
        model.load_state_dict(checkpoint["model_state_dict"])
        
        eval_flags = args.eval[1:] if len(args.eval) > 1 else ["loss", "nucleus"]
        evaluator = Evaluator(model, splits, tokenizer)
        if "loss" in eval_flags:
            evaluator.eval_loss()
        if any(flag in ["greedy", "beam", "topk", "nucleus"] for flag in eval_flags):
            evaluator.eval(eval_flags)

if __name__ == "__main__":
    main()