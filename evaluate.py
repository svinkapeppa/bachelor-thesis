import torch

from conlleval import evaluate


def evaluate_model(model, batches, idx_tag):
    true_seqs, pred_seqs = [], []

    model.eval()
    with torch.no_grad():
        for batch in batches:
            target = batch['tag']
            logits = model(batch).argmax(-1)

            for seq_ind, seq_len in enumerate(batch['lengths']):
                true_seqs.append(' '.join([idx_tag[ind.item()] for ind in target[seq_ind, :seq_len]]))
                pred_seqs.append(' '.join([idx_tag[ind.item()] for ind in logits[seq_ind, :seq_len]]))

    return evaluate(true_seqs, pred_seqs, verbose=False)
