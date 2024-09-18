import torch
import time
import numpy as np

from torch.utils.data import DataLoader
import torch.autograd as autograd

import logging

import pdb

def hv(loss, model_params, v):
    grad = autograd.grad(loss, model_params, create_graph=True, retain_graph=True)
    Hv = autograd.grad(grad, model_params, grad_outputs=v)
    return Hv


def gather_flat_grad(grads):
    views = []
    for p in grads:
        if p.data.is_sparse:
            view = p.data.to_dense().view(-1)
        else:
            view = p.data.view(-1)
        views.append(view)
    return torch.cat(views, 0)

# original scale : 1e4
def get_inverse_hvp_lissa(test_grads, model, param_influence, train_dataset, args):

    ihvp = None
    cur_estimate = test_grads
    scale = args.scale

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size)
    train_data_iterator = iter(train_dataloader)

    # lissa_depth - for stochastic estimation of ihvp
    # basically, how many samples you like to use from train dataset
    recursion_depth = int(len(train_dataloader) * args.lissa_depth)

    logging.info("Recursion depth for this dataset : %d" %recursion_depth)

    start_time = time.time()
    for depth in range(recursion_depth):
        try:
            sample = next(train_data_iterator)
        except StopIteration:
            train_data_iterator = iter(train_dataloader)
            sample = next(train_data_iterator)

        batch = tuple(s.to(args.device) for s in sample.values())

        model.zero_grad()
        train_loss = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2])[0]

        hvp = hv(train_loss, param_influence, cur_estimate)

        # pdb.set_trace()

        cur_estimate = [_a + (1 - args.damping) * _b - _c / scale for _a, _b, _c in zip(test_grads, cur_estimate, hvp)]

        if (depth % 50 == 0) or (depth == recursion_depth - 1):
            logging.info("Recursion at depth %s: norm is %f" % (depth, np.linalg.norm(gather_flat_grad(cur_estimate).cpu().numpy())))


    if ihvp == None:
        ihvp = [_a / scale for _a in cur_estimate]

    else:
        ihvp = [_a + _b / scale for _a, _b in zip(ihvp, cur_estimate)]

    # from influence-function-analysis
    # ihvp is the collection of gradients from all targeted parameters

    final_ihvp = gather_flat_grad(ihvp)

    end_time = time.time()
    total_time = (end_time - start_time) / 60

    logging.info("Took %.2f min", total_time)

    return final_ihvp