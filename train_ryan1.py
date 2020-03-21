import sys
import os.path
import argparse
import math
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import config
import data
import model
# import model_original
# import model_kor2vec
import utils
import numpy as np
from datetime import datetime


def run(net, loader, optimizer, scheduler, tracker, train=False, has_answers=True, prefix='', epoch=0):
    """ Run an epoch over the given loader """
    assert not (train and not has_answers)
    if train:
        net.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        net.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}
        answ = []
        all_answer = []
        idxs = []
        accs = []

    loader = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))

    for v, q, a, b, idx, q_len in loader:
        var_params = {
            'volatile': not train,
            'requires_grad': False,
        }
        # torch.from_numpy([0]).unsqueeze(1), torch.from_numpy([0])
        # temp = torch.from_numpy(np.arange(10, dtype=int).reshape(5, 2))
        # if b == temp:
        #     continuext

        v = Variable(v.cuda(async=True), **var_params)
        q = Variable(q.cuda(async=True), **var_params)
        a = Variable(a.cuda(async=True), **var_params)
        b = Variable(b.cuda(async=True), **var_params)
        q_len = Variable(q_len.cuda(async=True), **var_params)

        out = net(v, b, q, q_len)
        if has_answers:
            nll = -F.log_softmax(out, dim=1)
            loss = (nll * a / 10).sum(dim=1).mean()
            acc = utils.batch_accuracy(out.data, a.data).cpu()

        if train:
            # optimizer가 lr보다 먼저와야
            scheduler.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            # store information about evaluation of this minibatch
            all = out.data.cpu()
            for a in all:
                all_answer.append(a.view(-1))

            # print("max",out.data.cpu().max(dim=1))

            _, answer = out.data.cpu().max(dim=1)  # ???
            answ.append(answer.view(-1))
            if has_answers:
                accs.append(acc.view(-1))
            idxs.append(idx.view(-1).clone())

        if has_answers:
            loss_tracker.append(loss.item())
            acc_tracker.append(acc.mean())
            fmt = '{:.4f}'.format
            loader.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))

    if not train:
        answ = list(torch.cat(answ, dim=0))
        if has_answers:
            accs = list(torch.cat(accs, dim=0))
        else:
            accs = []
        idxs = list(torch.cat(idxs, dim=0))
        return answ, accs, idxs, all_answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs='*')
    parser.add_argument('--eval', dest='eval_only', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--resume', nargs='*')
    # parser.add_argument('--image_url', type = str, default='')
    # parser.add_argument('--question_id', type = int)
    # parser.add_argument('--embedding', default='kor2vec', type='str', help='embedding method')
    args = parser.parse_args()

    if args.test:
        args.eval_only = True

    # if args.image_

    # if args.embedding == 'kor2vec':
    #     src = open('model_kor2vec.py').read()
    # if args.embedding == '':
    # src = open('model_original.py').read()
    src = open('model.py').read()
    if args.name:
        name = ' '.join(args.name)
    else:
        from datetime import datetime
        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    target_name = os.path.join('logs', '{}.pth'.format(name))
    # target_name = '/mnt/crawl/counting_logs/{}.pth'.format(name)
    if not args.test:
        # target_name won't be used in test mode
        print('will save to {}'.format(target_name))
    if args.resume:
        logs = torch.load(' '.join(args.resume))

        # hacky way to tell the VQA classes that they should use the vocab without passing more params around
        data.preloaded_vocab = logs['vocab']

    cudnn.benchmark = True

    print('start loader...')
    if not args.eval_only:
        train_loader = data.get_loader(train=True)
    if not args.test:
        val_loader = data.get_loader(val=True)
    else:
        val_loader = data.get_loader(test=True)

    print('start net / optimizer / scheduler...')



    # if you want to use 2 gpus
    # net = nn.DataParallel(model_original.Net(val_loader.dataset.num_tokens)).cuda()
    net = nn.DataParallel(model.Net(val_loader.dataset.num_tokens)).cuda()

    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad], lr=config.initial_lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.5 ** (1 / config.lr_halflife))

    # epoch = config.epochs

    if args.resume:
        net.load_state_dict(logs['weights'])
        # checkpoint = torch.load(logs)
        # print("cp", checkpoint)
        # model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(logs['optimizer_state_dict'])
        # epoch = logs['epoch']
        # loss = logs['loss']
        # model.train()

    tracker = utils.Tracker()
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}

    # if not args.test and not args.eval:

    print('start training...')

    for i in range(config.epochs):
        if not args.eval_only:
            print('train.. not args.eval_only')
            print('start train!!!!...')
            run(net, train_loader, optimizer, scheduler, tracker, train=True, prefix='train', epoch=i)

        print('start val!!!!...')
        r = run(net, val_loader, optimizer, scheduler, tracker, train=False, prefix='val', epoch=i,
                has_answers=not args.test)

        # print("r3", r[3])

        if not args.test:
            print('train.. if not args.test')
            if i % 5 == 0:
                now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                save_name = '/mnt/backup/ran/vqa/vqa/save/ko100_{0}_{1}.pth'.format(now, i)  # 20191028기준 lr1이 메인
                # target_name = os.path.join('logs', '{}.pth'.format(name))
                results = {
                    'name': name,
                    'tracker': tracker.to_dict(),
                    'config': config_as_dict,
                    'weights': net.state_dict(),
                    'eval': {
                        'answers': r[0],
                        'accuracies': r[1],
                        'idx': r[2],
                    },

                    'vocab': val_loader.dataset.vocab,
                    'src': src,
                    'epoch': i,
                    # 'model_state_dict': net.state_dict(), #weight
                    'optimizer_state_dict': optimizer.state_dict(),  # optimizer weight
                    # 'loss': loss,
                }

                torch.save(results, save_name)

            if i == 99:
                now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                save_name = '/mnt/crawl/counting_logs/eng_{0}_final.pth'.format(now)
                # target_name = os.path.join('logs', '{}.pth'.format(name))
                results = {
                    'name': name,
                    'tracker': tracker.to_dict(),
                    'config': config_as_dict,
                    'weights': net.state_dict(),
                    'eval': {
                        'answers': r[0],
                        'accuracies': r[1],
                        'idx': r[2],
                    },
                    'vocab': val_loader.dataset.vocab,
                    'src': src,

                }
                torch.save(results, save_name)
        else:
            # in test mode, save a results file in the format accepted by the submission server
            print('train.. else -> yes args.test')
            # torch.load("")
            answer_index_to_string = {a: s for s, a in val_loader.dataset.answer_to_index.items()}
            # with open('answer_index_to_string.txt', 'w') as f:
            #    json.dump(answer_index_to_string, f)
            results = []

            for answer, index in zip(r[0], r[2]):  # r3
                try:
                    answer = answer_index_to_string[answer.item()]
                except KeyError:
                    continue
                # answer = answer_index_to_string[answer.item()]

                qid = val_loader.dataset.question_ids[index]
                entry = {
                    'question_id': qid,
                    'answer': answer
                    # 'all_answer': rr
                }
                results.append(entry)
            with open('results_test.json', 'w') as fd:
                json.dump(results, fd)

        if args.eval_only:
            # else:
            # in test mode, save a results file in the format accepted by the submission server
            print('eval+pnp')
            # torch.load("")
            answer_index_to_string = {a: s for s, a in val_loader.dataset.answer_to_index.items()}
            # with open('answer_index_to_string.txt', 'w') as f:
            #    json.dump(answer_index_to_string, fd)
            results = []

            for answer, index in zip(r[0], r[2]):
                try:
                    answer = answer_index_to_string[answer.item()]
                except KeyError:
                    continue

                qid = val_loader.dataset.question_ids[index]
                entry = {
                    'question_id': qid,
                    'answer': answer,
                    # 'all_answer':rr
                }
                results.append(entry)
            with open('results_test.json', 'w') as fd:
                json.dump(results, fd)

            break


if __name__ == '__main__':
    main()
