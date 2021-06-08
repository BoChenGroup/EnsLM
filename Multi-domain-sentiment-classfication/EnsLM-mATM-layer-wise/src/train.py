import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
# from cnn_utils.optim_Noam import NoamOpt
import time


def train(train_iter, dev_iter, model, args, logger):
    if args.cuda:
        model.to(args.device)

    if args.optimizer == 'adam':
       optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
       # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=args.momentum)

    elif args.optimizer == 'adamw':
       optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09, weight_decay=args.weight_decay,
                                   amsgrad=True)
       schedular = None
       schedular = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.warnup)
       ow_factor = 2
       ow_warmup = 20000
       ow_model_size = 300
       if args.optimizer_warper and schedular is not None:
           optimizer = NoamOpt(ow_model_size, ow_factor, ow_warmup, optimizer)


    steps = 1
    best_acc = 0
    loss_mean = 0
    corrects = 0

    for epoch in range(1, args.epochs+1):
        for i, (feature, target, task) in enumerate(train_iter):
            model.train()
            if args.cuda:
                feature, target, task = feature.to(args.device).long(), target.to(args.device).long(),  task.to(args.device).long()

            if args.sent_loss:
                sent_target = target.repeat(feature.size(1))
                sent_logit, doc_logit = model(feature)
                doc_loss = F.cross_entropy(doc_logit)
                sent_loss = F.cross_entropy(sent_logit, sent_target)
                loss = doc_loss + sent_loss
                loss = loss / args.accumulation_steps
                loss.backward()
            else:
                logit = model(feature, task)  # N*class_num
                loss = F.cross_entropy(logit, target)
                loss = loss / args.accumulation_steps
                loss.backward()

            loss_mean = loss_mean + loss.item()
            corrects = corrects + (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()/args.batchsize

            if ((i + 1) % args.accumulation_steps) == 0:
                # optimizer the net
                # update parameters of net
                if args.optimizer_warper:
                    optimizer.optimizer.step()
                    optimizer.optimizer.zero_grad()  # for warper
                else:
                    optimizer.step()
                    optimizer.zero_grad()  # origin

                loss_mean = 0
                corrects = 0
                steps += 1

        dev_avg_loss, dev_acc, dev_corrects, dev_size = eval(dev_iter, model, args)
        if dev_acc > best_acc:
            best_acc = dev_acc
        logger.info('Evaluation  - epoch {} loss: {:.6f}  acc: {:.4f}%({}/{}) best: {:.4f}'.format(epoch,
                                                                       dev_avg_loss,
                                                                       dev_acc,
                                                                       dev_corrects,
                                                                       dev_size,
                                                                       best_acc))


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    with torch.no_grad():
        for i, (feature, target, task) in enumerate(data_iter):
            if args.cuda:
                feature, target, task = feature.to(args.device).long(), target.to(args.device).long(), task.to(args.device).long()

            if args.sent_loss:
                _, logit = model(feature, task)
            else:
                logit = model(feature, task)
            loss = F.cross_entropy(logit, target, size_average=False)

            avg_loss += loss.item()
            corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * float(corrects)/float(size)

    return avg_loss, accuracy, corrects, size


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.item()+1]


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps.pt'.format(save_prefix)
    torch.save(model.state_dict(), save_path)
