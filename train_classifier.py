import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import NN


def argmax(predict, labels):
    pred = predict.data.max(1)[1]
    return float(pred.eq(labels.data.view_as(pred)).sum()) / float(pred.shape[0])


def count_success(predict, labels):
    pred = [predict > 0.5] * 1
    success = np.floor((pred+labels)/2)
    return float(success.sum()/float(predict.shape[0]))


def train(train_dataset, valid_dataset, batch_size=128, num_of_layers=5, epochs=100, lr=1e-3, output_func='softmax',
          input_size=39*9, output_size=39, hidden_cells=1000, out_folder='logs', activation_function='relu'):
    # Load dataset
    print('Loading dataset ...\n')

    loader_train = DataLoader(dataset=train_dataset, num_workers=4, batch_size=batch_size, shuffle=True)
    loader_valid = DataLoader(dataset=valid_dataset, num_workers=1, batch_size=1, shuffle=False)
    print("# of training samples: %d\n" % int(len(train_dataset)))

    # Build model
    net = NN(num_of_layers=num_of_layers, output_func=output_func, input_size=input_size,
             output_size=output_size, hidden_cells=hidden_cells, activation_function=activation_function)

    accuracy_functions={
        'argmax': argmax,
        'count_success': count_success
    }
    try:
        if output_func.lower() == 'softmax':
            criterion = nn.CrossEntropyLoss()
            acurracy_func = accuracy_functions['argmax']
        elif output_func.lower() == 'sigmoid':
            criterion = nn.MSELoss()
            acurracy_func = accuracy_functions['count_success']
        else:
            raise NameError('Activation not supported')
    except NameError:
        print(output_func, 'not support, Please change')

    # Move to GPU
    device_ids = [0]
    if torch.cuda.is_available():
        model = nn.DataParallel(net, device_ids=device_ids).cuda()
        criterion.cuda()
    else:
        model = nn.DataParallel(net, device_ids=device_ids)
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # training
    writer = SummaryWriter(log_dir=out_folder)

    for epoch in range(epochs):  # loop over the dataset multiple times

        # _____________Train:______________________
        running_loss = []
        running_accuracy = []
        for i, (inputs, labels) in enumerate(loader_train, 0):
            # wrap them in Variable
            if torch.cuda.is_available():
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            # loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()

            # for loss per epoch
            running_loss.append(loss.item())
            # for accuracy per epoch
            running_accuracy.append(acurracy_func(outputs, labels))  # TODO: find a way to calc accuracy
        # print statistics
        print("epoch " + str(epoch) + ", Train: loss = " + str(np.mean(running_loss)) +
              ", accuracy = " + str(np.mean(running_accuracy)))
        writer.add_scalar('Train/Loss', float(np.mean(running_loss)), epoch)
        writer.add_scalar('Train/accuracy', float(np.mean(running_accuracy)), epoch)

        # ____________Validation:____________________
        net.eval()  # changing to eval mode
        valid_running_accuracy = []

        for k, (valid_inputs, valid_labels) in enumerate(loader_valid, 0):
            # wrap them in Variable
            if torch.cuda.is_available():
                valid_inputs, valid_labels = Variable(valid_inputs.cuda()), \
                                             Variable(valid_labels.cuda())
            else:
                valid_inputs, valid_labels = Variable(valid_inputs), \
                                             Variable(valid_labels)

            valid_outputs = net(valid_inputs)

            valid_running_accuracy.append(acurracy_func(outputs, labels)) # TODO: find a way to calc accuracy

        print("epoch " + str(epoch) + ", Validation: accuracy = " + str(np.mean(valid_running_accuracy)))
        writer.add_scalar('Validation/accuracy', float(np.mean(valid_running_accuracy)), epoch)
        net.train()