import Module.Training as Training

import torch
torch.random.manual_seed(1234)

# Serially run all tasks, comment out if you don't need any of them.

task_1 = Training.model('Laplace', 'EXP')
task_1.train()

task_2 = Training.model('Poisson', 'EXP')
task_2.train()

