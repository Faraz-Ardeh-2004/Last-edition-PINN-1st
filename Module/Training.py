# coding = utf-8
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
import os
import importlib
import time
import Module.PINN as PINN
import Module.SingleVis as SingleVis

torch.manual_seed(1234)  # Set random seed

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU is available")
else:
    device = torch.device('cpu')


class model():
    # language: python
    def __init__(self, ques_name, ini_num):
        # Initialize using file
        self.ques_name = ques_name
        self.ini_num = ini_num
        self.ini_file_path = f'./Config/{ques_name}_{ini_num}.csv'

        # Parse CSV file with Names,Values columns
        self.model_ini_dict = {}

        try:
            # Read CSV, using only first 2 columns to avoid issues with variable column counts
            df = pd.read_csv(self.ini_file_path, usecols=[0, 1], names=['key', 'value'],
                             header=0, skip_blank_lines=True)

            # Process each row
            for index, row in df.iterrows():
                key = str(row['key']).strip()
                value = str(row['value']).strip()

                # Skip empty or invalid rows
                if pd.isna(row['key']) or key == '' or key == 'nan':
                    continue

                # Skip the header row values if they appear in data
                if key.lower() in ['names', 'key']:
                    continue

                # Convert values based on key patterns
                if 'min' in key or 'max' in key:
                    try:
                        self.model_ini_dict[key] = float(value)
                    except:
                        pass  # Skip if can't convert
                elif 'num' in key or 'state' in key:
                    try:
                        self.model_ini_dict[key] = int(value)
                    except:
                        pass  # Skip if can't convert
                else:
                    # Try to preserve the value type
                    try:
                        if ',' in value:
                            # Keep as string for comma-separated values
                            self.model_ini_dict[key] = value
                        elif value.replace('.', '', 1).replace('-', '', 1).isdigit():
                            if '.' in value:
                                self.model_ini_dict[key] = float(value)
                            else:
                                self.model_ini_dict[key] = int(value)
                        else:
                            self.model_ini_dict[key] = value
                    except Exception:
                        self.model_ini_dict[key] = value

        except FileNotFoundError:
            raise
        except Exception as e:
            # if parsing fails, show available keys for debugging
            raise KeyError(
                f"Failed to parse `./Config/{ques_name}_{ini_num}.csv`. Error: {e}")

        # Handle input_num: if not present, derive from coord_num and para_ctrl_add
        if 'input_num' not in self.model_ini_dict:
            if 'coord_num' in self.model_ini_dict:
                coord_num = self.model_ini_dict['coord_num']
                para_ctrl_add = self.model_ini_dict.get('para_ctrl_add', 0)
                self.model_ini_dict['input_num'] = coord_num + para_ctrl_add
            else:
                raise KeyError("Neither 'input_num' nor 'coord_num' found in configuration file.")

        # Ensure required keys exist
        required = ['node_num', 'input_num', 'output_num', 'x_min', 'x_max', 'y_min', 'y_max']
        missing = [k for k in required if k not in self.model_ini_dict]
        if missing:
            raise KeyError(
                f"Missing required configuration keys: {missing}. Available keys: {list(self.model_ini_dict.keys())}")

        # Continue initialization (unchanged)
        self.node_num = self.model_ini_dict['node_num']
        self.input_num = self.model_ini_dict['input_num']
        self.output_num = self.model_ini_dict['output_num']

        # Problem-specific learning rate
        if 'Poisson' in ques_name:
            default_lr = 1e-3  # Poisson needs higher learning rate
        else:
            default_lr = 1e-4  # Laplace uses lower for smoothness

        self.learning_rate = self.model_ini_dict.get('learning_rate', default_lr)
        if isinstance(self.learning_rate, str):
            self.learning_rate = float(self.learning_rate)

        # Batch sampling (disabled for Poisson by default)
        if 'Poisson' in ques_name:
            self.batch_size = self.model_ini_dict.get('batch_size', 999999)  # Use all points
        else:
            self.batch_size = self.model_ini_dict.get('batch_size', 2000)

        if isinstance(self.batch_size, str):
            self.batch_size = int(self.batch_size)

        self.x_min = self.model_ini_dict['x_min']
        self.x_max = self.model_ini_dict['x_max']
        self.y_min = self.model_ini_dict['y_min']
        self.y_max = self.model_ini_dict['y_max']

        self.hidden_layers_group = list(map(float, self.model_ini_dict['hidden_layers_group'].split(',')))
        self.layer = [self.input_num, self.output_num]
        self.layer[1:1] = list(map(lambda x: x * self.node_num, self.hidden_layers_group))
        self.layer = list(map(int, self.layer))

        self.grid_node_num = self.model_ini_dict['grid_node_num']
        self.monitor_state = True if 'inv' in self.ques_name or 'global' in self.ques_name else False
        self.regular_state = self.model_ini_dict['regularization_state']
        self.bun_node_num = self.model_ini_dict['bun_node_num']
        self.figure_node_num = self.model_ini_dict['figure_node_num']

        # Read train_steps from config file, use default of 10000 if not found
        self.train_steps = self.model_ini_dict.get('train_steps', 10000)
        if isinstance(self.train_steps, str):
            self.train_steps = int(self.train_steps)

        self.save_desti = f'./Results/{self.ques_name}_{str(self.ini_num)}/'
        self.milestone = list(
            map(int, self.model_ini_dict['milestone'].split(','))) if 'milestone' in self.model_ini_dict else [50000,
                                                                                                               75000]
        self.gamma = float(self.model_ini_dict['gamma']) if 'gamma' in self.model_ini_dict else 0.5
        self.para_ctrl_num = 1 if self.monitor_state else 0

    # Define calculation field
    def mesh_init(self):
        x = torch.linspace(self.x_min, self.x_max, self.grid_node_num).float().to(device)
        y = torch.linspace(self.y_min, self.y_max, self.grid_node_num).float().to(device)
        x, y = torch.meshgrid(x, y, indexing='ij')
        self.x_full = x.reshape([-1, 1])
        self.y_full = y.reshape([-1, 1])
        self.total_points = self.x_full.shape[0]

    def sample_batch(self):
        """Sample random batch of points for training"""
        if self.batch_size >= self.total_points:
            # Use all points if batch size is larger
            self.x = self.x_full.clone().detach().requires_grad_(True)
            self.y = self.y_full.clone().detach().requires_grad_(True)
        else:
            # Random sampling
            indices = torch.randperm(self.total_points)[:self.batch_size]
            self.x = self.x_full[indices].clone().detach().requires_grad_(True)
            self.y = self.y_full[indices].clone().detach().requires_grad_(True)

    # Boundary condition loss
    def net_b(self):
        loss_b = torch.tensor(0.).to(device)

        self.bun_node_num = 1000

        # x minimum, y arbitrary
        y_b = torch.linspace(self.y_min, self.y_max, self.bun_node_num, requires_grad=True).float().to(device).reshape(
            [-1, 1])
        x_b = torch.full_like(y_b, self.x_min, requires_grad=True).float().to(device).reshape([-1, 1])
        u_b = self.net(torch.cat([x_b, y_b], dim=1))

        # y minimum, x arbitrary
        x_down = torch.linspace(self.x_min, self.x_max, self.bun_node_num, requires_grad=True).float().to(
            device).reshape([-1, 1])
        y_down = torch.full_like(x_down, self.y_min, requires_grad=True).float().to(device).reshape([-1, 1])
        u_down = self.net(torch.cat([x_down, y_down], dim=1))

        # y maximum, x arbitrary
        x_up = torch.linspace(self.x_min, self.x_max, self.bun_node_num, requires_grad=True).float().to(device).reshape(
            [-1, 1])
        y_up = torch.full_like(x_up, self.y_max, requires_grad=True).float().to(device).reshape([-1, 1])
        u_up = self.net(torch.cat([x_up, y_up], dim=1))

        # x maximum, y arbitrary
        y_f = torch.linspace(self.y_min, self.y_max, self.bun_node_num, requires_grad=True).float().to(device).reshape(
            [-1, 1])
        x_f = torch.full_like(y_f, self.x_max, requires_grad=True).float().to(device).reshape([-1, 1])
        u_f = self.net(torch.cat([x_f, y_f], dim=1))

        if 'Laplace' in self.ques_name:
            u_b_moni = (x_b ** 3 - 3 * x_b * y_b ** 2)  # Laplace boundary
            loss_b += torch.mean((u_b - u_b_moni) ** 2)

            u_down_moni = (x_down ** 3 - 3 * x_down * y_down ** 2)
            loss_b += torch.mean((u_down - u_down_moni) ** 2)

            u_up_moni = (x_up ** 3 - 3 * x_up * y_up ** 2)
            loss_b += torch.mean((u_up - u_up_moni) ** 2)

            u_f_moni = (x_f ** 3 - 3 * x_f * y_f ** 2)
            loss_b += torch.mean((u_f - u_f_moni) ** 2)

        elif 'Poisson' in self.ques_name:
            x_total = torch.cat([x_b, x_down, x_up, x_f], dim=0)
            y_total = torch.cat([y_b, y_down, y_up, y_f], dim=0)
            u_total = self.net(torch.cat([x_total, y_total], dim=1))
            loss_b += torch.mean((u_total) ** 2)

        return loss_b

    def net_f(self):
        loss_f = torch.tensor(0.).to(device)
        u = self.net(torch.cat([self.x, self.y], dim=1)).to(device)

        u_x = torch.autograd.grad(u, self.x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = \
        torch.autograd.grad(u_x, self.x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        u_y = torch.autograd.grad(u, self.y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_yy = \
        torch.autograd.grad(u_y, self.y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]

        if 'Laplace' in self.ques_name:
            if 'inv' in self.ques_name:
                loss_f = torch.mean((u_xx + self.para_undetermin[0] * u_yy) ** 2)
            else:
                loss_f = torch.mean((u_xx + u_yy) ** 2)

        elif 'Poisson' in self.ques_name:
            k = torch.arange(1, 5).to(device)
            f = sum([1 / 2 * ((-1) ** (k + 1)) * (k ** 2) * (
                        torch.sin(k * torch.pi * self.x) * torch.sin(k * torch.pi * self.y)) for k in k])

            if 'inv' in self.ques_name:
                loss_f = torch.mean((u_xx + self.para_undetermin[0] * u_yy - f) ** 2)
            else:
                loss_f = torch.mean((u_xx + u_yy - f) ** 2)

        return loss_f

    def net_rgl(self, reg_type='l2', weight_rgl=1e-3):
        loss_rgl = torch.tensor(0.).to(device)

        for name, param in self.net.named_parameters():
            if reg_type == 'l2':
                loss_rgl += weight_rgl * torch.norm(param, p=2)
            elif reg_type == 'l1':
                loss_rgl += weight_rgl * torch.norm(param, p=1)

        return loss_rgl

    # Supervised error for known full-field data
    def net_global(self):
        loss_global = torch.tensor(0.).to(device)
        u = self.net(torch.cat([self.x, self.y], dim=1)).to(device)

        if 'Laplace' in self.ques_name:
            loss_global += torch.mean((u - (self.x) ** 3 + 3 * self.x * self.y ** 2) ** 2)

        elif 'Poisson' in self.ques_name:
            u_moni = 0.5 / (2 * torch.pi ** 2) * (
                    (torch.sin(torch.pi * self.x) * torch.sin(torch.pi * self.y)) -
                    (2 * torch.sin(2 * torch.pi * self.x) * torch.sin(2 * torch.pi * self.y)) +
                    (3 * torch.sin(3 * torch.pi * self.x) * torch.sin(3 * torch.pi * self.y)) -
                    (4 * torch.sin(4 * torch.pi * self.x) * torch.sin(4 * torch.pi * self.y))
            )

            if 'lf' in self.ques_name:
                u_moni = torch.sin(torch.pi * self.x) * torch.sin(torch.pi * self.y) + \
                         torch.sin(2 * torch.pi * self.x) * torch.sin(2 * torch.pi * self.y)

            loss_global += torch.mean((u_moni - u) ** 2)

        return loss_global

    def train_adam(self):
        self.para_undetermin = torch.zeros(self.para_ctrl_num, requires_grad=True).float().to(device)
        self.para_undetermin = torch.nn.Parameter(self.para_undetermin)

        self.optimizer = optim.Adam(list(self.net.parameters()) + [self.para_undetermin], lr=self.learning_rate)

        # Problem-specific schedulers
        if 'Poisson' in self.ques_name:
            # Poisson: MultiStepLR for stable decay
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.milestone, gamma=self.gamma
            )
            use_plateau = False
        else:
            # Laplace: ReduceLROnPlateau for adaptive smoothing
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=1000, min_lr=1e-6
            )
            use_plateau = True

        self.current_time = time.time()
        self.time_list = [0.]

        # Moving average for smoother loss tracking (only for Laplace)
        loss_window = []
        window_size = 100

        for iter_step in range(self.train_steps):
            # Sample new batch each iteration (Poisson uses all points)
            self.sample_batch()

            self.optimizer.zero_grad()

            self.loss_f = self.net_f()

            if 'inv' in self.ques_name:
                self.loss_d = self.net_global()
            else:
                self.loss_d = torch.tensor(0.).to(device)

            self.loss_b = torch.tensor(0.).to(device) if self.monitor_state else self.net_b()
            self.loss_rgl = self.net_rgl(reg_type='l2') if self.regular_state else torch.tensor(0.).to(device)

            if self.monitor_state:
                self.loss = self.loss_d + self.loss_f
            else:
                # Adaptive loss weighting for Poisson
                if 'Poisson' in self.ques_name:
                    # Increase PDE weight progressively for Poisson
                    pde_weight = 1.0 + min(9.0, iter_step / (self.train_steps * 0.1))
                    boundary_weight = max(1.0, 10.0 - iter_step / (self.train_steps * 0.05))
                    self.loss = pde_weight * self.loss_f + boundary_weight * self.loss_b
                else:
                    self.loss = self.loss_f + self.loss_b

            if self.regular_state:
                self.loss += self.loss_rgl

            self.loss.backward(retain_graph=True)

            # Gradient clipping (not for Poisson - needs full gradients)
            if 'Poisson' not in self.ques_name:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update scheduler
            if use_plateau:
                loss_window.append(self.loss.item())
                if len(loss_window) > window_size:
                    loss_window.pop(0)
                avg_loss = sum(loss_window) / len(loss_window)
                self.scheduler.step(avg_loss)
            else:
                self.scheduler.step()

            self.net.iter += 1
            self.net.iter_list.append(self.net.iter)
            self.net.loss_list.append(self.loss.item())
            self.net.loss_f_list.append(self.loss_f.item())
            self.net.loss_b_list.append(self.loss_b.item())
            self.net.loss_d_list.append(self.loss_d.item())
            self.net.loss_rgl_list.append(self.loss_rgl.item())

            if self.monitor_state:
                self.net.para_ud_list.append(self.para_undetermin.tolist())

            if self.net.iter % 100 == 0:
                loss_dict = {
                    'Iter': self.net.iter,
                    'Loss': self.loss.item(),
                    'Loss_f': self.loss_f.item(),
                    'Loss_b': self.loss_b.item(),
                    'Loss_d': self.loss_d.item(),
                    'Loss_rgl': self.loss_rgl.item()
                }

                loss_str = ', '.join([f'{key}: {int(value) if key == "Iter" else value:.5e}'
                                      for key, value in loss_dict.items() if key != "Iter" and value != 0])
                iter_str = f'Iter: {{{self.net.iter}/{self.train_steps}}}'
                print(f'{iter_str}, {loss_str}')

            self.time_list[0] += time.time() - self.current_time
            self.current_time = time.time()

        print(f'\nTime occupied: {(self.time_list[0]):.5e} s.\n')

    def model_save(self):
        if not os.path.exists(f'./Results/'):
            os.mkdir(f'./Results/')

        if not os.path.exists(self.save_desti):
            os.mkdir(self.save_desti)
        if not os.path.exists(f'{self.save_desti}/Models/'):
            os.mkdir(f'{self.save_desti}/Models/')

        torch.save(self.net.state_dict(), f"{self.save_desti}/Models/{self.ques_name}_{self.ini_num}_PINN.pth")

        # Copy control parameters
        self.control_paras = pd.read_csv(self.ini_file_path)
        self.control_paras.to_csv(f'{self.save_desti}{self.ques_name}_{self.ini_num}.csv', index=False)

        # Store time
        self.time_save = pd.DataFrame({
            'Question': [self.ques_name],
            'Number': [self.ini_num],
            'Module': ['PINN'],
            'Training Time': [self.time_list[0]]
        })
        self.time_save.to_csv(self.save_desti + 'Clock time.csv', mode='w', index=False)

        # Store loss data
        loss_data_dict = {
            'iter': self.net.iter_list,
            'loss': self.net.loss_list,
            'loss_f': self.net.loss_f_list,
            'loss_b': self.net.loss_b_list,
            'loss_d': self.net.loss_d_list,
            'loss_rgl': self.net.loss_rgl_list
        }

        df_loss_data = pd.DataFrame(loss_data_dict)
        df_loss_data = df_loss_data.loc[:, (df_loss_data != 0).any(axis=0)]

        if not os.path.exists(self.save_desti + '/Loss/'):
            os.mkdir(self.save_desti + '/Loss/')

        df_loss_data.to_csv(f"{self.save_desti}/Loss/{self.ques_name}_{str(self.ini_num)}_loss_PINN.csv", index=False)

        # Store calculated parameters for inverse problems
        if self.monitor_state:
            iter_list = np.array(self.net.iter_list).reshape([-1, 1])
            para_ud = np.array(np.hstack([iter_list, self.net.para_ud_list]))
            para_ud_columns = ['iter'] + [f'parameter_{i + 1}' for i in range(self.para_ctrl_num)]
            df_para_ud = pd.DataFrame(para_ud, columns=para_ud_columns)

            if not os.path.exists(self.save_desti + '/Parameters/'):
                os.mkdir(self.save_desti + '/Parameters/')
            df_para_ud.to_csv(f"{self.save_desti}/Parameters/{self.ques_name}_{str(self.ini_num)}_paras_PINN.csv",
                              index=False)

    def compute_ground_truth(self, x, y):
        """Compute analytical solution for ground truth"""
        if 'Laplace' in self.ques_name:
            return (x ** 3 - 3 * x * y ** 2).detach().cpu().numpy()

        elif 'Poisson' in self.ques_name:
            x_np = x.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()

            if 'lf' in self.ques_name:
                u_true = np.sin(np.pi * x_np) * np.sin(np.pi * y_np) + \
                         np.sin(2 * np.pi * x_np) * np.sin(2 * np.pi * y_np)
            else:
                u_true = 0.5 / (2 * np.pi ** 2) * (
                        (np.sin(np.pi * x_np) * np.sin(np.pi * y_np)) -
                        (2 * np.sin(2 * np.pi * x_np) * np.sin(2 * np.pi * y_np)) +
                        (3 * np.sin(3 * np.pi * x_np) * np.sin(3 * np.pi * y_np)) -
                        (4 * np.sin(4 * np.pi * x_np) * np.sin(4 * np.pi * y_np))
                )
            return u_true

        return None

    def result_show(self):
        x = np.linspace(self.x_min, self.x_max, self.figure_node_num).reshape([-1, 1])
        y = np.linspace(self.y_min, self.y_max, self.figure_node_num).reshape([-1, 1])
        x, y = np.meshgrid(x, y)

        input_data = torch.tensor(np.concatenate([x.reshape([-1, 1]), y.reshape([-1, 1])], axis=1),
                                  dtype=torch.float32, requires_grad=True).float().to(device)
        u = self.net(input_data)

        # Compute ground truth
        x_torch = input_data[:, 0:1]
        y_torch = input_data[:, 1:2]
        u_true = self.compute_ground_truth(x_torch, y_torch)

        input_data = input_data.detach().cpu().numpy()
        u = u.detach().cpu().numpy()

        # Pass ground truth to visualization
        u_vis = SingleVis.Vis(self.ques_name, self.ini_num, self.save_desti, 'PINN',
                              input_data, u, u_true=u_true)

        # Generate comprehensive visualization
        u_vis.comprehensive_results_figure()
        u_vis.simple_comparison_figure()
        u_vis.loss_vis()

        if self.monitor_state:
            u_vis.para_vis()

    def workflow(self):
        self.mesh_init()
        self.train_adam()
        self.model_save()
        self.result_show()

    def train(self):
        self.original_lr = self.learning_rate

        # Use PINN model
        self.net = PINN.Net(self.layer).float().to(device)

        print(f'\nRunning PINN Model for {self.ques_name}\n')
        self.workflow()
