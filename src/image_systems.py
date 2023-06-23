import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb
import random
import copy
import math

from opacus.grad_sample import GradSampleModule
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from opacus.optimizers import DPOptimizer
from opacus.data_loader import DPDataLoader
from src.dp_optimizers import DiNNODPOptimizer
from src.utils.graphs import generate_graph, get_metropolis_weights, get_neighbor_weights
from src.models.conv_nn import ConvNet
from src.models.resnet import ResNet9
from src.models.dp_resnet import DPResNet9
import src.datasets.datasets as datasets


class CentralSystem():

    def __init__(self, config):
        self.config = config
        self.batch_size = self.config.optim_params.batch_size
        self.momentum = self.config.optim_params.momentum or 0
        self.num_iterations = self.config.num_iterations or 0
        self.iterations_per_validate = self.config.iterations_per_validate
        self.save_checkpoints_per_iter = self.config.save_checkpoints_per_iter or None

        # Define the base loss function
        if self.config.loss == "NLL":
            self.base_loss_function = torch.nn.NLLLoss()
            print("Using NLL Loss")
        elif self.config.loss == "CrossEntropy":
            self.base_loss_function = torch.nn.CrossEntropyLoss()
            print("Using CrossEntropy Loss")
        else:
            raise NameError("Unknown loss function")

        # Check for gpu and assign device
        if self.config.gpu_device != "cpu" and torch.cuda.is_available():
            # Ensure it's a string, even if from an older config
            gpu_device = "cuda:"  + str(self.config.gpu_device)
            self.device = torch.device(gpu_device)
            print(f"Device is set to ({gpu_device})")
        else:
            self.device = torch.device("cpu")
            print("Device is set to CPU")

        print(f"Loading dataset: {self.config.data_params.dataset}")
        self.train_dataset, self.val_dataset = datasets.get_image_datasets(self.config.data_params.dataset)
        train_labels = self.train_dataset.dataset.targets
        self.train_ordered_labels = np.array(train_labels)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        batch_size=self.batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=self.config.num_workers)
        self.train_iter = iter(self.train_loader)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset,
                                                      batch_size=self.batch_size,
                                                      shuffle=True,
                                                      pin_memory=True,
                                                      num_workers=self.config.num_workers)

        # Load the model
        model = self.create_model(self.config.model)
        self.base_model = model.to(self.device)
        print(self.base_model)

        # Set up learning rate
        self.lr = np.logspace(
            math.log(self.config.optim_params.learning_rate_start, 10),
            math.log(self.config.optim_params.learning_rate_finish, 10),
            self.num_iterations,
        )

        # Configure the optimizer
        if self.config.optim_params.optimizer == "adam":
            self.optim = torch.optim.Adam(self.base_model.parameters(), lr=self.lr[0])
        elif self.config.optim_params.optimizer == "sgd":
            self.optim = torch.optim.SGD(self.base_model.parameters(),
                                         lr=self.lr[0],
                                         momentum=self.config.optim_params.momentum or 0,
                                         weight_decay=self.config.optim_params.weight_decay or 0)
        else:
            raise NameError("Optimizer is not supported")

    def create_model(self, model_name):
        if model_name == "convnet":
            model = ConvNet(num_channels=self.train_dataset.NUM_CHANNELS,
                            num_classes=self.train_dataset.NUM_CLASSES)
            print("Set up ConvNet model")
        elif model_name == "resnet9":
            model = ResNet9(num_channels=self.train_dataset.NUM_CHANNELS,
                             num_classes=self.train_dataset.NUM_CLASSES)
        elif model_name == "dpresnet9":
            model = DPResNet9(num_channels=self.train_dataset.NUM_CHANNELS,
                             num_classes=self.train_dataset.NUM_CLASSES)
            print("Set up DP ResNet9 model (no batch norm)")
        else:
            raise NotImplementedError("The model is not implemented")
        return model

    def train(self):
        # Train over total iterations rather than epochs for consistency
        for iteration in range(self.num_iterations):
            self.training_step(iteration)
            if iteration % self.iterations_per_validate == 0:
                self.validate(iteration)
            # Save model checkpoints periodically
            if self.save_checkpoints_per_iter and iteration % self.save_checkpoints_per_iter == 0:
                torch.save(self.base_model, self.config.checkpoint_dir + f"iteration_{iteration}")
        # Validate after finishing training
        accuracy = self.validate(self.num_iterations)
        if self.save_checkpoints_per_iter:
            torch.save(self.base_model, self.config.checkpoint_dir + f"iteration_{self.num_iterations}")
        return accuracy

    def training_step(self, iteration):
        # Overwrite previous learning rate with current iteration value
        for g in self.optim.param_groups:
            g['lr'] = self.lr[iteration]

        # Train one iteration
        self.optim.zero_grad()
        train_loss = self.batch_loss()
        train_loss.backward()
        self.optim.step()
        wandb.log({'iteration': iteration, 'train_loss': train_loss.item()})

    def batch_loss(self):
        # Get a batch of data from the dataloader
        try: (_, data, label) = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            (_, data, label) = next(self.train_iter)
        data = data.to(self.device)
        label = label.to(self.device)
        output = self.base_model(data)
        loss = self.base_loss_function(output, label)
        return loss

    def validate(self, iteration):
        self.base_model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            # Compute prediction accuracy over batches
            for (_, data, label) in self.val_loader:
                data = data.to(self.device)
                label = label.to(self.device)
                output = self.base_model(data)
                val_loss += self.base_loss_function(output, label)
                preds = output.data.max(1, keepdim=True)[1]
                correct += preds.eq(label.data.view_as(preds)).sum()
        val_loss = val_loss.item() / len(self.val_dataset)
        accuracy = correct.item() / len(self.val_dataset)
        self.base_model.train() # return the model to train mode
        wandb.log({'iteration': iteration, 'val_loss': val_loss, 'val_acc': accuracy})
        print(f"Iteration {iteration} accuracy: {accuracy}")
        return accuracy


class CentralDPSystem(CentralSystem):

    def __init__(self, config):
        super().__init__(config)

        # Use Opacus wrapper to clip per sample gradients
        self.base_model = GradSampleModule(self.base_model, strict=True).to(self.device)

        # Configure the optimizer (using the GradSampleModule parameters)
        if self.config.optim_params.optimizer == "adam":
            self.optim = torch.optim.Adam(self.base_model.parameters(), lr=self.lr[0])
        elif self.config.optim_params.optimizer == "sgd":
            self.optim = torch.optim.SGD(self.base_model.parameters(),
                                         lr=self.lr[0],
                                         momentum=self.config.optim_params.momentum,
                                         weight_decay=self.config.optim_params.weight_decay)
        else:
            raise NameError("Optimizer is not supported")
        
        # Enable uniform/Poisson sampling from the dataset
        self.train_loader = DPDataLoader.from_data_loader(self.train_loader, distributed=False)
        self.train_iter = iter(self.train_loader)

        sample_rate = 1 / len(self.train_loader)
        expected_batch_size = int(len(self.train_loader.dataset) * sample_rate)
        print("Setting up a Central DP System using an RDP accountant")
        print(f"Using sample rate: {sample_rate} and expected batch size: {expected_batch_size}")
        self.epsilon = self.config.dp_params.epsilon
        self.delta = self.config.dp_params.delta
        self.max_grad_norm = self.config.dp_params.max_grad_norm
        if isinstance(self.config.dp_params.effective_batch_size, int):
            self.effective_batch_size = self.config.dp_params.effective_batch_size
        else:
            self.effective_batch_size = self.batch_size
        if self.config.dp_params.noise_multiplier is None:
            self.noise_multiplier = get_noise_multiplier(target_epsilon=self.epsilon,
                                                         target_delta=self.delta,
                                                         sample_rate=sample_rate,
                                                         steps=self.num_iterations,
                                                         accountant="rdp")
        else:
            self.noise_multiplier = self.config.dp_params.noise_multiplier
        self.dp_optim = DPOptimizer(optimizer=self.optim,
                                 noise_multiplier=self.noise_multiplier,
                                 max_grad_norm=self.max_grad_norm,
                                 expected_batch_size=self.effective_batch_size)
        self.accountant = RDPAccountant()
        self.dp_optim.attach_step_hook(self.accountant.get_optimizer_hook_fn(sample_rate=sample_rate))
        print(f"DP using initial batch std: {self.noise_multiplier * self.max_grad_norm}")
        
    def training_step(self, iteration):
        # Overwrite previous learning rate with current iteration value
        for g in self.dp_optim.param_groups:
            g['lr'] = self.lr[iteration]

        num_iterations = self.effective_batch_size // self.batch_size
        self.dp_optim.zero_grad() # reset summed_grad to 0
        train_loss = 0
        for i in range(num_iterations - 1):
            loss = self.batch_loss() # compute loss
            loss.backward() # populate gradients
            train_loss += loss
            self.dp_optim.clip_and_accumulate()
            self.reset_param_grads() # reset all params except for summed_grad
        loss = self.batch_loss() # compute loss
        loss.backward() # populate gradients
        train_loss += loss
        self.dp_optim.step() # add noise and update using the gradients
        noise_std = self.noise_multiplier * self.max_grad_norm
        wandb.log({'iteration': iteration, 'train_loss': train_loss.item(), 'batch_noise_std': noise_std})

    def reset_param_grads(self):
        for p in self.base_model.parameters():
            p.grad_sample = None
        self.optim.zero_grad(False)

    def validate(self, iteration):
        super().validate(iteration)
        epsilon_used = self.accountant.get_epsilon(self.delta)
        wandb.log({'iteration': iteration, 'epsilon': epsilon_used, 'delta': self.delta})
  

class DistributedSystem(CentralSystem):

    def __init__(self, config):
        super().__init__(config)
        print("Setting up a DistributedSystem")

        # Create communication graph
        self.graph = generate_graph(config.graph_params)
        self.num_nodes = config.graph_params.num_nodes

        # Save the graph and fiedler value
        fiedler = nx.linalg.algebraic_connectivity(self.graph, tol=1e-3, method="lanczos")
        adjacency_matrix = nx.adjacency_matrix(self.graph).toarray()
        print(f"Adjacency Matrix: {adjacency_matrix}")
        print(f"Fiedler value: {fiedler}")
        plt.clf()
        nx.draw(self.graph)
        wandb.log({"graph": wandb.Image(plt), "graph_fiedler": fiedler})

        # Split data among nodes
        self.train_subsets = self.split_data(self.config.data_params)

        # Set up datastructures for each node
        self.models = []
        self.train_loaders = []
        self.train_iters = []
        for i in range(self.num_nodes):
            model = copy.deepcopy(self.base_model)
            self.models.append(model.to(self.device))
            train_loader = torch.utils.data.DataLoader(
                self.train_subsets[i],
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=self.config.num_workers,
            )
            self.train_loaders.append(train_loader)
            self.train_iters.append(iter(train_loader))

    def split_data(self, config):
        # Split the data among nodes according to data split type
        print("Splitting the data among nodes")
        num_classes = self.train_dataset.NUM_CLASSES
        t = config.data_split_value
        frac_data_matrix = self.get_data_matrix(t, self.num_nodes, num_classes)
        if num_classes % self.num_nodes != 0:
            raise RuntimeError(f"num_nodes {self.num_nodes} does not evenly divide num_classes {num_classes}")
        
        if t == 1: # distribute data by class to ensure no unwanted overlap (roughly equivalent to t = 1)
            classes = torch.tensor([i for i in range(self.train_dataset.NUM_CLASSES)])
            train_subsets = []
            node_classes = list(torch.split(classes, len(classes) // self.num_nodes))
            random.shuffle(node_classes) # shuffle in place
            for i in range(self.num_nodes):
                locs = [torch.tensor(int(lab) == self.train_ordered_labels) for lab in node_classes[i]]
                idx_keep = torch.nonzero(torch.stack(locs).sum(0)).reshape(-1)
                train_subsets.append(torch.utils.data.Subset(self.train_dataset, idx_keep))
            self.print_class_information(train_subsets, frac_data_matrix)
            return train_subsets

        used = []
        idx_by_class = []
        for i in range(num_classes): # aggregate indexes for every sample by class
            idx_by_class.append([])
            used.append(0)
        for i in range(len(self.train_ordered_labels)):
            class_ = self.train_ordered_labels[i]
            idx_by_class[class_].append(i)

        train_subset_idxs = []
        for i in range(self.num_nodes): # find training sample indexes for each agent
            idxs = []
            for j in range(num_classes):
                if i == self.num_nodes - 1:
                    idxs += idx_by_class[j][used[j]: len(idx_by_class[j])] # give the remaining amount to the last agent
                else:
                    num_elems = int(frac_data_matrix[i, j] * len(idx_by_class[j])) # number of class j samples agent i takes
                    idxs += idx_by_class[j][used[j]: used[j] + num_elems]
                    used[j] += num_elems
            train_subset_idxs.append(np.array(idxs))
        
        train_subsets = []
        for i in range(self.num_nodes): # select samples in a train subset 
            train_subsets.append(torch.utils.data.Subset(self.train_dataset, torch.tensor(train_subset_idxs[i])))
        self.print_class_information(train_subsets, frac_data_matrix)
        return train_subsets

    def get_data_matrix(self, t, num_nodes, num_classes):
        A = np.zeros((num_nodes, num_classes))
        N, M = A.shape
        # Update off-diagonal terms
        for i in range(N):
            for j in range(M):
                if i == j % N:
                    continue
                A[i, j] = (1 - t) / N
        # Update diagonal terms next
        for i in range(N):
            for j in range(M):
                if i != j % N:
                    continue
                A[i, j] = 1 - (np.sum(A[:i, j]) + np.sum(A[i + 1:, j]))
        return A

    def print_class_information(self, train_subsets, data_matrix):
        # Map length of local dataset to labels for user information
        print("-------------------")
        print("Data distribution matrix:")
        print(data_matrix)
        print("-------------------")
        for i in range(self.num_nodes):
            print(f"Node {i} has dataset of length {len(train_subsets[i])}")
        print("------------------")

    def train(self):
        # Train over total iterations rather than epochs since
        # nodes may have datasets of varying sizes (hence epoch would be inconsistent)
        for iteration in range(self.num_iterations):
            self.training_step(iteration)
            if iteration % self.iterations_per_validate == 0:
                self.validate(iteration)
            # Save model 0 checkpoints periodically
            if self.save_checkpoints_per_iter and iteration % self.save_checkpoints_per_iter == 0:
                torch.save(self.models[0], self.config.checkpoint_dir + f"iteration_{iteration}")
        accuracy = self.validate(self.num_iterations)
        if self.save_checkpoints_per_iter:
            torch.save(self.models[0], self.config.checkpoint_dir + f"iteration_{self.num_iterations}")
        return accuracy
    
    def training_step(self, iteration):
        # Implemented differently in child classes
        raise NotImplementedError

    def local_batch_loss(self, node):
        # Get a batch of data from the dataloader
        try: (_, data, label) = next(self.train_iters[node])
        except StopIteration:
            self.train_iters[node] = iter(self.train_loaders[node])
            (_, data, label) = next(self.train_iters[node])
        data = data.to(self.device)
        label = label.to(self.device)
        output = self.models[node](data)
        loss = self.base_loss_function(output, label)
        return loss
    
    def validate(self, iteration):
        accuracies = []
        for i in range(self.num_nodes):
            accuracy = self.node_validate(i, iteration)
            accuracies.append(accuracy)
        accuracies = np.array(accuracies)
        wandb.log({'iteration': iteration, 'val_acc': accuracies.mean(), 'std_acc': accuracies.std()})
        print(f"Iteration {iteration} average accuracy: {accuracies.mean()}")
        return accuracies.mean()

    def node_validate(self, node, iteration):
        self.models[node].eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for (_, data, label) in self.val_loader:
                data = data.to(self.device)
                label = label.to(self.device)
                output = self.models[node](data)
                val_loss += self.base_loss_function(output, label)
                preds = output.data.max(1, keepdim=True)[1]
                correct += preds.eq(label.data.view_as(preds)).sum()
        val_loss = val_loss.item() / len(self.val_dataset)
        accuracy = correct.item() / len(self.val_dataset)
        self.models[node].train() # return model to train mode
        wandb.log({'iteration': iteration, f'val_loss_{node}': val_loss, f'val_acc_{node}': accuracy})
        return accuracy
    
    def log_model_diff_norms(self, iteration):
        frozen_model_thetas = [] # extract the current weights
        for i in range(self.num_nodes):
            theta = torch.nn.utils.parameters_to_vector(self.models[i].parameters()).detach().clone()
            frozen_model_thetas.append(theta)

        for i in range(self.num_nodes):
            theta_l2_norm = np.linalg.norm(frozen_model_thetas[i].cpu().numpy(), ord=2)
            theta_linf_norm = np.linalg.norm(frozen_model_thetas[i].cpu().numpy(), ord=np.inf)
            neighbors = list(self.graph.neighbors(i))

            # Compute theta difference norms for neighboring agents
            for n in neighbors:
                theta_diff_norm = np.linalg.norm((frozen_model_thetas[i] - frozen_model_thetas[n]).cpu().numpy())
                wandb.log({'iteration': iteration, 'theta_diff_norm': theta_diff_norm, f'theta_l2_norm_{i}': theta_l2_norm, f'theta_linf_norm_{i}': theta_linf_norm})
        return frozen_model_thetas
    
    def initialize_dp_system(self):
        self.epsilon = self.config.dp_params.epsilon
        self.delta = self.config.dp_params.delta
        self.max_grad_norm = self.config.dp_params.max_grad_norm
        if isinstance(self.config.dp_params.effective_batch_size, int):
            self.effective_batch_size = self.config.dp_params.effective_batch_size
        else:
            self.effective_batch_size = self.batch_size

        # Set up privacy objects for each node
        self.noise_multipliers = []
        self.batch_sizes = []
        print("------------------")
        for i in range(len(self.models)):
            self.models[i] = GradSampleModule(self.models[i], strict=True).to(self.device) # for per-sample gradient computation
            self.train_loaders[i] = DPDataLoader.from_data_loader(self.train_loaders[i], distributed=False) # for Poisson sampling
            self.train_iters[i] = iter(self.train_loaders[i])
            sample_rate = 1 / len(self.train_loaders[i])
            self.batch_sizes.append(int(len(self.train_loaders[i].dataset) * sample_rate))
            if self.config.dp_params.noise_multiplier is None:
                noise_multiplier = get_noise_multiplier(target_epsilon=self.epsilon,
                                                        target_delta=self.delta,
                                                        sample_rate=sample_rate,
                                                        steps=self.num_iterations,
                                                        accountant="rdp")
            else:
                noise_multiplier = self.config.dp_params.noise_multiplier
            self.noise_multipliers.append(noise_multiplier)
            print(f"Node {i} sample rate: {sample_rate} and batch noise std: {noise_multiplier * self.max_grad_norm}")
        print("------------------")


class DiNNOSystem(DistributedSystem):
    # DiNNO paper: https://arxiv.org/abs/2109.08665
    # Referenced: https://github.com/javieryu/nn_distributed_training
    
    def __init__(self, config):
        super().__init__(config)
        dual_shape = torch.nn.utils.parameters_to_vector(self.base_model.parameters()).shape[0]
        self.duals = [torch.zeros((dual_shape), device=self.device) for i in range(self.num_nodes)]
        self.primal_iterations = self.config.dinno_params.primal_iterations
        self.rho = self.config.dinno_params.rho_init
        self.rho_scaling = self.config.dinno_params.rho_scaling

    def training_step(self, iteration):
        # log and extract the current primal variables
        thetas = self.log_model_diff_norms(iteration)
        self.rho *= self.rho_scaling # update the penalty parameter

        # Update per node
        for i in range(self.num_nodes):
            neighbors = list(self.graph.neighbors(i))
            theta_j = torch.stack([thetas[j] for j in neighbors])
            self.duals[i] += self.rho * torch.sum(thetas[i] - theta_j, dim=0) # equation 4a
            theta_reg = (theta_j + thetas[i]) / 2.0
            self.primal_update(i, theta_reg, iteration)
    
    def primal_update(self, node, theta_reg, iteration):
        # Configure the optimizer
        if self.config.optim_params.optimizer == "adam":
            optim = torch.optim.Adam(self.models[node].parameters(), self.lr[iteration])
        elif self.config.optim_params.optimizer == "sgd":
            optim = torch.optim.SGD(self.models[node].parameters(),
                                    lr=self.lr[iteration],
                                    momentum=self.config.optim_params.momentum,
                                    weight_decay=self.config.optim_params.weight_decay)
        else:
            raise NameError("DiNNO primal optimizer is unknown")
        
        train_loss = 0
        for _ in range(self.primal_iterations):
            optim.zero_grad()
            pred_loss = self.local_batch_loss(node) # model pass on the batch

            # Get the primal variable with the autodiff graph attached
            theta = torch.nn.utils.parameters_to_vector(self.models[node].parameters())
            reg = torch.sum((torch.cdist(theta.reshape(1, -1), theta_reg))**2)

            # Aggregate loss per node
            loss = pred_loss + torch.dot(theta, self.duals[node]) + self.rho * reg # equation 3
            loss.backward()
            train_loss += loss
            optim.step()
        wandb.log({'iteration': iteration, f'train_loss_{node}': train_loss.item()})
    

class DPDiNNOSystem(DiNNOSystem):
    
    def __init__(self, config):
        super().__init__(config)

        print("Setting up a DP DiNNO System using an RDP accountant")
        self.initialize_dp_system()

    def primal_update(self, node, theta_reg, iteration):
        # Configure the optimizer
        if self.config.optim_params.optimizer == "adam":
            optim = torch.optim.Adam(self.models[node].parameters(), self.lr[iteration])
        elif self.config.optim_params.optimizer == "sgd":
            optim = torch.optim.SGD(self.models[node].parameters(),
                                    lr=self.lr[iteration],
                                    momentum=self.config.optim_params.momentum,
                                    weight_decay=self.config.optim_params.weight_decay)
        else:
            raise NameError("DiNNO primal optimizer is unknown")
        
        # Wrap optimizer to perform DP operations
        optim = DiNNODPOptimizer(optimizer=optim,
                                 noise_multiplier=self.noise_multipliers[node],
                                 max_grad_norm=self.max_grad_norm,
                                 expected_batch_size=self.batch_sizes[node])
        
        train_loss = 0
        for _ in range(self.primal_iterations):
            optim.zero_grad()
            pred_loss = self.local_batch_loss(node) # model pass on the batch
            pred_loss.backward() # compute per sample gradients over data
            optim.pre_step() # accumulate gradients and add noise
            optim.clear_pred_grads()

            # Get the primal variable with the autodiff graph attached
            theta = torch.nn.utils.parameters_to_vector(self.models[node].parameters())
            reg = torch.sum((torch.cdist(theta.reshape(1, -1), theta_reg))**2)

            # Aggregate loss and gradients over full loss
            loss = torch.dot(theta, self.duals[node]) + self.rho * reg
            loss.backward()
            train_loss += loss + pred_loss
            optim.dinno_step() # update parameters
        noise_std = self.noise_multipliers[node] * self.max_grad_norm
        wandb.log({'iteration': iteration, f'train_loss_{node}': train_loss.item(), f'batch_noise_std_{node}': noise_std})


class DSGTSystem(DistributedSystem):
    # DSGT paper: https://arxiv.org/abs/1805.11454
    # Referenced: https://github.com/javieryu/nn_distributed_training
    
    def __init__(self, config):
        super().__init__(config)
        # Get list of all model parameter pointers
        self.param_lists = [list(self.models[i].parameters()) for i in range(self.num_nodes)]
        self.num_params = len(self.param_lists[0])

        # Compute the symmetric weight matrix - this is done once as we assume fixed graphs
        if self.config.graph_params.neighbor_weight == "average":
            self.W = get_neighbor_weights(self.graph).to(self.device)
        elif self.config.graph_params.neighbor_weight == "metropolis":
            self.W = get_metropolis_weights(self.graph).to(self.device)
        else:
            raise NameError("Invalid neighbor weighting scheme provided.")

        # Initialize gradients and y variables
        base_zeros = [torch.zeros_like(p, requires_grad=False, device=self.device) for p in self.param_lists[0]]
        self.grad_lists = [copy.deepcopy(base_zeros) for i in range(self.num_nodes)] # private quantity for each node
        self.y_lists = [copy.deepcopy(base_zeros) for i in range(self.num_nodes)] # shared among neighbor nodes

    def training_step(self, iteration):
        self.log_model_diff_norms(iteration)
        # Iterate over the agents for communication step
        for i in range(self.num_nodes):
            neighbors = list(self.graph.neighbors(i))
            self.weight_update(i, neighbors, self.lr[iteration])

        # Compute the batch loss and update using the gradients
        for i in range(self.num_nodes):
            neighbors = list(self.graph.neighbors(i))
            self.gradient_update(i, neighbors, iteration)

    def weight_update(self, node, neighbors, lr):
        # Update each parameter individually across self and all neighbors
        with torch.no_grad():
            for p in range(self.num_params):
                # Update from self parameters
                self.param_lists[node][p].multiply_(self.W[node, node])
                self.param_lists[node][p].add_(self.y_lists[node][p], alpha=-lr * self.W[node, node])
                
                # Update from neighbor parameters
                for j in neighbors:
                    self.param_lists[node][p].add_(self.param_lists[j][p], alpha=self.W[node, j])
                    self.param_lists[node][p].add_(self.y_lists[j][p], alpha=-lr * self.W[node, j])

    def gradient_update(self, node, neighbors, iteration):
        loss = self.local_batch_loss(node)
        loss.backward() # populate gradients

        with torch.no_grad():
            for p in range(self.num_params):
                # Compute weighted average of neighbors' y values
                self.y_lists[node][p].multiply_(self.W[node, node])
                for j in neighbors:
                    self.y_lists[node][p].add_(self.y_lists[j][p], alpha=self.W[node, j])
                # Add current parameter gradient
                self.y_lists[node][p].add_(self.param_lists[node][p].grad)
                # Subtract previous parameter gradient
                self.y_lists[node][p].add_(self.grad_lists[node][p], alpha=-1.0)

                # Update parameter gradient list with new gradient
                self.grad_lists[node][p] = self.param_lists[node][p].grad.detach().clone()
            # Zero out all parameter gradients
            self.models[node].zero_grad()
        wandb.log({'iteration': iteration, f'train_loss_{node}': loss.item()})


class DPDSGTSystem(DSGTSystem):
    
    def __init__(self, config):
        super().__init__(config)

        print("Setting up a DP DSGT System using an RDP accountant")
        self.initialize_dp_system()

    def gradient_update(self, node, neighbors, iteration):
        # Configure an optimizer to carry out DP operations, step is done manually in 'self.weight_update'
        original_optim = torch.optim.SGD(self.models[node].parameters(), lr=self.lr[iteration])

        # Wrap optimizer to perform DP operations
        optim = DPOptimizer(optimizer=original_optim,
                            noise_multiplier=self.noise_multipliers[node],
                            max_grad_norm=self.max_grad_norm,
                            expected_batch_size=self.effective_batch_size)

        num_iterations = self.effective_batch_size // self.batch_size
        optim.zero_grad() # reset summed_grad to 0
        train_loss = 0
        for i in range(num_iterations - 1):
            loss = self.local_batch_loss(node) # compute loss
            loss.backward() # populate gradients
            train_loss += loss
            optim.clip_and_accumulate()
            self.reset_param_grads(node, original_optim) # reset all params except for summed_grad
        loss = self.local_batch_loss(node) # compute loss
        loss.backward() # populate gradients
        train_loss += loss
        optim.pre_step() # clip, accumulate, and add noise to gradients

        with torch.no_grad():
            for p in range(self.num_params):
                # Compute weighted average of neighbors' y values
                self.y_lists[node][p].multiply_(self.W[node, node])
                for j in neighbors:
                    self.y_lists[node][p].add_(self.y_lists[j][p], alpha=self.W[node, j])
                # Add current parameter gradient
                self.y_lists[node][p].add_(self.param_lists[node][p].grad)
                # Subtract previous parameter gradient
                self.y_lists[node][p].add_(self.grad_lists[node][p], alpha=-1.0)
                # Update parameter gradient list with new gradient
                self.grad_lists[node][p] = self.param_lists[node][p].grad.detach().clone()
        optim.zero_grad() # zero out all parameter gradients
        noise_std = self.noise_multipliers[node] * self.max_grad_norm
        wandb.log({'iteration': iteration, f'train_loss_{node}': train_loss.item(), f'batch_noise_std_{node}': noise_std})

    def reset_param_grads(self, node, original_optim):
        for p in self.models[node].parameters():
            p.grad_sample = None
        original_optim.zero_grad(False)


class DistributedConsensusSystem(DistributedSystem):
    # Paper: https://arxiv.org/abs/1706.07880
    # Referenced: https://github.com/javieryu/nn_distributed_training
    
    def __init__(self, config):
        super().__init__(config)
        if self.config.optim_params.optimizer != "sgd":
            raise NameError("Invalid optimizer provided.")
        
        # Get list of all model parameter pointers
        self.param_lists = [list(self.models[i].parameters()) for i in range(self.num_nodes)]
        self.num_params = len(self.param_lists[0])

        # Compute the symmetric weight matrix - this is done once as we assume fixed graphs
        if self.config.graph_params.neighbor_weight == "average":
            self.W = get_neighbor_weights(self.graph).to(self.device)
        elif self.config.graph_params.neighbor_weight == "metropolis":
            self.W = get_metropolis_weights(self.graph).to(self.device)
        else:
            raise NameError("Invalid neighbor weighting scheme provided.")
        
    def training_step(self, iteration):
        self.log_model_diff_norms(iteration)

        # Communicate parameters with neighbors
        for i in range(self.num_nodes):
            neighbors = list(self.graph.neighbors(i))
            with torch.no_grad():
                for p in range(self.num_params): # update each parameter individually
                    self.param_lists[i][p].multiply_(self.W[i, i]) # accumulate self weights
                    for n in neighbors: # accumulate neighbor weights
                        self.param_lists[i][p].add_(self.param_lists[n][p] * self.W[i, n])

        # Compute training loss and update
        for i in range(self.num_nodes):
            self.update_step(i, iteration)

    def update_step(self, node, iteration):
        optim = torch.optim.SGD(self.models[node].parameters(),
                                lr=self.lr[iteration],
                                momentum=self.config.optim_params.momentum,
                                weight_decay=self.config.optim_params.weight_decay)
        optim.zero_grad()
        train_loss = self.local_batch_loss(node) # compute loss
        train_loss.backward() # populate gradients
        optim.step() # update using the gradients
        wandb.log({'iteration': iteration, f'train_loss_{node}': train_loss.item()})

    def manual_update_step(self, node, iteration):
        loss = self.local_batch_loss(node) # compute loss
        loss.backward() # populate gradients

        # Update each parameter individually across self and all neighbors
        with torch.no_grad():
            for p in range(self.num_params):
                self.param_lists[node][p].add_(-self.lr[iteration] * self.param_lists[node][p].grad)
        self.models[node].zero_grad()
        return loss


class DistributedDPConsensusSystem(DistributedConsensusSystem):
    
    def __init__(self, config):
        super().__init__(config)
        print("Setting up a DP Consensus System using an RDP accountant")
        self.initialize_dp_system()

    def update_step(self, node, iteration):
        original_optim = torch.optim.SGD(self.models[node].parameters(),
                                lr=self.lr[iteration],
                                momentum=self.config.optim_params.momentum,
                                weight_decay=self.config.optim_params.weight_decay)
        
        # Wrap optimizer to perform DP operations
        optim = DPOptimizer(optimizer=original_optim,
                            noise_multiplier=self.noise_multipliers[node],
                            max_grad_norm=self.max_grad_norm,
                            expected_batch_size=self.effective_batch_size)
            
        num_iterations = self.effective_batch_size // self.batch_size
        optim.zero_grad() # reset summed_grad to 0
        train_loss = 0
        for i in range(num_iterations - 1):
            loss = self.local_batch_loss(node) # compute loss
            loss.backward() # populate gradients
            train_loss += loss
            optim.clip_and_accumulate()
            self.reset_param_grads(node, original_optim) # reset all params except for summed_grad
        loss = self.local_batch_loss(node) # compute loss
        loss.backward() # populate gradients
        train_loss += loss
        optim.step() # update using the gradients
        noise_std = self.noise_multipliers[node] * self.max_grad_norm
        wandb.log({'iteration': iteration, f'train_loss_{node}': train_loss.item(), f'batch_noise_std_{node}': noise_std})
  
    def reset_param_grads(self, node, original_optim):
        for p in self.models[node].parameters():
            p.grad_sample = None
        original_optim.zero_grad(False)

class DistributedSystemNoComms(DistributedSystem):
    
    def __init__(self, config):
        super().__init__(config)

    def training_step(self, iteration):
        self.log_model_diff_norms(iteration)
        for i in range(self.num_nodes):
            train_loss = self.weight_update(i, iteration)
            wandb.log({'iteration': iteration, f'train_loss_{i}': train_loss.item()})

    def weight_update(self, node, iteration):
        # Update learning rate
        for g in self.optims[node].param_groups:
            g['lr'] = self.lr[iteration]

        # Train one iteration
        self.optims[node].zero_grad()
        train_loss = self.local_batch_loss(node)
        train_loss.backward()
        self.optims[node].step()
        return train_loss