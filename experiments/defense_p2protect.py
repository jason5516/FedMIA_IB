# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import copy
from tqdm import tqdm

# --- Model Structure ---
class Net(torch.nn.Module):
    def __init__(self, n_feature):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_feature, 500)
        self.fc2 = nn.Linear(500, 300)
        self.fc3 = nn.Linear(300, 2)
        self.layer_output_list = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def eval_layer_output(self, x):
        self.layer_output_list = []
        x = F.relu(self.fc1(x))
        self.layer_output_list.append(x)
        x = F.relu(self.fc2(x))
        self.layer_output_list.append(x)
        x = self.fc3(x)
        self.layer_output_list.append(x)
        return self.layer_output_list, x

# --- Helper Functions for Defense ---

def _create_ohe(num_output_classes):
    """Creates a one-hot encoding matrix."""
    return F.one_hot(torch.arange(0, num_output_classes), num_classes=num_output_classes).float()

def _create_label_tensors(num_output_classes):
    """Creates a tensor of labels."""
    return torch.LongTensor([[i] for i in range(num_output_classes)])

def _get_grad_direction(example_input, net, optimizer, y_ext_label):
    """Calculates the gradient direction for a given input and label."""
    net.zero_grad()
    model_output = net(torch.unsqueeze(example_input, 0))
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(model_output, y_ext_label)
    
    # We need to calculate gradients, so we can't use torch.no_grad() here.
    # Instead, we'll manually zero the gradients and handle them.
    optimizer.zero_grad()
    loss.backward()
    
    grad_vecs = []
    for param in net.parameters():
        if param.grad is not None:
            grad_vecs.append(param.grad.clone().flatten())
            
    net.zero_grad() # Clean up gradients
    
    if not grad_vecs:
        # This case should ideally not happen if the model has trainable parameters
        # and the loss is connected to them.
        return torch.tensor([], device=next(net.parameters()).device)

    return torch.cat(grad_vecs)

def _get_point_via_bisection(org_y, graddif_y, eps, seg_point_start, seg_point_end):
    """Finds the optimal interpolation point using bisection search."""
    seg_point = (seg_point_start + seg_point_end) / 2
    cur_y = (1 - seg_point) * org_y + seg_point * graddif_y

    org_label = torch.argmax(org_y)
    cur_label = torch.argmax(cur_y)

    dif_stop = 1e-3

    if torch.abs(torch.max(org_y) - torch.max(cur_y)) <= eps:
        if org_label != cur_label:
            return _get_point_via_bisection(org_y, graddif_y, eps, seg_point_start, seg_point)
        else:
            if seg_point_end - seg_point_start < dif_stop:
                return seg_point
            else:
                return _get_point_via_bisection(org_y, graddif_y, eps, seg_point, seg_point_end)
    else:
        return _get_point_via_bisection(org_y, graddif_y, eps, seg_point_start, seg_point)

def _get_modified_y_list(org_y_list, max_graddif_y_list, eps):
    """Generates the list of modified target outputs."""
    modified_y_list = []
    for org_y, max_graddif_y in tqdm(zip(org_y_list, max_graddif_y_list), desc="Step 2: Generating modified targets", total=len(org_y_list)):
        seg_point = _get_point_via_bisection(org_y, max_graddif_y, eps, 0, 1)
        modified_y = (1 - seg_point) * org_y + seg_point * max_graddif_y
        modified_y_list.append(modified_y)
    return modified_y_list

class _MyLoss(nn.Module):
    """Custom loss function to measure distance to modified targets."""
    def __init__(self):
        super(_MyLoss, self).__init__()

    def forward(self, org_y_prob_list, modified_y_list):
        sum_loss = torch.tensor(0, dtype=torch.float, requires_grad=True).to(org_y_prob_list[0].device)
        for i in range(len(org_y_prob_list)):
            cur_loss = torch.norm(org_y_prob_list[i] - modified_y_list[i], p=2)
            sum_loss = sum_loss + cur_loss
        return sum_loss

# --- Main Defense Function ---

def defend_local_model(
    original_model,
    x_train,
    y_train,
    num_classes=2,
    eps_setting=0.1,
    retrain_epochs=1,
    learning_rate=0.0003,
    batch_size=128,
    device='cpu'
):
    """
    Applies a gradient-based defense mechanism to a local model.

    Args:
        original_model (torch.nn.Module): The model to be defended.
        x_train (torch.Tensor): Training data features.
        y_train (torch.Tensor): Training data labels.
        num_classes (int): The number of classes in the dataset.
        eps_setting (float): The defense intensity parameter (epsilon).
        retrain_epochs (int): Number of epochs to retrain the model on modified targets.
        learning_rate (float): Learning rate for the retraining optimizer.
        batch_size (int): Batch size for retraining.
        device (str): The device to run the defense on ('cpu' or 'cuda').

    Returns:
        torch.nn.Module: The defended model.
    """
    # 1. Setup and Initialization
    defended_model = copy.deepcopy(original_model).to(device)
    defended_model.train()
    for param in defended_model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, defended_model.parameters()), lr=learning_rate)
    
    y_ext_list_prob = _create_ohe(num_classes).to(device)
    y_ext_list_label = _create_label_tensors(num_classes).to(device)
    
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    # len_example = x_train.size(0)
    len_example = 500  # Reduce number of examples for faster defense
    x_train = x_train[:len_example]
    y_train = y_train[:len_example]
    y_train_squeezed = y_train.squeeze()

    # 2. Find the most confusing label for each data point
    y_ext_list = []
    
    # Get original model outputs (probabilities)
    with torch.no_grad():
        original_outputs = defended_model(x_train)
        org_y_list_prob = F.softmax(original_outputs, dim=1)

    for i in tqdm(range(len_example), desc="Step 1: Finding most confusing labels"):
        example_y_org_label = torch.LongTensor([y_train_squeezed[i]]).to(device)
        grad_dir_org = _get_grad_direction(x_train[i], defended_model, optimizer, example_y_org_label)

        max_norm_square = -1
        y_star_max_label = -1

        for j in range(len(y_ext_list_label)):
            if y_ext_list_label[j].item() == example_y_org_label.item():
                continue # Skip the true label
            
            grad_dir_ext = _get_grad_direction(x_train[i], defended_model, optimizer, y_ext_list_label[j])
            
            # Normalize gradients before comparing
            norm_grad_org = grad_dir_org / torch.norm(grad_dir_org, p=2)
            norm_grad_ext = grad_dir_ext / torch.norm(grad_dir_ext, p=2)
            
            cur_dif = norm_grad_ext - norm_grad_org
            cur_norm_square = torch.pow(torch.norm(cur_dif, p=2), 2)

            if cur_norm_square > max_norm_square:
                max_norm_square = cur_norm_square
                y_star_max_label = y_ext_list_label[j]
        
        y_ext_list.append(torch.squeeze(y_ext_list_prob[y_star_max_label]))

    # 3. Generate modified targets using bisection search
    modified_y_list = _get_modified_y_list(org_y_list_prob, y_ext_list, eps_setting)
    modified_y_tensor = torch.stack(modified_y_list).detach()

    # 4. Retrain the model on the modified targets
    loss_func = _MyLoss()
    torch_dataset = torch.utils.data.TensorDataset(x_train, modified_y_tensor)
    train_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)

    for epoch in tqdm(range(retrain_epochs), desc="Step 3: Retraining model"):
        for step, (batch_x, batch_y_modified) in enumerate(train_loader):
            batch_x, batch_y_modified = batch_x.to(device), batch_y_modified.to(device)
            output = defended_model(batch_x)
            y_prob_list = F.softmax(output, dim=1)
            
            loss = loss_func(y_prob_list, batch_y_modified)
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True) # retain_graph might be needed depending on loss structure
            optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch [{epoch+1}/{retrain_epochs}], Loss: {loss.item():.4f}")

    print("Defense process completed.")
    defended_model.eval()
    return defended_model

if __name__ == '__main__':
    # This is an example of how to use the defend_local_model function.
    # It simulates loading data and a model for a single client and defending it.
    
    import pandas as pd

    print("--- Running Defense Module Example ---")
    
    # --- Parameters ---
    CLIENT_ID = 0
    N_FEATURES = 14 # For Adult dataset
    
    # --- Load Data ---
    print(f"Loading data for client {CLIENT_ID}...")
    # Note: Adjust paths if you run this script from a different location.
    try:
        x_train_df = pd.read_csv(f'../dataset/adult/x_train_adult_{CLIENT_ID}.csv')
        y_train_df = pd.read_csv(f'../dataset/adult/y_train_adult_{CLIENT_ID}.csv')
        x_test_df = pd.read_csv(f'../dataset/adult/x_test_adult_{CLIENT_ID}.csv')
        y_test_df = pd.read_csv(f'../dataset/adult/y_test_adult_{CLIENT_ID}.csv')

        x_train_tensor = torch.FloatTensor(np.array(x_train_df))
        y_train_tensor = torch.LongTensor(np.array(y_train_df))
        x_test_tensor = torch.FloatTensor(np.array(x_test_df))
        y_test_tensor = torch.LongTensor(np.array(y_test_df)).squeeze()
    except FileNotFoundError:
        print("\nERROR: Example data not found. Make sure you are running this from the 'defend' directory")
        print("and the dataset paths like '../dataset/adult/...' are correct.\n")
        exit()


    # --- Load Model ---
    # In a real scenario, this model would be the global model from the server.
    # Here, we just initialize a new one for demonstration.
    print("Initializing a new model for demonstration...")
    client_model = Net(n_feature=N_FEATURES)

    # --- Test Accuracy Before Defense ---
    client_model.eval()
    with torch.no_grad():
        out_test = client_model(x_test_tensor)
        prediction = torch.max(out_test, 1)[1]
        accuracy = (prediction == y_test_tensor).sum().item() / y_test_tensor.size(0)
        print(f"Accuracy BEFORE defense: {accuracy:.4f}")

    # --- Apply Defense ---
    defended_model = defend_local_model(
        original_model=client_model,
        x_train=x_train_tensor,
        y_train=y_train_tensor,
        eps_setting=0.1,
        retrain_epochs=50 # Reduced for a quick example run
    )

    # --- Test Accuracy After Defense ---
    defended_model.eval()
    with torch.no_grad():
        out_test_defended = defended_model(x_test_tensor)
        prediction_defended = torch.max(out_test_defended, 1)[1]
        accuracy_defended = (prediction_defended == y_test_tensor).sum().item() / y_test_tensor.size(0)
        print(f"Accuracy AFTER defense: {accuracy_defended:.4f}")

    # --- Save The Defended Model (Optional) ---
    # save_path = f'../model/adult_defend/protected_client_{CLIENT_ID}_example.pkl'
    # torch.save(defended_model, save_path)
    # print(f"Defended model saved to {save_path}")