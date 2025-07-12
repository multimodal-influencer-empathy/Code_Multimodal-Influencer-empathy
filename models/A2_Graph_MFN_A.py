"""
paper: Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph
Reference From: https://github.com/pliang279/MFN & https://github.com/A2Zadeh/CMU-MultimodalSDK
"""

import copy
from torch import Tensor
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain, combinations

class DynamicFusionGraph(nn.Module):
    """
    Input: List of modality inputs [InputModalA, InputModalB, ..., InputModalZ], each representing a singleton vertex.
    Output (from forward): (t_output, outputs, efficacies)
        - t_output: top-level representation (τ), equivalent to c_hat in MFN
        - outputs: representations of each node
        - efficacies: alpha coefficients [batch_size, n] used in weighted fusion
    """

    def __init__(self, pattern_model, in_dimensions, out_dimension, efficacy_model, device):
        super(DynamicFusionGraph, self).__init__()
        self.num_modalities = len(in_dimensions)
        self.in_dimensions = in_dimensions
        self.out_dimension = out_dimension

        # Compute all non-empty subsets (excluding empty set) for powerset construction
        self.powerset = list(chain.from_iterable(
            combinations(range(self.num_modalities), r)
            for r in range(self.num_modalities + 1)))[1:]

        self.input_shapes = {tuple([key]): value for key, value in zip(range(self.num_modalities), in_dimensions)}
        self.networks = {}
        self.total_input_efficacies = 0

        # Construct network for each multimodal subset
        for key in self.powerset[self.num_modalities:]:
            unimodal_dims = sum(in_dimensions[modality] for modality in key)
            multimodal_dims = ((2 ** len(key) - 2) - len(key)) * out_dimension
            self.total_input_efficacies += 2 ** len(key) - 2
            final_dims = unimodal_dims + multimodal_dims
            self.input_shapes[key] = final_dims
            pattern_copy = copy.deepcopy(pattern_model)

            final_model = nn.Sequential(
                *[nn.Linear(self.input_shapes[key], list(pattern_copy.children())[0].in_features), pattern_copy]
            ).to(device)
            self.networks[key] = final_model

        self.total_input_efficacies += 2 ** self.num_modalities - 1
        self.t_in_dimension = unimodal_dims + (2 ** self.num_modalities - self.num_modalities - 1) * out_dimension

        pattern_copy = copy.deepcopy(pattern_model)
        self.t_network = nn.Sequential(
            *[nn.Linear(self.t_in_dimension, list(pattern_copy.children())[0].in_features), pattern_copy]
        ).to(device)
        self.efficacy_model = nn.Sequential(
            *[nn.Linear(sum(in_dimensions), list(efficacy_model.children())[0].in_features),
              efficacy_model,
              nn.Linear(list(efficacy_model.children())[-1].out_features, self.total_input_efficacies)] ).to(device)


    def __call__(self, in_modalities):
        return self.fusion(in_modalities)

    def fusion(self, in_modalities):
        outputs = {}
        for modality, index in zip(in_modalities, range(len(in_modalities))):
            outputs[tuple([index])] = modality

        # Get fusion weights (efficacies)
        efficacies = self.efficacy_model(torch.cat([x for x in in_modalities], dim=1))

        efficacy_index = 0
        for key in self.powerset[self.num_modalities:]:
            small_power_set = list(chain.from_iterable(combinations(key, r) for r in range(len(key) + 1)))[1:-1]
            this_input = torch.cat(
                [outputs[x].cuda() * efficacies[:, efficacy_index + y].view(-1, 1).cuda()
                 for x, y in zip(small_power_set, range(len(small_power_set)))], dim=1)
            outputs[key] = self.networks[key](this_input)
            efficacy_index += len(small_power_set)

        small_power_set.append(tuple(range(self.num_modalities)))
        t_input = torch.cat(
            [outputs[x].cuda() * efficacies[:, efficacy_index + y].view(-1, 1).cuda()
             for x, y in zip(small_power_set, range(len(small_power_set)))], dim=1)
        t_output = self.t_network(t_input)

        return t_output, outputs, efficacies

    def forward(self, x):
        print("Not yet implemented for nn.Sequential")
        exit(-1)

class OnesLike(nn.Module):
    """Return a tensor of all ones with shape (batch_size, 19) for constant fusion weights."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x) -> Tensor:
        b = x.shape[0]
        return torch.ones((b, 19))

class Concat(nn.Module):
    """Concatenate input list along specified dimension."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: list) -> Tensor:
        return torch.concat(x, dim=self.dim)

class MultiOutputConcat(Concat):
    """Wrapper for Concat, returning additional placeholders (used when graph fusion is disabled)."""

    def forward(self, x: list) -> Tensor:
        return super().forward(x), 0, torch.ones(1)

class Select(nn.Module):
    """Select a specific index from list input."""

    def __init__(self, select_index: int) -> None:
        super().__init__()
        self.index = select_index

    def forward(self, x: list) -> Tensor:
        return x[self.index]

class MultiSelect(nn.Module):
    """Select multiple indices from list input."""

    def __init__(self, select_index: list) -> None:
        super().__init__()
        self.index = select_index

    def forward(self, x: list) -> Tensor:
        return [x[i] for i in self.index]

# Utility: parse which modalities are active from ablation string
def get_active_modalities(abl):
    return [m for m in ['l', 'a', 'v'] if m in abl]

# Get input dimensions based on active modalities
def get_input_dims_by_abl(abl, l_in=768, a_in=72, v_in=35):
    mod_dict = {'l': l_in, 'a': a_in, 'v': v_in}
    return [mod_dict[m] for m in abl if m in mod_dict]

class A2_Graph_MFN_A(nn.Module):
    def __init__(self, args):
        super(A2_Graph_MFN_A, self).__init__()

        self.abl: str = args.abl
        self.d_l, self.d_a, self.d_v = args.feature_dims
        self.dh_l, self.dh_a, self.dh_v = args.hidden_dims

        self.noW = args.noW
        self.noG = args.noG
        self.noM = args.noM
        self.num_modalities = len(self.abl)

        # Compute total hidden dim
        total_h_dim = sum(d for m, d in zip('lav', [self.dh_l, self.dh_a, self.dh_v]) if m in self.abl)

        self.mem_dim = args.memsize
        self.inner_node_dim = args.inner_node_dim
        self.singleton_l_size, self.singleton_a_size, self.singleton_v_size = args.hidden_dims

        output_dim = 1
        gammaInShape = total_h_dim + self.mem_dim if self.noG else self.inner_node_dim + self.mem_dim

        if self.noM:
            self.mem_dim = 0

        final_out = total_h_dim if (self.num_modalities == 1 or self.noM) else total_h_dim + self.mem_dim

        # Network architecture parameters
        h_att2 = args.NNConfig["shapes"]
        h_gamma1 = args.gamma1Config["shapes"]
        h_gamma2 = args.gamma2Config["shapes"]
        h_out = args.outConfig["shapes"]
        att2_dropout = args.NNConfig["drop"]
        gamma1_dropout = args.gamma1Config["drop"]
        gamma2_dropout = args.gamma2Config["drop"]
        out_dropout = args.outConfig["drop"]

        # Define LSTM and transformation layers for each modality
        self.lstm_l = nn.LSTMCell(768, self.dh_l)
        self.l_transform = self.build_lstm_transform(self.dh_l * 2, self.singleton_l_size)

        self.lstm_a = nn.LSTMCell(self.d_a, self.dh_a)
        self.a_transform = self.build_lstm_transform(self.dh_a * 2, self.singleton_a_size)

        self.lstm_v = nn.LSTMCell(self.d_v, self.dh_v)
        self.v_transform = self.build_lstm_transform(self.dh_v * 2, self.singleton_v_size)

        if self.noG:
            print("no g", "*" * 10)
            self.graph_mfn = MultiOutputConcat(dim=1)
        else:
            pattern_model = nn.Sequential(nn.Linear(100, self.inner_node_dim)).to(args.device)
            efficacy_model = nn.Sequential(nn.Linear(100, self.inner_node_dim)).to(args.device)
            self.graph_mfn = DynamicFusionGraph(
                pattern_model,
                [self.singleton_l_size, self.singleton_a_size, self.singleton_v_size],
                self.inner_node_dim,
                efficacy_model,
                args.device,
            ).to(args.device)

        self.att2_fc1 = nn.Linear(total_h_dim if self.noG else self.inner_node_dim, h_att2)
        self.att2_fc2 = nn.Linear(h_att2, self.mem_dim)
        self.att2_dropout = nn.Dropout(att2_dropout)

        if self.noM:
            print("no m", "*" * 10)
            self.MSM = self.forward_one
            self.mem = 1
            self.out_concant = nn.Sequential(MultiSelect([0, 1, 2]), Concat(dim=1))
        else:
            self.gamma1_fc1 = nn.Linear(gammaInShape, h_gamma1)
            self.gamma1_fc2 = nn.Linear(h_gamma1, self.mem_dim)
            self.gamma1_dropout = nn.Dropout(gamma1_dropout)

            self.gamma2_fc1 = nn.Linear(gammaInShape, h_gamma2)
            self.gamma2_fc2 = nn.Linear(h_gamma2, self.mem_dim)
            self.gamma2_dropout = nn.Dropout(gamma2_dropout)

            self.MSM = self.forward_msm
            self.out_concant = Concat(dim=1)

        self.out_fc1 = nn.Linear(final_out, h_out)
        self.out_fc2 = nn.Linear(h_out, output_dim)
        self.out_dropout = nn.Dropout(out_dropout)

    def build_lstm_transform(self, in_ch, out_ch):
        if self.noW:
            print("no w", "*" * 10, nn.Sequential(Select(1), nn.ReLU()))
            return nn.Sequential(Select(1), nn.ReLU())
        return nn.Sequential(Concat(dim=1), nn.Linear(in_ch, out_ch), nn.ReLU())

    def forward_msm(self, attended):
        t1 = self.att2_fc1(attended)
        t2 = self.att2_dropout(F.relu(t1))
        t3 = self.att2_fc2(t2)
        cHat = torch.tanh(t3)
        both = torch.cat([attended, self.mem], dim=1)
        gamma1 = torch.sigmoid(self.gamma1_fc2(self.gamma1_dropout(F.relu(self.gamma1_fc1(both)))))
        gamma2 = torch.sigmoid(self.gamma2_fc2(self.gamma2_dropout(F.relu(self.gamma2_fc1(both)))))
        self.mem = gamma1 * self.mem + gamma2 * cHat
        return self.mem

    def forward_one(self, _):
        pass

    def forward(self, text_x, audio_x, video_x):
        text_x = text_x.permute(1, 0, 2)
        audio_x = audio_x.permute(1, 0, 2)
        video_x = video_x.permute(1, 0, 2)

        n = text_x.size()[1]
        t = text_x.size()[0]

        self.h_l = torch.zeros(n, self.dh_l).to(text_x.device)
        self.h_a = torch.zeros(n, self.dh_a).to(text_x.device)
        self.h_v = torch.zeros(n, self.dh_v).to(text_x.device)
        self.c_l = torch.zeros(n, self.dh_l).to(text_x.device)
        self.c_a = torch.zeros(n, self.dh_a).to(text_x.device)
        self.c_v = torch.zeros(n, self.dh_v).to(text_x.device)
        self.mem = torch.zeros(n, self.mem_dim).to(text_x.device)

        all_h_ls, all_h_as, all_h_vs, all_mems, all_efficacies = [], [], [], [], []

        for i in range(t):
            prev_h_l, prev_h_a, prev_h_v = self.h_l, self.h_a, self.h_v

            new_h_l, new_c_l = self.lstm_l(text_x[i], (self.h_l, self.c_l))
            new_h_a, new_c_a = self.lstm_a(audio_x[i], (self.h_a, self.c_a))
            new_h_v, new_c_v = self.lstm_v(video_x[i], (self.h_v, self.c_v))

            l_singleton_input = self.l_transform([prev_h_l, new_h_l])
            a_singleton_input = self.a_transform([prev_h_a, new_h_a])
            v_singleton_input = self.v_transform([prev_h_v, new_h_v])

            if hasattr(self, 'abl'):
                l_singleton_input = l_singleton_input if 'l' in self.abl else torch.zeros_like(l_singleton_input)
                a_singleton_input = a_singleton_input if 'a' in self.abl else torch.zeros_like(a_singleton_input)
                v_singleton_input = v_singleton_input if 'v' in self.abl else torch.zeros_like(v_singleton_input)


            attended, _, efficacies = self.graph_mfn([l_singleton_input, a_singleton_input, v_singleton_input])
            all_efficacies.append(efficacies.cpu().detach().tolist())
            self.MSM(attended)
            all_mems.append(self.mem)

            self.h_l, self.c_l = new_h_l, new_c_l
            self.h_a, self.c_a = new_h_a, new_c_a
            self.h_v, self.c_v = new_h_v, new_c_v
            all_h_ls.append(self.h_l)
            all_h_as.append(self.h_a)
            all_h_vs.append(self.h_v)



        last_h_l, last_h_a, last_h_v = all_h_ls[-1], all_h_as[-1], all_h_vs[-1]
        last_mem = all_mems[-1]

        if 'l' not in self.abl:
            last_h_l = torch.zeros_like(last_h_l)
        if 'a' not in self.abl:
            last_h_a = torch.zeros_like(last_h_a)
        if 'v' not in self.abl:
            last_h_v = torch.zeros_like(last_h_v)

        if self.abl == "l":
            last_hs = last_h_l
        elif self.abl == "a":
            last_hs = last_h_a
        elif self.abl == "v":
            last_hs = last_h_v
        elif self.abl == "la":
            last_hs = self.out_concant([last_h_l, last_h_a, last_mem])
        elif self.abl == "lv":
            last_hs = self.out_concant([last_h_l, last_h_v, last_mem])
        elif self.abl == "av":
            last_hs = self.out_concant([last_h_a, last_h_v, last_mem])
        elif self.abl == "lav":
            last_hs = self.out_concant([last_h_l, last_h_a, last_h_v, last_mem])

        out_fc1_result = self.out_fc1(last_hs)
        relu_result = F.relu(out_fc1_result)
        dropout_result = self.out_dropout(relu_result)
        output = self.out_fc2(dropout_result)

        return output
