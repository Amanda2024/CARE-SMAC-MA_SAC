import torch
import numpy as np
from torch import nn
from typing import List


ModelType = torch.nn.Module
TensorType = torch.Tensor

def _get_list_of_layers(
    input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
) -> List[nn.Module]:
    """Utility function to get a list of layers. This assumes all the hidden
    layers are using the same dimensionality.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): dimension of the hidden layers.
        output_dim (int): dimension of the output layer.
        num_layers (int): number of layers in the mlp.

    Returns:
        ModelType: [description]
    """
    mods: List[nn.Module]
    if num_layers == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        mods.append(nn.Linear(hidden_dim, output_dim))
    return mods

def build_mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> ModelType:
    """Utility function to build a mlp model. This assumes all the hidden
    layers are using the same dimensionality.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): dimension of the hidden layers.
        output_dim (int): dimension of the output layer.
        num_layers (int): number of layers in the mlp.

    Returns:
        ModelType: [description]
    """
    mods: List[nn.Module] = _get_list_of_layers(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
    )
    return nn.Sequential(*mods)


class Linear(nn.Module):
    def __init__(
        self, num_experts: int, in_features: int, out_features: int, bias: bool = True
    ):
        """torch.nn.Linear layer extended for use as a mixture of experts.

        Args:
            num_experts (int): number of experts in the mixture.
            in_features (int): size of each input sample for one expert.
            out_features (int): size of each output sample for one expert.
            bias (bool, optional): if set to ``False``, the layer will
                not learn an additive bias. Defaults to True.
        """
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.rand(self.num_experts, self.in_features, self.out_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.rand(self.num_experts, 1, self.out_features))
            self.use_bias = True
        else:
            self.use_bias = False

    def forward(self, x: TensorType) -> TensorType:
        if self.use_bias:
            return x.matmul(self.weight) + self.bias
        else:
            return x.matmul(self.weight)

    def extra_repr(self) -> str:
        return f"num_experts={self.num_experts}, in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias}"

class FeedForward(nn.Module):
    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        num_layers: int,
        hidden_features: int,
        bias: bool = True,
    ):
        """A feedforward model of mixture of experts layers.

        Args:
            num_experts (int): number of experts in the mixture.
            in_features (int): size of each input sample for one expert.
            out_features (int): size of each output sample for one expert.
            num_layers (int): number of layers in the feedforward network.
            hidden_features (int): dimensionality of hidden layer in the
                feedforward network.
            bias (bool, optional): if set to ``False``, the layer will
                not learn an additive bias. Defaults to True.
        """
        super().__init__()
        layers: List[nn.Module] = []
        current_in_features = in_features
        for _ in range(num_layers - 1):
            linear = Linear(
                num_experts=num_experts,
                in_features=current_in_features,
                out_features=hidden_features,
                bias=bias,
            )
            layers.append(linear)
            layers.append(nn.ReLU())
            current_in_features = hidden_features
        linear = Linear(
            num_experts=num_experts,
            in_features=current_in_features,
            out_features=out_features,
            bias=bias,
        )
        layers.append(linear)
        self._model = nn.Sequential(*layers)

    def forward(self, x: TensorType) -> TensorType:
        return self._model(x)

    def __repr__(self) -> str:
        return str(self._model)


class AttentionBasedExperts(nn.Module):
    def __init__(
        self,
        num_experts: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        temperature: bool,
    ):
        super().__init__()
        self.temperature = temperature
        self.trunk = build_mlp(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=num_experts,
            num_layers=num_layers,
        )
        # self.trunk.apply(weight_init)
        self._softmax = torch.nn.Softmax(dim=1)

    def forward(self, encoding) -> TensorType:
        # import pdb
        # pdb.set_trace()
        emb = encoding

        output = self.trunk(emb)  # 10.50  --> 10.4
        gate = self._softmax(output / self.temperature)

        if len(gate.shape) > 2:
            breakpoint()
        return gate.t().unsqueeze(2)

class CARE_ENCODER(nn.Module):
    def __init__(self, env_obs_shape, out_features, task_emb_dim):
        super().__init__()
        self.num_experts = 4
        self.task_emb_dim = task_emb_dim
        self.num_layers = 2
        self.hidden_dim = 50
        self.feature_dim = 50

        self.moe = FeedForward(
            num_experts=self.num_experts,
            in_features=env_obs_shape,
            # encoder_cfg.feature_dim,
            out_features=out_features,
            num_layers=self.num_layers,
            hidden_features=self.hidden_dim,
            bias=True,
        )
        self.att_expert = AttentionBasedExperts(
            num_experts=self.num_experts,
            embedding_dim=self.task_emb_dim,
            hidden_dim=self.feature_dim,
            num_layers=self.num_layers,
            temperature=1.0,
        )
    def forward(self, env_obs, task_emb):
        encoder_mask = self.att_expert(task_emb[[0]].float())  # 一个task， 所以取task_emb[0]：--》 torch.Size([4, 1, 1])
        encoding = self.moe(env_obs.float())  # torch.Size([4, 1, 50])

        sum_of_masked_encoding = (encoding * encoder_mask).sum(dim=0)  # torch.Size([1, 50])
        sum_of_encoder_count = encoder_mask.sum(dim=0)  # 4.1.1  -->   1.1
        encoding_exp = sum_of_masked_encoding / sum_of_encoder_count  ## (1.50)/(1.1)  == (1.50)

        new_state = torch.cat((encoding_exp, task_emb), dim=1)  # 1.100

        return new_state


if __name__ == "__main__":
    env_obs = torch.rand(1, 32)
    task_emb = torch.rand(1, 50)
    env_obs_shape = env_obs.shape
    task_emb_dim = 50
    out_features = 50
    care = CARE_ENCODER(env_obs_shape[1], out_features, task_emb_dim)
    print(care(env_obs, task_emb))
    print(care(env_obs, task_emb).shape)
    # num_experts = 4
    # feature_dim = 50
    # emb_dim = 50
    # num_layers = 2
    # hidden_dim = 50
    #
    # moe = FeedForward(
    #     num_experts=num_experts,
    #     in_features=env_obs_shape[1],
    #     # encoder_cfg.feature_dim,
    #     out_features=feature_dim,
    #     num_layers=num_layers,
    #     hidden_features=hidden_dim,
    #     bias=True,
    # )
    # att_expert = AttentionBasedExperts(
    #     num_experts = num_experts,
    #     embedding_dim = emb_dim,
    #     hidden_dim = feature_dim,
    #     num_layers = num_layers,
    #     temperature = 1.0,
    # )
    # encoder_mask = att_expert(task_emb)  # torch.Size([4, 1, 1])
    # encoding = moe(env_obs)              # torch.Size([4, 1, 50])
    #
    # sum_of_masked_encoding = (encoding * encoder_mask).sum(dim=0)  # torch.Size([1, 50])
    # sum_of_encoder_count = encoder_mask.sum(dim=0)  # 4.1.1  -->   1.1
    # encoding_exp = sum_of_masked_encoding / sum_of_encoder_count  ## (1.50)/(1.1)  == (1.50)
    #
    # new_state = torch.cat((encoding_exp, task_emb), dim=1)





