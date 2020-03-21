import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pack_padded_sequence

import config
import counting

import sys
import os
#
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# from networks.fusions.fusions import MCB, MCB_classify, MLB, ConcatMLP, Tucker, Mutan, BlockTucker, Block


class Net(nn.Module):
    """ Based on ``Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering'' [0]
    [0]: https://arxiv.org/abs/1704.03162
    """

    def __init__(self, embedding_tokens):
        super(Net, self).__init__()
        question_features = 1024
        vision_features = config.output_features
        glimpses = 2
        # glimpses = 3
        objects = 10

        self.text = TextProcessor(
            embedding_tokens=embedding_tokens,
            embedding_features=300,
            lstm_features=question_features,
            drop=0.5,
        )
        self.attention = Attention(
            v_features=vision_features,
            q_features=question_features,
            mid_features=512,
            glimpses=glimpses,
            drop=0.5,
        )
        self.classifier = Classifier(
            in_features=(glimpses * vision_features, question_features),
            mid_features=1024,
            out_features=config.max_answers,
            count_features=objects + 1,
            drop=0.5,
        )
        self.counter = counting.Counter(objects)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, v, b, q, q_len):
        q = self.text(q, list(q_len.data))

        v = v / (v.norm(p=2, dim=1, keepdim=True) + 1e-12).expand_as(v)

        a = self.attention(v, q) # stacked attention
        v = apply_attention(v, a)

        # this is where the counting component is used
        # pick out the first attention map
        a1 = a[:, 0, :, :].contiguous().view(a.size(0), -1)
        # give it and the bounding boxes to the component
        count = self.counter(b, a1)

        answer = self.classifier(v, q, count)
        # answer = self.classifier(v, q)
        return answer

class Fusion(nn.Module):
    """ Crazy multi-modal fusion: negative squared difference minus relu'd sum
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # self.linearSum = LinearSum(input_dim, output_dim)
        self.mfh = MFH(input_dim, output_dim)

    def forward(self, x, y, feature):
        # f = torch.cuda.FloatTensor([])
        #
        # for i in range(0, config.batch_size-1): # 36 이 x, y의 한계점
        #     fusion_unit = (self.mfh([x[i],y[i]])).view(feature, 1, -1)
        #     fusion_unit = fusion_unit.unsqueeze(0)
        #     # print("\n\nfusion unit unsqueezed : ", fusion_unit.shape)
        #     # fusion_unit = fusion_unit.view(512, 1, -1)
        #     f = torch.cat([f, fusion_unit], 0)

        f = - (x - y)**2 + F.relu(x + y)
        return f
#
# class Fusion(nn.Module):
#     """ Crazy multi-modal fusion: negative squared difference minus relu'd sum
#     """
#     # def __init__(self, input_dim, output_dim):
#     #     super().__init__()
#     #     self.linearSum = LinearSum(input_dim, output_dim)
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x, y):
#         # found through grad student descent ;)
#         # input_dims = [x.shape[-1], y.shape[-1]]
#         # output_dim = config.fusion_output_size
#         # mcb = MCB(input_dims, out_dim)
#         # ls = Mutan(input_dims, out_dim)
#         # return mcb([x,y])
#         # f = torch.cuda.FloatTensor([])
#         # for i in range(config.batch_size):
#         #     fusion_unit = (self.linearSum([x[i],y[i]])).view(512,1, -1)
#         #     # print("\n\nfusion unit : ", fusion_unit.shape)
#         #     fusion_unit = fusion_unit.unsqueeze(0)
#         #     # print("\n\nfusion unit unsqueezed : ", fusion_unit.shape)
#         #     # fusion_unit = fusion_unit.view(512, 1, -1)
#         #     f = torch.cat([f, fusion_unit], 0)
#
#         # fusion = self.linearSum([x,y])
#         f = - (x - y)**2 + F.relu(x + y)
#         # print("Fusion f : ", f.shape)
#         # print("Fusion z : ", z.shape)
#         # z[1] = z[1] + f[1]
#         # z[2] = z[2] + f[2]
#         # z[3] = z[3] + f[3]
#         return f

class Fusion2(nn.Module):
    """ Crazy multi-modal fusion: negative squared difference minus relu'd sum
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linearSum2 = LinearSum2(input_dim, output_dim)

    def forward(self, x, y):
        # found through grad student descent ;)
        # input_dims = [x.shape[-1], y.shape[-1]]
        # output_dim = config.fusion_output_size
        # mcb = MCB(input_dims, out_dim)
        # ls = Mutan(input_dims, out_dim)
        # return mcb([x,y])

        # print("\nfusion2 x :", x.shape)
        # print("\nfusion2 y : ", y.shape)

        # f = torch.cuda.FloatTensor([])
        # for i in range(config.batch_size):
        #     fusion_unit = (self.linearSum2([x[i],y[i]])).view(-1,1, 1024)
        #     print("\n\nfusion unit : ", fusion_unit.shape)
        #     fusion_unit = fusion_unit.unsqueeze(0)
        #     print("\n\nfusion unit unsqueezed : ", fusion_unit.shape)
        #     # fusion_unit = fusion_unit.view(512, 1, -1)
        #     f = torch.cat([f, fusion_unit], 0)

        # fusion = self.linearSum([x,y])
        z = - (x - y)**2 + F.relu(x + y)
        # print("Fusion f : ", f.shape)
        # print("Fusion z : ", z.shape)
        # z[1] = z[1] + f[1]
        # z[2] = z[2] + f[2]
        # z[3] = z[3] + f[3]
        return z

class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, count_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()
        self.fusion = Fusion([mid_features, mid_features], mid_features)
        # self.fusion = Fusion()
        self.lin11 = nn.Linear(in_features[0], mid_features)
        self.lin12 = nn.Linear(in_features[1], mid_features)
        self.lin2 = nn.Linear(mid_features, out_features)
        self.lin_c = nn.Linear(count_features, mid_features)
        self.bn = nn.BatchNorm1d(mid_features)
        self.bn2 = nn.BatchNorm1d(mid_features)

        self.f_feature = mid_features

    def forward(self, x, y, c):
        # x = self.fusion(self.lin11(self.drop(x)), self.lin12(self.drop(y)))
        x = self.fusion(self.lin11(self.drop(x)), self.lin12(self.drop(y)), self.f_feature)
        x = x.contiguous().view(-1, self.f_feature)
        x = x + self.bn2(self.relu(self.lin_c(c)))
        x = self.lin2(self.drop(self.bn(x)))
        return x


class TextProcessor(nn.Module):
    def __init__(self, embedding_tokens, embedding_features, lstm_features, drop=0.0):
        super(TextProcessor, self).__init__()
        self.embedding = nn.Embedding(embedding_tokens, embedding_features, padding_idx=0)
        self.drop = nn.Dropout(drop)
        self.tanh = nn.Tanh()
        self.lstm = nn.GRU(input_size=embedding_features,
                           hidden_size=lstm_features,
                           num_layers=1)
        self.features = lstm_features

        self._init_lstm(self.lstm.weight_ih_l0)
        self._init_lstm(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

        init.xavier_uniform_(self.embedding.weight)

    def _init_lstm(self, weight):
        for w in weight.chunk(3, 0):
            init.xavier_uniform_(w)

    def forward(self, q, q_len):
        embedded = self.embedding(q)
        tanhed = self.tanh(self.drop(embedded))
        packed = pack_padded_sequence(tanhed, q_len, batch_first=True)
        # self.lstm.flatten_parameters()
        _, h = self.lstm(packed)
        return h.squeeze(0)

class Attention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(Attention, self).__init__()
        self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)  # let self.lin take care of bias
        self.q_lin = nn.Linear(q_features, mid_features)
        self.x_conv = nn.Conv2d(mid_features, glimpses, 1)

        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)
        # self.fusion = Fusion([config.att_fusion_input_size, config.att_fusion_input_size], config.att_fusion_output_size)
        self.fusion = Fusion([mid_features, mid_features], mid_features)
        self.f_feature = mid_features
        # self.fusion = Fusion()
    def forward(self, v, q):
        # attention layer 1
        q_in = q
        v = self.v_conv(self.drop(v))
        q = self.q_lin(self.drop(q))
        q = tile_2d_over_nd(q, v)

        # attention layer 1 - fusion
        x = self.fusion(v, q, self.f_feature)
        # print("\n\n____________x shape : {0}_______\n\n".format(x.shape))

        # attention layer 2
        x = self.x_conv(self.drop(x))

        # attention layer 2 - fusion

        return x

# class Attention2(nn.Module):
#     def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
#         super(Attention2, self).__init__()
#         self.v_conv = nn.Conv2d(in_channels=v_features, out_channels=mid_features*2, kernel_size=1, bias=False)  # let self.lin take care of bias
#         self.q_lin = nn.Linear(in_features=q_features, out_features=mid_features*2)
#         self.x_conv = nn.Conv2d(in_channels=mid_features*2, out_channels=mid_features, kernel_size=1)
#         self.scnd_q_lin = nn.Linear(in_features=q_features, out_features=int(mid_features))
#
#         self.scnd_conv = nn.Conv2d(in_channels=mid_features, out_channels=int(mid_features/2), kernel_size=1, bias=False)
#         self.scnd_x_conv = nn.Conv2d(in_channels=int(mid_features), out_channels=glimpses, kernel_size=1)
#
#         self.drop = nn.Dropout(drop)
#         self.relu = nn.ReLU(inplace=True)
#         self.fusion1 = Fusion([mid_features*2, mid_features*2], mid_features*2)
#         self.fusion2 = Fusion([mid_features, mid_features], mid_features)
#         # self.fusion = Fusion()
#         self.f1_feature = mid_features*2
#         self.f2_feature = mid_features
#
#     def forward(self, v, q):
#         # print("I am att 2!! Ya Hoo!!!")
#         # attention layer 1
#         q_in = q
#         v = self.v_conv(self.drop(v))
#         q = self.q_lin(self.drop(q))
#         q = tile_2d_over_nd(q, v)
#         f1 = self.fusion1(v, q, self.f1_feature)
#
#         # attention layer 2
#         x = self.x_conv(self.drop(f1)) # [mid dim * mid dim]
#         f1 = tile_2d_over_nd(self.scnd_q_lin(self.drop(q_in)), x)
#         f2 = self.fusion2(x, f1, self.f2_feature)
#         # attention layer 3
#         # attention layer 3m
#         x2 = self.scnd_x_conv(self.drop(f2))
#         return x2
#
#     # 1.
#     # v
#     # conv -> 2048, 1024
#     # q
#     # lin -> 1024, 1024
#     # q
#     # tile2d -> 2048, 1024
#     #
#     # fusion
#     # conv -> 1024
#     # 512
#     #
#     # -----------
#     #
#     # v
#     # conv -> 512
#     # 256
#     # q
#     # lin -> 1024, 256
#     # q
#     # tile2d -> 2048, 256
#     #
#     # fusion
#     # conv -> 256, 2


def apply_attention(input, attention):
    """ Apply any number of attention maps over the input.
        The attention map has to have the same size in all dimensions except dim=1.
    """
    n, c = input.size()[:2]
    glimpses = attention.size(1)

    # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
    input = input.view(n, c, -1)
    attention = attention.view(n, glimpses, -1)
    s = input.size(2)

    # apply a softmax to each attention map separately
    # since softmax only takes 2d inputs, we have to collapse the first two dimensions together
    # so that each glimpse is normalized separately
    attention = attention.view(n * glimpses, -1)
    attention = F.softmax(attention, dim=1)

    # apply the weighting by creating a new dim to tile both tensors over
    target_size = [n, glimpses, c, s]
    input = input.view(n, 1, c, s).expand(*target_size)
    attention = attention.view(n, glimpses, 1, s).expand(*target_size)
    weighted = input * attention
    # sum over only the spatial dimension
    weighted_mean = weighted.sum(dim=3, keepdim=True)
    # the shape at this point is (n, glimpses, c, 1)
    return weighted_mean.view(n, -1)


def tile_2d_over_nd(feature_vector, feature_map):
    """ Repeat the same feature vector over all spatial positions of a given feature map.
        The feature vector should have the same batch size and number of features as the feature map.
    """
    # n, c, _, _ = feature_vector.size()
    n = feature_vector.size()[0]
    c = feature_vector.size()[1]

    spatial_sizes = feature_map.size()[2:]
    tiled = feature_vector.view(n, c, *([1] * len(spatial_sizes))).expand(n, c, *spatial_sizes)
    return tiled


# class LinearSum(nn.Module):
#
#     def __init__(self,
#             input_dims,
#             output_dim,
#             mm_dim=512,
#             activ_input='relu',
#             activ_output='relu',
#             normalize=False,
#             dropout_input=0.,
#             dropout_pre_lin=0.,
#             dropout_output=0.):
#         super(LinearSum, self).__init__()
#         self.input_dims = input_dims
#         self.output_dim = output_dim
#         self.mm_dim = mm_dim
#         self.activ_input = activ_input
#         self.activ_output = activ_output
#         self.normalize = normalize
#         self.dropout_input = dropout_input
#         self.dropout_pre_lin = dropout_pre_lin
#         self.dropout_output = dropout_output
#         # Modules
#         self.linear0 = nn.Linear(input_dims[0], mm_dim)
#         self.linear1 = nn.Linear(input_dims[1], mm_dim)
#         self.linear_out = nn.Linear(mm_dim, output_dim)
#         self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
#
#     def forward(self, x):
#         # print("\n\natt x[0] shape : ", x[0].shape)
#         # print("x[0][1:] shape : ", x[0][1:].shape)
#         # print("x[0][0] shape : ", x[0][0].shape)
#         # x0 = x[0].contiguous().view(-1, self.input_dims[0])
#         x0 = x[0].contiguous().view(512, 100)
#         # print("x0 : ", x0.shape)
#         x0 = self.linear0(x0)
#
#         # print("\n\natt x[1] shape : ", x[1].shape)
#         # print("x[1][1:] shape : ", x[1][1:].shape)
#         # print("x[1][0] shape : ", x[1][0].shape)
#         # x1 = x[1].contiguous().view(-1, self.input_dims[1])
#         x1 = x[1].contiguous().view(512, 100)
#         # print("x1 : ", x1.shape)
#         x1 = self.linear1(x1)
#
#         if self.activ_input:
#             x0 = getattr(F, self.activ_input)(x0)
#             x1 = getattr(F, self.activ_input)(x1)
#
#         if self.dropout_input > 0:
#             x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
#             x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
#
#         z = x0 + x1
#
#         if self.normalize:
#             z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
#             z = F.normalize(z,p=2)
#
#         if self.dropout_pre_lin > 0:
#             z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)
#
#         z = self.linear_out(z)
#
#         if self.activ_output:
#             z = getattr(F, self.activ_output)(z)
#
#         if self.dropout_output > 0:
#             z = F.dropout(z, p=self.dropout_output, training=self.training)
#
#         # z = z.contiguous().view(2, 512, 1, 1)
#         # print("LS z : ", z.shape)
#         return z


class MFH(nn.Module):

    def __init__(self,
            input_dims,
            output_dim,
            mm_dim=800,
            factor=2,
            activ_input='relu',
            activ_output='relu',
            normalize=False,
            dropout_input=0.,
            dropout_pre_lin=0.,
            dropout_output=0.):
        super(MFH, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.factor = factor
        self.activ_input = activ_input
        self.activ_output = activ_output
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        # Modules
        self.linear0_0 = nn.Linear(input_dims[0], mm_dim*factor)
        self.linear1_0 = nn.Linear(input_dims[1], mm_dim*factor)
        self.linear0_1 = nn.Linear(input_dims[0], mm_dim*factor)
        self.linear1_1 = nn.Linear(input_dims[1], mm_dim*factor)
        self.linear_out = nn.Linear(mm_dim*2, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        # print("mfh x[0] : ", x[0].shape)
        # print("mfh x[1] : ", x[1].shape)
        x[0] = x[0].contiguous().view(-1, self.input_dims[0])
        x[1] = x[1].contiguous().view(-1, self.input_dims[1])
        x0 = self.linear0_0(x[0])
        x1 = self.linear1_0(x[1])

        if self.activ_input:
            x0 = getattr(F, self.activ_input)(x0)
            x1 = getattr(F, self.activ_input)(x1)

        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)

        z_0_skip = x0 * x1

        if self.dropout_pre_lin:
            z_0_skip = F.dropout(z_0_skip, p=self.dropout_pre_lin, training=self.training)

        z_0 = z_0_skip.view(z_0_skip.size(0), self.mm_dim, self.factor)
        z_0 = z_0.sum(2)

        if self.normalize:
            z_0 = torch.sqrt(F.relu(z_0)) - torch.sqrt(F.relu(-z_0))
            z_0 = F.normalize(z_0, p=2)

        #
        x0 = self.linear0_1(x[0])
        x1 = self.linear1_1(x[1])

        if self.activ_input:
            x0 = getattr(F, self.activ_input)(x0)
            x1 = getattr(F, self.activ_input)(x1)

        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)

        z_1 = x0 * x1 * z_0_skip

        if self.dropout_pre_lin > 0:
            z_1 = F.dropout(z_1, p=self.dropout_pre_lin, training=self.training)

        z_1 = z_1.view(z_1.size(0), self.mm_dim, self.factor)
        z_1 = z_1.sum(2)

        if self.normalize:
            z_1 = torch.sqrt(F.relu(z_1)) - torch.sqrt(F.relu(-z_1))
            z_1 = F.normalize(z_1, p=2)

        #
        cat_dim = z_0.dim() - 1
        z = torch.cat([z_0, z_1], cat_dim)
        z = self.linear_out(z)

        if self.activ_output:
            z = getattr(F, self.activ_output)(z)

        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


class LinearSum(nn.Module):

    def __init__(self,
            input_dims,
            output_dim,
            mm_dim=1200,
            activ_input='relu',
            activ_output='relu',
            normalize=False,
            dropout_input=0.,
            dropout_pre_lin=0.,
            dropout_output=0.):
        super(LinearSum, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.activ_input = activ_input
        self.activ_output = activ_output
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        self.drop = nn.Dropout(0.0)
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        self.linear1 = nn.Linear(input_dims[1], mm_dim)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x0 = self.linear0(self.drop(x[0]))
        x1 = self.linear1(self.drop(x[1]))

        if self.activ_input:
            x0 = getattr(F, self.activ_input)(x0)
            x1 = getattr(F, self.activ_input)(x1)

        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)

        z = x0 + x1

        if self.normalize:
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z,p=2)

        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)

        z = self.linear_out(z)

        if self.activ_output:
            z = getattr(F, self.activ_output)(z)

        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z

class LinearSum2(nn.Module):

    def __init__(self,
            input_dims,
            output_dim,
            mm_dim=1,
            activ_input='relu',
            activ_output='relu',
            normalize=False,
            dropout_input=0.,
            dropout_pre_lin=0.,
            dropout_output=0.):
        super(LinearSum2, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.activ_input = activ_input
        self.activ_output = activ_output
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        self.linear1 = nn.Linear(input_dims[1], mm_dim)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        # print("\n\nclassi x[0] shape : ", x[0].shape)
        # print("x[0][1:] shape : ", x[0][1:].shape)
        # print("x[0][0] shape : ", x[0][0].shape
        # x0 = x[0].contiguous().view(-1, self.input_dims[0])
        x0 = x[0].contiguous().view(1024, -1)
        # print("x0 : ", x0.shape)
        x0 = self.linear0(x0)

        # print("\n\nclassi x[1] shape : ", x[1].shape)
        # print("x[1][1:] shape : ", x[1][1:].shape)
        # print("x[1][0] shape : ", x[1][0].shape)
        # x1 = x[1].contiguous().view(-1, self.input_dims[1])
        x1 = x[1].contiguous().view(1024, -1)
        # print("x1 : ", x1.shape)
        x1 = self.linear1(x1)

        if self.activ_input:
            x0 = getattr(F, self.activ_input)(x0)
            x1 = getattr(F, self.activ_input)(x1)

        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)

        z = x0 + x1

        if self.normalize:
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z,p=2)

        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)

        z = self.linear_out(z)

        if self.activ_output:
            z = getattr(F, self.activ_output)(z)

        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)

        # z = z.contiguous().view(2, 512, 1, 1)
        # print("LS z : ", z.shape)
        return z