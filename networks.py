import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter


class DenseBlock(nn.Module):
    def __init__(self, input_num, num1, num2, rate, drop_out):
        super(DenseBlock, self).__init__()
        self.conv1x1 = nn.Conv1d(in_channels=input_num, out_channels=num1, kernel_size=1)
        self.ConvGN = nn.BatchNorm1d(num1)
        self.relu1 = nn.ReLU(inplace=True)
        self.dilaconv = nn.Conv1d(in_channels=num1, out_channels=num2, kernel_size=3, padding=1 * rate, dilation=rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.ConvGN(self.conv1x1(x))
        x = self.relu1(x)
        x = self.dilaconv(x)
        x = self.relu2(x)
        x = self.drop(x)
        return x

class DenseAPP(nn.Module):
    def __init__(self, num_channels=2048, channels1=512, channels2=256):
        super(DenseAPP, self).__init__()
        self.drop_out = 0.1
        self.channels1 = channels1
        self.channels2 = channels2
        self.num_channels = num_channels
        self.aspp3 = DenseBlock(self.num_channels, num1=self.channels1, num2=self.channels2, rate=3,
                                drop_out=self.drop_out)
        self.aspp6 = DenseBlock(self.num_channels + self.channels2 * 1, num1=self.channels1, num2=self.channels2,
                                rate=6,
                                drop_out=self.drop_out)
        self.aspp12 = DenseBlock(self.num_channels + self.channels2 * 2, num1=self.channels1, num2=self.channels2,
                                 rate=12,
                                 drop_out=self.drop_out)
        self.aspp18 = DenseBlock(self.num_channels + self.channels2 * 3, num1=self.channels1, num2=self.channels2,
                                 rate=18,
                                 drop_out=self.drop_out)
        self.aspp24 = DenseBlock(self.num_channels + self.channels2 * 4, num1=self.channels1, num2=self.channels2,
                                 rate=24,
                                 drop_out=self.drop_out)
        self.conv1x1 = nn.Conv1d(in_channels=5 * self.channels2, out_channels=channels1, kernel_size=5)
        self.ConvGN = nn.BatchNorm1d(channels1)

    def forward(self, feature): # (32, 2018, 1728)
        aspp3 = self.aspp3(feature) # (32, 256, 1728)
        feature = torch.cat((aspp3, feature), dim=1) # (32, 2304, 1728)
        aspp6 = self.aspp6(feature) # (32, 256, 1728)
        feature = torch.cat((aspp6, feature), dim=1) # (32, 2560, 1728)
        aspp12 = self.aspp12(feature) # (32, 256, 1728)
        feature = torch.cat((aspp12, feature), dim=1) # (32, 2816, 1728)
        aspp18 = self.aspp18(feature) # (32, 256, 1728)
        feature = torch.cat((aspp18, feature), dim=1) # (32, 3072, 1728)
        aspp24 = self.aspp24(feature)

        x = torch.cat((aspp3, aspp6, aspp12, aspp18, aspp24), dim=1) # (32, 1280, 1728)
        out = self.ConvGN(self.conv1x1(x))
        return out # (32, 512, 1728)

class MHA(nn.Module):  # multi-head attention
    # 用于指定每个注意力头中局部注意力的邻域大小
    neigh_k = list(range(3, 21, 2))

    def __init__(self, num_heads, hidden_dim):
        """
        :param num_heads: 多头注意力的头数
        :param hidden_dim: 隐藏层的维度
        """
        super(MHA, self).__init__()
        self.num_heads = num_heads
        # avgpool, maxpool: 分别是自适应平均池化层和自适应最大池化层，用于对输入的序列进行池化操作，将序列的时间维度降低为1。
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        # multi_head: 一个包含多个 CPAM（Channel- and Position-wise Attention Module）模块的列表，每个模块对应一个注意力头。
        # CPAM 是一个自定义的注意力模块，用于计算通道和位置上的注意力权重。
        self.multi_head = nn.ModuleList([
            CPAM(self.neigh_k[i])
            for i in range(num_heads)
        ])
        # self.high_lateral_attn = nn.Sequential(nn.Linear(num_heads * hidden_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,num_heads))
        # 用于对多头注意力进行加权的可学习参数 weight_var
        self.weight_var = Parameter(torch.ones(num_heads))

    def forward(self, x):
        # 1. 池化操作: 通过自适应平均池化和自适应最大池化，得到序列的平均池化特征和最大池化特征
        max_pool = self.maxpool(x)
        avg_pool = self.avgpool(x)
        # 2. 多头注意力计算
        # 对每个注意力头，通过 CPAM 模块计算通道和位置上的注意力权重，并对输入进行加权。
        # CPAM 模块接收平均池化和最大池化的特征作为输入，并输出一个权重张量，表示了不同通道和位置的重要性。这些权重张量将被用来对输入序列进行加权操作
        pool_feats = []
        for id, head in enumerate(self.multi_head):
            weight = head(max_pool, avg_pool)
            self_attn = x * weight
            pool_feats.append(torch.max(self_attn, dim=2)[0])

        # concat_pool_features = torch.cat(pool_feats, dim=1)
        # fusion_weights = self.high_lateral_attn(concat_pool_features)
        # fusion_weights = torch.sigmoid(fusion_weights)

        # high_pool_fusion = 0
        # for i in range(self.num_heads):
        #     high_pool_fusion += torch.unsqueeze(fusion_weights[:,i], dim=1) * pool_feats[i]

        # 3. 加权汇总
        # 将每个头计算得到的加权特征进行加权求和，得到最终的输出特征。
        # 在加权过程中，使用了 weight_var 参数对多个头的注意力结果进行加权。这样可以根据每个头的重要性动态地调整加权系数，以更好地利用不同头的信息。
        weight_var = [torch.exp(self.weight_var[i]) / torch.sum(torch.exp(self.weight_var)) for i in
                      range(self.num_heads)]
        # 4. 返回结果
        # 将最终的输出特征作为结果返回。这个特征代表了输入序列在多头注意力机制下的表示结果，捕捉了序列中不同通道和位置的重要信息。
        high_pool_fusion = 0
        for i in range(self.num_heads):
            high_pool_fusion += weight_var[i] * pool_feats[i]

        return high_pool_fusion


class CPAM(nn.Module):
    def __init__(self, k, pool_types=['avg', 'max']):
        """
        :param k: 用于指定局部注意力的邻域大小。
        :param pool_types: 用于指定池化类型，可以是平均池化（'avg'）或者最大池化（'max'）。
        """
        super(CPAM, self).__init__()
        self.pool_types = pool_types
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

    def forward(self, max_pool, avg_pool):
        channel_att_sum = 0.
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                channel_att_raw = self.conv(avg_pool.transpose(1, 2)).transpose(1, 2)
            elif pool_type == 'max':
                channel_att_raw = self.conv(max_pool.transpose(1, 2)).transpose(1, 2)

            channel_att_sum += channel_att_raw

        scale = torch.sigmoid(channel_att_sum)

        return scale

# AdaptiveCosineClassifier
class AdaCosClassifier(nn.Module):
    def __init__(self, n_feat=1024, num_classes=256):
        super(AdaCosClassifier, self).__init__()
        self.num_classes = num_classes
        self.scale = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        weight = torch.FloatTensor(n_feat, num_classes).normal_(
                    0.0, np.sqrt(2.0 / num_classes))
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x):
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1, eps=1e-12)
        weight = torch.nn.functional.normalize(self.weight, p=2, dim=0, eps=1e-12)
        cos_dist = x_norm @ weight
        scores = self.scale * cos_dist
        return scores

    def extra_repr(self):
        s = 'CosineClassifier: input_channels={}, num_classes={}; learned_scale: {}'.format(
            self.weight.shape[0], self.weight.shape[1], self.scale.item())
        return s

# AdaptiveCosineFeatureExtractor
class AdaCosFeaExtractor(nn.Module):
    def __init__(self, n_feat, num_classes, kernel_size=1):
        super(AdaCosFeaExtractor, self).__init__()
        self.scale = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        weight = torch.FloatTensor(num_classes, n_feat, kernel_size).normal_(
                    0.0, np.sqrt(2.0/num_classes))
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x):
        x_normalized = torch.nn.functional.normalize(
            x, p=2, dim=1, eps=1e-12)
        weight = torch.nn.functional.normalize(
            self.weight, p=2, dim=1, eps=1e-12)

        cos_dist = torch.nn.functional.conv1d(x_normalized, weight)
        scores = self.scale * cos_dist
        return scores

    def extra_repr(self):
        s = 'CosineConv: num_inputs={}, num_classes={}, kernel_size=1; scale_value: {}'.format(
            self.weight.shape[0], self.weight.shape[1], self.scale.item())
        return s


class SLExpert(nn.Module):
    def __init__(self,  input_dim=64, num_classes=256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, input_dim // 2, kernel_size=6, padding=2),
            nn.BatchNorm1d(input_dim // 2),
            nn.ReLU(inplace=True)
            )
        self.pool1 = nn.AdaptiveMaxPool1d(1)

        self.conv2 = nn.Sequential(
            nn.Conv1d(input_dim, input_dim // 2, kernel_size=12, padding=5),
            nn.BatchNorm1d(input_dim // 2),
            nn.ReLU(inplace=True)
            )
        self.pool2 = nn.AdaptiveMaxPool1d(1)
        self.cos = AdaCosClassifier(n_feat=input_dim, num_classes=num_classes)
        self.shortcut = nn.Sequential(nn.Conv1d(input_dim, input_dim // 2, kernel_size=1),
                                      nn.BatchNorm1d(input_dim // 2),
                                      )


    def forward(self, x):

        # 计算残差
        identity = self.shortcut(x)

        # 通过第一个卷积分支
        x1 = self.conv1(x)
        x1 = x1 + identity[:, :, :x1.size(2)]  # 直接相加，避免重复切片
        x1 = self.pool1(x1)

        # 通过第二个卷积分支
        x2 = self.conv2(x)
        x2 = x2 + identity[:, :, :x2.size(2)]  # 直接相加，避免重复切片
        x2 = self.pool2(x2)

        # 拼接并分类
        output = torch.cat([x1, x2], dim=1)
        output = torch.flatten(output, 1)
        output = self.cos(output)
        return output



class DLExpert(nn.Module):
    def __init__(self, num_head=8, input_dim=64, hidden_dim=256, num_classes=256):
        super().__init__()
        self.cosconv = AdaCosFeaExtractor(n_feat=input_dim, num_classes=input_dim)
        self.emb = nn.Sequential(nn.Conv1d(input_dim, 2048, kernel_size=5, padding=0,), nn.BatchNorm1d(2048))
        self.dense = DenseAPP(num_channels=2048, channels1=512, channels2=256)
        self.multi_head = MHA(num_head, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.cosconv(x)
        x = self.emb(x)
        dense = self.dense(x)
        output = self.multi_head(dense)
        output = self.fc_out(output)
        return output



class NoisyTopkRouter(nn.Module):
    def __init__(self, input_dim, num_experts, top_k, type='noisy'):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.type = type
        self.topkroute_linear = nn.Linear(input_dim, num_experts)
        self.noise_linear = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        logits = self.topkroute_linear(x)
        noise_logits = self.noise_linear(x)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        if self.type == 'avg':
            sparse_logits = zeros.scatter(-1, indices, 1.0)
        else:
            sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices

class MERGO(nn.Module):
    def __init__(self, num_classes=256, top_k=4, model_type='all'):
        super(MERGO, self).__init__()
        input_dim1 = 1024
        input_dim2 = 1280
        hidden_dim = 512
        num_experts = 4
        num_head = 4
        self.model_type = model_type

        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

        if self.model_type == 'all':
            num_experts = 4
            self.expert1 = DLExpert(num_head=num_head, input_dim=input_dim1, hidden_dim=hidden_dim,
                                    num_classes=num_classes)
            self.expert2 = SLExpert(input_dim=input_dim1, num_classes=num_classes)
            self.expert3 = DLExpert(num_head=num_head, input_dim=input_dim2, hidden_dim=hidden_dim,
                                    num_classes=num_classes)
            self.expert4 = SLExpert(input_dim=input_dim2, num_classes=num_classes)

        elif self.model_type == 'dl':
            num_experts = 2
            top_k = 2
            self.expert1 = DLExpert(num_head=num_head, input_dim=input_dim1, hidden_dim=hidden_dim,
                                    num_classes=num_classes)
            self.expert2 = DLExpert(num_head=num_head, input_dim=input_dim2, hidden_dim=hidden_dim,
                                    num_classes=num_classes)

        elif self.model_type == 'sl':
            num_experts = 2
            top_k = 2
            self.expert1 = SLExpert(input_dim=input_dim1, num_classes=num_classes)
            self.expert2 = SLExpert(input_dim=input_dim2, num_classes=num_classes)

        else:
            raise ValueError("Invalid model_type. Choose from 'all', 'sl', 'dl'.")


        self.router = NoisyTopkRouter(input_dim1 + input_dim2, num_experts=num_experts, top_k=top_k)

        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, data):
        # Compute 1D convolutional part and apply global max pooling
        x1 = data.x1
        x2 = data.x2

        x2 = x2.transpose(1, 2)  # (batch_size, emb_dim, seq_len) -> (batch_size, seq_len, emb_dim)
        x2 = self.pool(x2)
        x2 = x2.transpose(1, 2)

        if self.model_type == 'all':
            output1 = self.expert1(x1)
            output2 = self.expert2(x1)
            output3 = self.expert3(x2)
            output4 = self.expert4(x2)
            outputs = torch.stack([output1, output2, output3, output4], dim=2)

        elif self.model_type == 'sl':
            output1 = self.expert1(x1)
            output2 = self.expert2(x2)
            outputs = torch.stack([output1, output2], dim=2)

        else:
            output1 = self.expert1(x1)
            output2 = self.expert2(x2)
            outputs = torch.stack([output1, output2], dim=2)


        gating_inputs = torch.cat([x1.mean(dim=-1), x2.mean(dim=-1)], dim=-1)
        gating_weights, indices = self.router(gating_inputs)

        gating_weights = gating_weights.unsqueeze(1).expand_as(outputs)

        output = torch.sum(outputs * gating_weights, dim=2)
        # 确保 [0,1]
        output[output != output] = 0
        return output

# from torch_geometric.data import Data
# if __name__ == '__main__':
#     x1 = torch.randn(32, 1024, 1777)
#     x2 = torch.randn(32, 2560, 1777)
#     data = Data(x1=x1, x2=x2)
#     model = MERGO(num_classes=3992, top_k=2, model_type='dl')
#     print(model)
#     y = model(data)
#     print(y.shape)