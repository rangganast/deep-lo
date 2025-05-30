import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
import pointops

class LayerNorm1d(nn.BatchNorm1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (
            super()
            .forward(input.transpose(1, 2).contiguous())
            .transpose(1, 2)
            .contiguous()
        )

class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(
            nn.Linear(3, 3),
            LayerNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_planes),
        )
        self.linear_w = nn.Sequential(
            LayerNorm1d(mid_planes),
            nn.ReLU(inplace=True),
            nn.Linear(mid_planes, out_planes // share_planes),
            LayerNorm1d(out_planes // share_planes),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // share_planes, out_planes // share_planes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        x_k, idx = pointops.knn_query_and_group(
            x_k, p, o, new_xyz=p, new_offset=o, nsample=self.nsample, with_xyz=True
        )
        x_v, _ = pointops.knn_query_and_group(
            x_v,
            p,
            o,
            new_xyz=p,
            new_offset=o,
            idx=idx,
            nsample=self.nsample,
            with_xyz=False,
        )
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        p_r = self.linear_p(p_r)
        r_qk = (
            x_k
            - x_q.unsqueeze(1)
            + einops.reduce(
                p_r, "n ns (i j) -> n ns j", reduction="sum", j=self.mid_planes
            )
        )
        w = self.linear_w(r_qk)  # (n, nsample, c)
        w = self.softmax(w)
        x = torch.einsum(
            "n t s i, n t i -> n s i",
            einops.rearrange(x_v + p_r, "n ns (s i) -> n ns s i", s=self.share_planes),
            w,
        )
        x = einops.rearrange(x, "n s i -> n (s i)")
        return x
    
class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.farthest_point_sampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)
            x, _ = pointops.knn_query_and_group(
                x,
                p,
                offset=o,
                new_xyz=n_p,
                new_offset=n_o,
                nsample=self.nsample,
                with_xyz=True,
            )
            x = self.relu(
                self.bn(self.linear(x).transpose(1, 2).contiguous())
            )  # (m, c, nsample)
            x = self.pool(x).squeeze(-1)  # (m, c)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
        return [p, x, o]
    
class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(
                nn.Linear(2 * in_planes, in_planes),
                nn.BatchNorm1d(in_planes),
                nn.ReLU(inplace=True),
            )
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True)
            )
        else:
            self.linear1 = nn.Sequential(
                nn.Linear(out_planes, out_planes),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True),
            )
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, out_planes),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True),
            )
            self.linear_flow = nn.Sequential(
                nn.Linear(3, out_planes),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True),
            )

    def forward(self, pxo1, pxo2=None, pxo_flow=None):
        if pxo2 is None and pxo_flow is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i - 1], o[i], o[i] - o[i - 1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat(
                    (x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1
                )
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            p1, x1, o1 = pxo1
            p2, x2, o2 = pxo2
            p_flow, x_flow, o_flow = pxo_flow

            x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
            x_flow = pointops.interpolation(p_flow, p1, self.linear_flow(x_flow), o_flow, o1)
            x = x + x_flow
        return x

class PredictFlow(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.farthest_point_sampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)
            x, _ = pointops.knn_query_and_group(
                x,
                p,
                offset=o,
                new_xyz=n_p,
                new_offset=n_o,
                nsample=self.nsample,
                with_xyz=True,
            )
            x = self.relu(
                self.bn(self.linear(x).transpose(1, 2).contiguous())
            )  # (m, c, nsample)
            x = self.pool(x).squeeze(-1)  # (m, c)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
        return [p, x, o]

class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(Bottleneck, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]
    
class FlowEmbedding(nn.Module):
    def __init__(self, radius, nsample, in_channel, mlp, pooling='max', corr_func='concat', knn = True):
        super(FlowEmbedding, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.knn = knn
        self.pooling = pooling
        self.corr_func = corr_func
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if corr_func == 'concat':
            last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1, bias=False))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, pos1, pos2, feat1, feat2, offset1, offset2):
        """
        pos1:       (N, 3)
        pos2:       (N, 3)
        feat1:      (N, C)
        feat2:      (N, C)
        offset1:    (N)
        offset2:    (N)in_planes
        """

        if self.knn:
            idx, _ = pointops.knn_query(self.nsample, pos2, offset2, pos1, offset1)
        # else:
        #     # If the ball neighborhood points are less than nsample,
        #     # than use the knn neighborhood points
        #     idx, cnt = pointops.ball_query(self.radius, self.nsample, pos2_t, pos1_t)
        #     # 利用knn取最近的那些点
        #     _, idx_knn = pointops.knn_query(self.nsample, pos1_t, pos2_t)
        #     cnt = cnt.view(B, -1, 1).repeat(1, 1, self.nsample)
        #     idx = idx_knn[cnt > (self.nsample-1)]
        
        pos_diff, feat2_grouped = pointops.grouping(idx, feat2, pos2, new_xyz=pos1, with_xyz=True, coor=True)
        if self.corr_func == 'concat':
            feat1 = feat1.view(feat1.size(0), self.nsample, -1)
            new_feat = torch.cat([pos_diff, feat2_grouped, feat1], dim=-1)

        new_feat = torch.einsum('bnc -> bcn', new_feat)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_feat = F.leaky_relu(bn(conv(new_feat)))

        new_feat = torch.max(new_feat, dim=-1)[0]
        return pos1, new_feat, offset1
    
class PointTransformerLO(nn.Module):
    def __init__(self, block, blocks, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.in_planes, planes = in_channels, [32, 64, 128, 256]
        # self.in_planes, planes = in_channels, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [8, 8, 12, 16], [16, 16, 8, 4]
        self.enc1 = self._make_enc(
            block,
            planes[0],
            blocks[0],
            share_planes,
            stride=stride[0],
            nsample=nsample[0],
        )  # N/8
        self.enc2 = self._make_enc(
            block,
            planes[1],
            blocks[1],
            share_planes,
            stride=stride[1],
            nsample=nsample[1],
        )  # N/4
        self.enc3 = self._make_enc(
            block,
            planes[2],
            blocks[2],
            share_planes,
            stride=stride[2],
            nsample=nsample[2],
        )  # N/16
        self.enc4 = self._make_enc(
            block,
            planes[3],
            blocks[3],
            share_planes,
            stride=stride[3],
            nsample=nsample[3],
        )  # N/64

        self.corr4 = FlowEmbedding(
            radius=10.0,
            nsample=nsample[3],
            in_channel=323,
            # in_channel=512+35,
            mlp=[256, 256, 256],
            pooling='max',
            corr_func='concat'
        )

        self.corr3 = FlowEmbedding(
            radius=10.0,
            nsample=nsample[2],
            # in_channel=256+11,
            in_channel=147,
            mlp=[128, 128, 128],
            pooling='max',
            corr_func='concat'
        )

        self.corr2 = FlowEmbedding(
            radius=10.0,
            nsample=nsample[1],
            # in_channel=128+11,
            in_channel=71,
            mlp=[64, 64, 64],
            pooling='max',
            corr_func='concat'
        )

        self.corr1 = FlowEmbedding(
            radius=10.0,
            nsample=nsample[0],
            # in_channel=64+7,
            in_channel=37,
            mlp=[32, 32, 32],
            pooling='max',
            corr_func='concat'
        )

        # deconv
        self.dec4 = self._make_dec(block, planes[3], 1, share_planes, nsample=nsample[3], is_head=True)
        self.dec4_flow = self._make_dec(block, planes[3], 1, share_planes, nsample=nsample[3])
        
        self.dec3 = self._make_dec(block, planes[2], 1, share_planes, nsample=nsample[2])
        self.dec2 = self._make_dec(block, planes[1], 1, share_planes, nsample=nsample[1])
        self.dec1 = self._make_dec(block, planes[0], 1, share_planes, nsample=nsample[0]) 

        # flow
        self.flow_predict4 = PredictFlow(in_planes=256, out_planes=3, stride=1, nsample=nsample[3])
        self.flow_predict3 = PredictFlow(in_planes=128, out_planes=3, stride=1, nsample=nsample[2])
        self.flow_predict2 = PredictFlow(in_planes=64, out_planes=3, stride=1, nsample=nsample[1])
        self.flow_predict1 = PredictFlow(in_planes=32, out_planes=3, stride=1, nsample=nsample[0])

        # fcn
        self.fcn4 = nn.Conv1d(36, 32, 1)
        self.fcn3 = nn.Conv1d(585, 32, 1)
        self.fcn2 = nn.Conv1d(7029, 32, 1)
        self.fcn1 = nn.Conv1d(56250, 32, 1)

        # pred
        self.q_predict4 = nn.Linear(96, 4)
        self.t_predict4 = nn.Linear(96, 3)
        self.q_predict3 = nn.Linear(96, 4)
        self.t_predict3 = nn.Linear(96, 3)
        self.q_predict2 = nn.Linear(96, 4)
        self.t_predict2 = nn.Linear(96, 3)
        self.q_predict1 = nn.Linear(96, 4)
        self.t_predict1 = nn.Linear(96, 3)

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = [
            TransitionDown(self.in_planes, planes * block.expansion, stride, nsample)
        ]
        self.in_planes = planes * block.expansion
        for _ in range(blocks):
            layers.append(
                block(self.in_planes, self.in_planes, share_planes, nsample=nsample)
            )
        return nn.Sequential(*layers)     

    def _make_dec(
        self, block, planes, blocks, share_planes=8, nsample=16, is_head=False
    ):
        layers = [
            TransitionUp(self.in_planes, None if is_head else planes * block.expansion)
        ]
        self.in_planes = planes * block.expansion
        for _ in range(blocks):
            layers.append(
                block(self.in_planes, self.in_planes, share_planes, nsample=nsample)
            )
        return nn.Sequential(*layers)   
    
    def forward(self, data_prev, data):
        p_prev, x_prev, o_prev = data_prev
        p0_prev = p_prev
        x0_prev = x_prev
        o0_prev = o_prev

        p, x, o = data
        p0 = p
        x0 = x
        o0 = o

        # encode prev
        p1_prev_enc, x1_prev_enc, o1_prev_enc = self.enc1([p0_prev, x0_prev, o0_prev])
        p2_prev_enc, x2_prev_enc, o2_prev_enc = self.enc2([p1_prev_enc, x1_prev_enc, o1_prev_enc])
        p3_prev_enc, x3_prev_enc, o3_prev_enc = self.enc3([p2_prev_enc, x2_prev_enc, o2_prev_enc])
        p4_prev_enc, x4_prev_enc, o4_prev_enc = self.enc4([p3_prev_enc, x3_prev_enc, o3_prev_enc])

        # encode curr
        p1_enc, x1_enc, o1_enc = self.enc1([p0, x0, o0])
        p2_enc, x2_enc, o2_enc = self.enc2([p1_enc, x1_enc, o1_enc])
        p3_enc, x3_enc, o3_enc = self.enc3([p2_enc, x2_enc, o2_enc])
        p4_enc, x4_enc, o4_enc = self.enc4([p3_enc, x3_enc, o3_enc])

        # cost volume 4
        p4_corr, x4_corr, o4_corr = self.corr4(p4_prev_enc, p4_enc, x4_prev_enc, x4_enc, o4_prev_enc, o4_enc)
        # cost volume deconv 4
        x4_corr_deconv = self.dec4[1:]([p4_corr, self.dec4[0]([p4_corr, x4_corr, o4_corr]), o4_corr])[1]
        
        # flow 4
        x4_flow = torch.cat([x4_enc, x4_corr, x4_corr_deconv], dim=0)
        p4_flow = torch.cat([p4_enc, p4_corr, p4_enc], dim=0)
        o4_flow = torch.round(o0 * (x4_flow.shape[0]/o0[-1])).to(torch.int32)
        p4_flow, x4_flow, o4_flow = self.flow_predict4((p4_flow, x4_flow, o4_flow))
        
        print(x4_enc.shape)
        print(x4_corr_deconv.shape)
        print(x4_flow.shape)
        print(x3_prev_enc.shape)
        
        # pose estimation 4
        x4 = [x4_flow[o4_flow[i]:o4_flow[i+1]] for i in range(len(o4_flow) - 1)]
        x4 = torch.stack(x4)
        x4 = self.fcn4(x4).view(x4.size(0), -1)
        q4 = self.q_predict4(x4)
        t4 = self.t_predict4(x4)

        # cost volume 3
        p3_corr, x3_corr, _ = self.corr3(p3_prev_enc, p3_enc, x3_prev_enc, x3_enc, o3_prev_enc, o4_enc)
        # cost volume deconv 3
        x3_corr_deconv = self.dec3[1:]([p3_enc, self.dec3[0]([p3_enc, x3_corr, o3_enc], [p4_enc, x4_enc, o4_enc], [p4_flow, x4_flow, o4_flow]), o3_enc])[1]
        # flow 3
        x3_flow = torch.cat([x3_enc, x3_corr, x3_corr_deconv], dim=0)
        p3_flow = torch.cat([p3_enc, p3_corr, p3_enc], dim=0)
        o3_flow = torch.round(o0 * (x3_flow.shape[0]/o0[-1])).to(torch.int32)
        p3_flow, x3_flow, o3_flow = self.flow_predict3((p3_flow, x3_flow, o3_flow))
        # pose estimation 3
        x3 = [x3_flow[o3_flow[i]:o3_flow[i+1]] for i in range(len(o3_flow) - 1)]
        x3 = torch.stack(x3)
        x3 = self.fcn3(x3).view(x3.size(0), -1)
        q3 = self.q_predict3(x3)
        t3 = self.t_predict3(x3)

        # cost volume 2
        p2_corr, x2_corr, _ = self.corr2(p2_prev_enc, p2_enc, x2_prev_enc, x2_enc, o2_prev_enc, o2_enc)
        # cost volume deconv 2
        x2_corr_deconv = self.dec2[1:]([p2_enc, self.dec2[0]([p2_enc, x2_corr, o2_enc], [p3_enc, x3_enc, o3_enc], [p3_flow, x3_flow, o3_flow]), o2_enc])[1]
        # flow 2
        x2_flow = torch.cat([x2_enc, x2_corr, x2_corr_deconv], dim=0)
        p2_flow = torch.cat([p2_enc, p2_corr, p2_enc], dim=0)
        o2_flow = torch.round(o0 * (x2_flow.shape[0]/o0[-1])).to(torch.int32)
        p2_flow, x2_flow, o2_flow = self.flow_predict2((p2_flow, x2_flow, o2_flow))
        # pose estimation 2
        x2 = [x2_flow[o2_flow[i]:o2_flow[i+1]] for i in range(len(o2_flow) - 1)]
        x2 = torch.stack(x2)
        x2 = self.fcn2(x2).view(x2.size(0), -1)
        q2 = self.q_predict2(x2)
        t2 = self.t_predict2(x2)

        # cost volume 1
        p1_corr, x1_corr, _ = self.corr1(p1_prev_enc, p1_enc, x1_prev_enc, x1_enc, o1_prev_enc, o1_enc)
        # cost volume deconv 1
        x1_corr_deconv = self.dec1[1:]([p1_enc, self.dec1[0]([p1_enc, x1_corr, o1_enc], [p2_enc, x2_enc, o2_enc], [p2_flow, x2_flow, o2_flow]), o1_enc])[1]
        # flow 1
        x1_flow = torch.cat([x1_enc, x1_corr, x1_corr_deconv], dim=0)
        p1_flow = torch.cat([p1_enc, p1_corr, p1_enc], dim=0)
        o1_flow = torch.round(o0 * (x1_flow.shape[0]/o0[-1])).to(torch.int32)
        p1_flow, x1_flow, o1_flow = self.flow_predict1((p1_flow, x1_flow, o1_flow))
        # pose estimation 1
        x1 = [x1_flow[o1_flow[i]:o1_flow[i+1]] for i in range(len(o1_flow) - 1)]
        x1 = torch.stack(x1)
        x1 = self.fcn1(x1).view(x1.size(0), -1)
        q1 = self.q_predict1(x1)
        t1 = self.t_predict1(x1)        

        return q4, t4, q3, t3, q2, t2, q1, t1

class PointTransformerLO38(PointTransformerLO):
    def __init__(self, **kwargs):
        super(PointTransformerLO38, self).__init__(
            Bottleneck, [1, 1, 1, 1], **kwargs
        )

if __name__ == "__main__":
    import numpy as np

    DEVICE = "cuda:0"

    def load_pcd(scan):
        pcd = np.fromfile(scan, dtype=np.float32)
        return pcd.reshape((-1, 3))

    def load_pose():
        pose = np.array([1.000000e+00, 9.043680e-12, 2.326809e-11, 5.551115e-17, 9.043683e-12, 1.000000e+00, 2.392370e-10, 3.330669e-16, 2.326810e-11, 2.392370e-10, 9.999999e-01, -4.440892e-16])
        pose = pose.reshape(3, 4)

        return pose
    
    pcd1 = load_pcd("/home/wicom/lidar-odometry/point-transformer/sample_data/000001.bin")
    pcd2 = load_pcd("/home/wicom/lidar-odometry/point-transformer/sample_data/000002.bin")
    pcd3 = load_pcd("/home/wicom/lidar-odometry/point-transformer/sample_data/000003.bin")
    pcd4 = load_pcd("/home/wicom/lidar-odometry/point-transformer/sample_data/000004.bin")
    pcd5 = load_pcd("/home/wicom/lidar-odometry/point-transformer/sample_data/000005.bin")
    
    pcd_prev_list = [pcd1, pcd2, pcd3, pcd4]
    pcd_curr_list = [pcd2, pcd3, pcd4, pcd5]

    pcd_prev = np.vstack([pcd1, pcd2, pcd3, pcd4])
    pcd_curr = np.vstack([pcd2, pcd3, pcd4, pcd5])
    
    coors_prev = pcd_prev 
    coors_curr = pcd_curr

    coors_prev = torch.tensor(coors_prev).to(DEVICE)
    coors_curr = torch.tensor(coors_curr).to(DEVICE)

    pcd_prev = torch.tensor(pcd_prev).to(DEVICE)
    pcd_curr = torch.tensor(pcd_curr).to(DEVICE)

    offset_prev = [0]
    offset_curr = [0]

    offset = 0
    for pcd in pcd_prev_list:
        offset += pcd.shape[0]
        offset_prev.append(offset)

    offset = 0
    for pcd in pcd_curr_list:
        offset += pcd.shape[0]
        offset_curr.append(offset)

    offset_prev = torch.tensor(offset_prev).to(DEVICE)
    offset_curr = torch.tensor(offset_curr).to(DEVICE)

    input_prev = (coors_prev, pcd_prev, offset_prev)
    input_curr = (coors_curr, pcd_curr, offset_curr)

    model = PointTransformerLO38().to(DEVICE)

    x = model(input_prev, input_curr)
    
    print(x[0])

    print("success")

