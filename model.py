from typing import List, Tuple

import hydra.utils
import math
import torch
import torch.nn as nn
from opt_einsum import contract
from torch import Tensor

from long_seq import process_long_input, process_long_input_longformer
from losses import AFLoss, NCRLoss
import torch.nn.functional as F
from axial_attention import AxialAttention, AxialImageTransformer
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch import RelGraphConv
from transformers import AutoModel, AutoConfig
import numpy as np


class AxialTransformer_by_entity(nn.Module):
    def __init__(self, emb_size=768, dropout=0.1, num_layers=2, dim_index=-1, heads=8, num_dimensions=2, ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_index = dim_index
        self.heads = heads
        self.emb_size = emb_size
        self.dropout = dropout
        self.num_dimensions = num_dimensions

        self.axial_attns = nn.ModuleList(
            [AxialAttention(dim=self.emb_size, dim_index=dim_index, heads=heads, num_dimensions=num_dimensions, ) for i
             in range(num_layers)])

        self.ffns = nn.ModuleList([nn.Linear(self.emb_size, self.emb_size) for i in range(num_layers)])

        self.lns = nn.ModuleList([nn.LayerNorm(self.emb_size) for i in range(num_layers)])

        self.attn_dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(num_layers)])
        self.ffn_dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(num_layers)])

    def forward(self, x):
        for idx in range(self.num_layers):
            x = x + self.attn_dropouts[idx](self.axial_attns[idx](x))
            x = self.ffns[idx](x)
            x = self.ffn_dropouts[idx](x)
            x = self.lns[idx](x)
        return x

# AxialTransformer_by_entity
# class AxialTransformer_by_entity(nn.Module):
#     def __init__(self, emb_size=768, dropout=0.1, num_layers=2, dim_index=-1, heads=8, num_dimensions=2,):
#         super().__init__()
#         self.axial_attns = nn.ModuleList([
#             AxialAttention(dim=emb_size, dim_index=dim_index, heads=heads, num_dimensions=num_dimensions,)
#             for _ in range(num_layers)
#         ])
#         self.ffns = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(emb_size, emb_size * 4),
#                 nn.SiLU(),
#                 nn.Linear(emb_size * 4, emb_size),
#                 nn.Dropout(dropout)
#             )
#             for _ in range(num_layers)
#         ])

#         self.attn_dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(num_layers)])
#         # 添加层归一化
#         self.lns1 = nn.ModuleList([nn.LayerNorm(emb_size) for _ in range(num_layers)])
#         self.lns2 = nn.ModuleList([nn.LayerNorm(emb_size) for _ in range(num_layers)])

#     def forward(self, x):
#         for i in range(len(self.axial_attns)):
#             # 自注意力+残差+层归一化
#             x = x + self.attn_dropouts[i](self.axial_attns[i](x))
#             x = self.lns1[i](x)
#             # 前馈网络+残差+层归一化
#             x = x + self.ffns[i](x)
#             x = self.lns2[i](x)
#         return x

class NoAxialTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.zeros_like(x)


class GATGraphConvLayer(nn.Module):
    def __init__(self, in_feat, out_feat, rel_names, fp, ap, residual, activation):
        super(GATGraphConvLayer, self).__init__()
        self.conv = dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(in_feat, out_feat, num_heads=1, feat_drop=fp, attn_drop=ap, residual=residual,
                               activation=activation)
            for rel in rel_names
        })
        

    def forward(self, g, inputs):
        hs = self.conv(g, inputs)
        return {ntype: h.squeeze(1) for ntype, h in hs.items()}


class GATGraphConv(nn.Module):
    def __init__(self, hidden_dim, edge_types, feat_drop, attn_drop, residual, activation, num_layers):
        super().__init__()
        self.graph_conv = nn.ModuleList([
            GATGraphConvLayer(hidden_dim, hidden_dim, edge_types, feat_drop, attn_drop, residual, activation)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, graph, feat):
        for graph_layer, layer_norm in zip(self.graph_conv, self.layer_norms):
            feat = graph_layer(graph, {'node': feat})['node']
            feat = layer_norm(feat)
        return feat



class NoGraphConv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph, feat):
        return torch.zeros_like(feat)

class KGConv(nn.Module):
    def __init__(self, num_rels, feat_dim, emb_dim):
        super().__init__()
        self.num_rels = num_rels
        self.rel_projection = nn.Embedding(num_rels, emb_dim)
        self.triple_projection = nn.Linear(2 * feat_dim + emb_dim, feat_dim)

    def forward(self, graph, inputs):
        graph.nodes['node'].data['ent'] = inputs
        rel_embeddings = self.rel_projection(torch.arange(self.num_rels, device=inputs.device))
        for rel_id in range(self.num_rels):
            graph.edges['node', rel_id, 'node'].data['rel'] = \
                rel_embeddings[rel_id].unsqueeze(0).expand(graph.num_edges(rel_id), -1)

        def message_func(edges):
            sub, rel, obj = edges.src['ent'], edges.data['rel'], edges.dst['ent']
            triple = torch.cat([sub, obj, rel], dim=-1)
            return {"trp_emb": self.triple_projection(triple)}

        def reduce_func(nodes):
            return {"ent_": torch.tanh(nodes.mailbox["trp_emb"].sum(dim=1))}

        graph.multi_update_all({('node', rel_id, 'node'): (message_func, reduce_func)
                                for rel_id in range(self.num_rels)})
        return graph.nodes['node'].data.pop('ent_')


class KgRelGCN(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_layers, regularizer, num_bases, dropout):
        super().__init__()
        self.conv = nn.ModuleList([
            RelGraphConv(in_feat, out_feat, num_rels, regularizer=regularizer, num_bases=num_bases,
                         dropout=dropout, activation=nn.Tanh()) for _ in range(num_layers)
        ])

    def forward(self, graph, feat, etype):
        for layer in self.conv:
            feat = layer(graph, feat, etype)
        return feat


def batched_l2_dist(a, b):
    a_squared = a.norm(dim=-1).pow(2)
    b_squared = b.norm(dim=-1).pow(2)

    squared_res = torch.baddbmm(
        b_squared.unsqueeze(-2), a, b.transpose(-2, -1), alpha=-2
    ).add_(a_squared.unsqueeze(-1))
    res = squared_res.clamp_min_(1e-30).sqrt_()
    return res


def batched_l1_dist(a, b):
    res = torch.cdist(a, b, p=1)
    return res



class DistMultScore(nn.Module):
    def __init__(self):
        super(DistMultScore, self).__init__()

    def score_func(self, edges):
        head = edges.src['emb']
        tail = edges.dst['emb']
        rel = edges.data['emb']
        score = head * rel * tail
        return {'dr': score.sum(-1)}

    def edge_func(self, edges):
        head = edges.src['emb']
        rel = edges.data['emb']
        score_logits = edges.data['dr']
        trans = head * rel
        return {'score': torch.sigmoid(score_logits), 'trans': trans}

    def reduce_func(self, nodes):
        s, t = nodes.mailbox['score'], nodes.mailbox['trans']
        h = (s.unsqueeze(-1) * t).sum(1)
        return {'h': h}

    def forward(self, g):
        # g.apply_edges(lambda edges: self.edge_func(edges))
        g.apply_edges(lambda edges: self.score_func(edges))
        g.update_all(lambda edges: self.edge_func(edges), lambda nodes: self.reduce_func(nodes))
        return g.ndata.pop('h'), g.edata.pop('dr')


class TypeEmbedding(nn.Module):
    def __init__(self, num_rels, num_bases, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.coeff = nn.Parameter(torch.Tensor(num_rels, num_bases))
        self.W = nn.Parameter(torch.Tensor(num_bases, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            nn.init.xavier_uniform_(self.coeff, gain=nn.init.calculate_gain('tanh'))
            nn.init.uniform_(self.W, -1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim))

    def get_weight(self):
        return self.coeff @ self.W

    def forward(self, etype):
        w = self.get_weight()
        return w[etype.long()]


class KGEmbeddingLayer(nn.Module):
    def __init__(self, hidden_dim, num_rels, num_bases, dropout, score_func, activation):
        super().__init__()
        self.type_emb = TypeEmbedding(num_rels, num_bases, hidden_dim)
        self.score_func = score_func
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, feat, etypes):
        if graph.num_edges() == 0:
            return torch.zeros_like(feat), torch.zeros_like(etypes), False
        with graph.local_scope():
            graph.ndata['emb'] = feat
            graph.edata['etype'] = etypes

            graph.apply_edges(lambda edges: {'emb': self.type_emb(edges.data['etype'])})
            h, score = self.score_func(graph)

            return self.dropout(self.activation(h)), score, True


class NoKGEmbeddingLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph, feat, etypes):
        return torch.zeros_like(feat), torch.zeros_like(etypes), False


class KGEmbedding(nn.Module):
    def __init__(self, hidden_dim, num_rels, num_layers, num_bases, dropout, score_func):
        super().__init__()
        self.conv = nn.ModuleList([
            KGEmbeddingLayer(hidden_dim, num_rels, num_bases, dropout, score_func, nn.Tanh())
            for _ in range(num_layers)
        ])

    def forward(self, graph, feat, etype):
        for layer in self.conv:
            feat, score = layer(graph, feat, etype)
        return feat, score


class DocREModel(nn.Module):
    def __init__(self,
                 model_name_or_path,
                 max_seq_length,
                 transformer_type,
                 tokenizer,
                 graph_conv,
                 residual,
                 coref,
                 num_class,
                 block_size,
                 kg_conv,
                 axial_conv,
                 rel_loss_fnt,
                 kg_loss_weight,
                 max_sent_num=25,
                 evi_thresh=0.2,
                 evi_lambda=0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.max_seq_length = max_seq_length
        self.config.cls_token_id = tokenizer.cls_token_id
        self.config.sep_token_id = tokenizer.sep_token_id
        self.config.transformer_type = transformer_type
        self.config.model_max_len = self.config.max_position_embeddings
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.hidden_size = self.config.hidden_size
        self.emb_size = self.hidden_size
        self.block_size = block_size
        self.num_class = num_class
        self.rel_loss_fnt = rel_loss_fnt
        self.kg_loss_weight = kg_loss_weight
        self.kg_loss_fnt = nn.BCEWithLogitsLoss()
        self.head_extractor = nn.Linear(2 * self.hidden_size, self.emb_size)
        self.tail_extractor = nn.Linear(2 * self.hidden_size, self.emb_size)
        self.projection = nn.Linear(self.emb_size * block_size, self.hidden_size, bias=False)
        self.classifier = nn.Linear(self.hidden_size, self.num_class)
        self.residual = residual
        assert coref in {'gated', 'e_context'}
        self.coref = coref

        if isinstance(graph_conv, NoGraphConv):  # -w/o graph neural network
            self.graph_conv = graph_conv
        else:
            self.graph_conv = graph_conv(hidden_dim=self.hidden_size)
            
        if isinstance(kg_conv, NoKGEmbeddingLayer):  # -w/o knowledge augmentation method
            self.kg_conv = kg_conv
        else:
            self.kg_conv = kg_conv(hidden_dim=self.hidden_size)
        if isinstance(axial_conv, NoAxialTransformer):  # -w/o axial attention
            self.axial_conv = axial_conv
        else:
            self.axial_conv = axial_conv(emb_size=self.hidden_size)

        # ======================= Evidence ==============================
        self.loss_fnt_evi = nn.KLDivLoss(reduction="batchmean")
        self.max_sent_num = max_sent_num
        self.evi_thresh = evi_thresh
        self.evi_lambda = evi_lambda
        # ===============================================================

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert" or config.transformer_type == 'deberta':
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        elif config.transformer_type == 'longformer':
            return process_long_input_longformer(self.model, input_ids, attention_mask)

        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens,
                                                        512)
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, hts, sent_pos, entity_pos, coref_pos, mention_pos, men_graphs, ent_graphs, etypes, ne, sent_labels):
        offset = 1 if self.config.transformer_type in ["bert", "roberta", "longformer", "deberta"] else 0
        batch_size, num_neads, seq_len, _ = attention.size()
        batch_size, seq_len, hidden_size = sequence_output.size()
        hss, rss, tss = [], [], []
        ht_atts = []
        device = sequence_output.device
        n_e = ne
        feats = []
        nms, nss, nes = [len(m) for m in mention_pos], [len(s) for s in sent_pos], [len(e) for e in entity_pos]

        for i in range(len(entity_pos)):  # for each batch
            _entity_embs, _entity_atts = [], []

            # obtain entity embedding from mention embeddings.
            for eid, e in enumerate(entity_pos[i]):  # for each entity
                if len(e) > 1:
                    _e_emb, _e_att = [], []
                    for mid, (start, end) in enumerate(e):  # for every mention
                        if start + offset < seq_len:
                            # In case the entity mention is truncated due to limited max seq length.
                            _e_emb.append(sequence_output[i, start + offset])
                            _e_att.append(attention[i, :, start + offset])

                    if len(_e_emb) > 0:
                        _e_emb = torch.logsumexp(torch.stack(_e_emb, dim=0), dim=0)
                        _e_att = torch.stack(_e_att, dim=0).mean(0)
                    else:
                        _e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        _e_att = torch.zeros(num_neads, seq_len).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < seq_len:
                        _e_emb = sequence_output[i, start + offset]
                        _e_att = attention[i, :, start + offset]
                    else:
                        _e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        _e_att = torch.zeros(num_neads, seq_len).to(attention)

                _entity_embs.append(_e_emb)
                _entity_atts.append(_e_att)

            # entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            _entity_atts = torch.stack(_entity_atts, dim=0)  # [n_e, h, seq_len]
            _ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            _h_att = torch.index_select(_entity_atts, 0, _ht_i[:, 0])
            _t_att = torch.index_select(_entity_atts, 0, _ht_i[:, 1])

            _ht_att = (_h_att * _t_att).mean(1)  # average over all heads
            _ht_att = _ht_att / (_ht_att.sum(1, keepdim=True) + 1e-30)
            ht_atts.append(_ht_att)

        batch_rel = [len(ht) for ht in hts]
        ht_atts = torch.cat(ht_atts, dim=0)
        s_attn = self.calculate_evidence_attn(ht_atts, sent_pos, batch_rel)
        

        
        for i in range(batch_size):
            doc_emb = sequence_output[i][0].unsqueeze(0)
            mention_embs = sequence_output[i, mention_pos[i] + offset]
            sentence_embs = [torch.logsumexp(sequence_output[i, offset + sent_start:offset + sent_end], dim=0)
                             for sent_start, sent_end in sent_pos[i]]
            sentence_embs = torch.stack(sentence_embs)
            

            if sent_labels is not None and self.evi_lambda > 0:
                attn = s_attn[sum(batch_rel[:i]):sum(batch_rel[:i + 1])].mean(dim=0)[:len(sentence_embs)]
                assert len(attn) == len(
                    sentence_embs), f"Length of attn ({len(attn)}) != length of sentence_embs ({len(sentence_embs)})"
                sentence_embs = sentence_embs * attn.unsqueeze(-1)
            all_embs = torch.cat([doc_emb, mention_embs, sentence_embs], dim=0)
            feats.append(all_embs)
        feats = torch.cat(feats, dim=0)
        assert len(feats) == batch_size + sum(nms) + sum(nss)
        feats = self.graph_conv(men_graphs, feats)

        def get_entity_embeddings(i, e, men_idx, cur_idx):
            if len(e) > 1:
                e_emb_list, g_emb_list, e_att_list = [], [], []
                for start, end in e:
                    if start + offset < seq_len:
                        e_emb_list.append(sequence_output[i, start + offset])
                        g_emb_list.append(feats[cur_idx + 1 + men_idx])
                        e_att_list.append(attention[i, :, start + offset])
                if e_emb_list:
                    e_emb_tensor = torch.stack(e_emb_list)
                    g_emb_tensor = torch.stack(g_emb_list)
                    if self.residual:
                        e_emb = e_emb_tensor + g_emb_tensor
                    else:
                        e_emb = g_emb_tensor
                    if self.coref == 'gated':
                        att = torch.stack(e_att_list).mean(0).sum(0)
                        gate_score = att / att.sum()
                        coref_emb = []
                        for start, end in coref_pos[i][e_id]:
                            coref_emb.append(
                                (gate_score[start:end].unsqueeze(-1) * sequence_output[i, start:end]).sum(0))
                        if coref_emb:
                            e_emb = torch.cat([e_emb, torch.stack(coref_emb)])
                    e_emb = torch.logsumexp(e_emb, dim=0)
                    if self.coref == 'e_context':
                        for start, end in coref_pos[i][e_id]:
                            e_att_list.append(attention[i, :, start:end].mean(1))
                    e_att = torch.stack(e_att_list).mean(0)
                else:
                    e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                    e_att = torch.zeros(num_neads, seq_len).to(attention)
            else:
                start, end = e[0]
                if start + offset < seq_len:
                    if self.residual:
                        e_emb = sequence_output[i, start + offset] + feats[cur_idx + 1 + men_idx]
                    else:
                        e_emb = feats[cur_idx + 1 + men_idx]
                    if self.coref == 'gated':
                        e_att = attention[i, :, start + offset]
                        att = e_att.sum(0)
                        gate_score = att / att.sum()
                        coref_emb = []
                        for start, end in coref_pos[i][e_id]:
                            coref_emb.append(
                                (gate_score[start:end].unsqueeze(-1) * sequence_output[i, start:end]).sum(0))
                        if coref_emb:
                            e_emb = torch.cat([e_emb.unsqueeze(0), torch.stack(coref_emb)])
                            e_emb = torch.logsumexp(e_emb, dim=0)
                    else:
                        if not coref_pos[i][e_id]:
                            e_att = attention[i, :, start + offset]
                        else:
                            e_att = [attention[i, :, start + offset]]
                            for start, end in coref_pos[i][e_id]:
                                e_att.append(attention[i, :, start:end].mean(1))
                            e_att = torch.stack(e_att).mean(0)
                else:
                    e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                    e_att = torch.zeros(num_neads, seq_len).to(attention)
            return e_emb, e_att

        batch_entity_embs, batch_entity_atts = [], []
        for i in range(batch_size):
            entity_embs, entity_atts = [], []
            men_idx = -1
            cur_idx = sum(nms[:i]) + sum(nss[:i])
            for e_id, e in enumerate(entity_pos[i]):
                e_emb, e_att = get_entity_embeddings(i, e, men_idx, cur_idx)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)
                men_idx += 1
            entity_embs = torch.stack(entity_embs)
            entity_atts = torch.stack(entity_atts)
            batch_entity_embs.append(entity_embs)
            batch_entity_atts.append(entity_atts)

        all_entity_embs = torch.cat(batch_entity_embs)
        kg_feats, kg_score, kg_flag = self.kg_conv(ent_graphs, all_entity_embs, etypes)

        m = nn.Threshold(0, 0)

        cur_idx = 0
        for i in range(batch_size):
            entity_embs = batch_entity_embs[i] + kg_feats[cur_idx:cur_idx + nes[i]]
            cur_idx += nes[i]
            entity_atts = batch_entity_atts[i]
            s_ne, _ = entity_embs.size()
            ht_i = torch.LongTensor(hts[i]).to(device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])
            pad_hs = torch.zeros((n_e, n_e, hidden_size)).to(device)
            pad_ts = torch.zeros((n_e, n_e, hidden_size)).to(device)
            pad_hs[:s_ne, :s_ne, :] = hs.view(s_ne, s_ne, hidden_size)
            pad_ts[:s_ne, :s_ne, :] = ts.view(s_ne, s_ne, hidden_size)
            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = m((h_att * t_att).sum(1))
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-10)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            pad_rs = torch.zeros(n_e, n_e, hidden_size).to(device)
            pad_rs[:s_ne, :s_ne, :] = rs.view(s_ne, s_ne, hidden_size)
            hss.append(pad_hs)
            rss.append(pad_rs)
            tss.append(pad_ts)

        hss = torch.stack(hss)
        tss = torch.stack(tss)
        rss = torch.stack(rss)

        
        return hss, rss, tss, s_attn, kg_score, kg_flag

    def calculate_evidence_attn(self, doc_attn, sent_pos, batch_rel):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        max_sent_num = max([len(sent) for sent in sent_pos])
        rel_sent_attn = []
        seq_len = doc_attn.size(-1)  
        for i in range(len(sent_pos)):  # for each batch
            # the relation ids corresponds to document in batch i is [sum(batch_rel[:i]), sum(batch_rel[:i+1]))
            curr_attn = doc_attn[sum(batch_rel[:i]):sum(batch_rel[:i + 1])]
            curr_sent_pos = []
            for s_start, s_end in sent_pos[i]:
                
                assert s_start < s_end, \
                    f"Invalid sentence interval in document {i}: (s_start={s_start}, s_end={s_end})"
                assert s_start < seq_len, \
                    f"Sentence start index out of bounds in document {i}: s_start={s_start}, seq_len={seq_len}"
                s_end = min(s_end, seq_len)  
                curr_sent_pos.append(torch.arange(s_start, s_end).to(curr_attn.device) + offset)

            curr_attn_per_sent = [curr_attn.index_select(-1, sent) for sent in curr_sent_pos]
            curr_attn_per_sent += [torch.zeros_like(curr_attn_per_sent[0])] * (max_sent_num - len(curr_attn_per_sent))
            sum_attn = torch.stack([attn.sum(dim=-1) for attn in curr_attn_per_sent],
                                   dim=-1)  # sum across those attentions
            rel_sent_attn.append(sum_attn)

        s_attn = torch.cat(rel_sent_attn, dim=0)
        return s_attn

    def compute_evidence_loss(self, s_attn, sent_labels):
        
        norm_s_labels = sent_labels / (sent_labels.sum(dim=-1, keepdim=True) + 1e-30)
        norm_s_labels[norm_s_labels == 0] = 1e-30
        s_attn[s_attn == 0] = 1e-30
        return self.loss_fnt_evi(s_attn.log(), norm_s_labels)

    def forward(self,
                input_ids,
                attention_mask,
                hts,
                sent_pos,
                entity_pos,
                coref_pos,
                mention_pos,
                entity_types,
                men_graphs,
                ent_graphs,
                etypes,
                e_labels,
                labels,
                sent_labels=None,
                ):
        sequence_output, attention = self.encode(input_ids, attention_mask)
        batch_size, num_heads, seq_len, seq_len = attention.size()
        sequence_output[:, self.max_seq_length:, :] = 0
        # device = sequence_output.device
        nes = [len(x) for x in entity_pos]
        ne = max(nes)
                                                   
        hs_e, rs_e, ts_e, s_attn, e_scores, kg_flag = self.get_hrt(sequence_output, attention, hts, sent_pos, entity_pos,
                                                           coref_pos, mention_pos, men_graphs, ent_graphs, etypes, ne, sent_labels)
        hs_e = torch.tanh(self.head_extractor(torch.cat([hs_e, rs_e], dim=3)))
        ts_e = torch.tanh(self.tail_extractor(torch.cat([ts_e, rs_e], dim=3)))

        b1_e = hs_e.view(batch_size, ne, ne, self.emb_size // self.block_size, self.block_size)
        b2_e = ts_e.view(batch_size, ne, ne, self.emb_size // self.block_size, self.block_size)

        bl_e = (b1_e.unsqueeze(5) * b2_e.unsqueeze(4)).view(batch_size, ne, ne, self.emb_size * self.block_size)

        feature = self.projection(bl_e)

        # feature = self.axial_conv(feature) + feature
        feature = self.axial_conv(feature)
        rel_logits = self.classifier(feature)

        self_mask = (1 - torch.diag(torch.ones(ne))).unsqueeze(0).unsqueeze(-1).to(sequence_output)
        rel_logits = rel_logits * self_mask
        final_logits = torch.cat([
            rel_logits[x, :nes[x], :nes[x], :].reshape(-1, self.num_class) for x in range(batch_size)
        ])

        if labels is not None:
            rel_loss = self.rel_loss_fnt(final_logits, labels)
            total_loss = rel_loss
            loss_dict = {'rel_loss': rel_loss.item()}
            
            if sent_labels is not None and self.evi_lambda > 0:
                
                evi_pred = F.pad(s_attn > self.evi_thresh, (0, self.max_sent_num - s_attn.shape[-1]))
                
                assert s_attn.shape == sent_labels.shape, \
                    f"Evidence attn shape {s_attn.shape} != sent_labels shape {sent_labels.shape}"
                idx_used = torch.nonzero(labels[:, 1:].sum(dim=-1)).view(-1)  
                s_attn = s_attn[idx_used]
                sent_labels = sent_labels[idx_used]
                evi_loss = self.compute_evidence_loss(s_attn, sent_labels)
                evi_loss = self.evi_lambda * evi_loss
                total_loss = rel_loss + evi_loss  
                loss_dict['evi_loss'] = evi_loss.item()
                

            if self.kg_loss_weight < 0 or not kg_flag:
                return total_loss
            kg_loss = self.kg_loss_fnt(e_scores, e_labels)
            kg_loss = self.kg_loss_weight * kg_loss
            output = total_loss + kg_loss
            loss_dict['kg_loss'] = kg_loss.item()
            print(loss_dict)
            return output

        else:
            return self.rel_loss_fnt.get_label(final_logits)

