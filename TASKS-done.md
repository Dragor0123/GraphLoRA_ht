<!-- 작업내용 : model/GraphLoRA.py의 코드에서 다음 두 loss term에 대한 논리 흐름을 정리한 후, 공통된 부분은 앞에 두고, 분기된 이후부터의 계산 경로를 각각 모듈화 해라.
각 loss에 대해 함수 호출로 바로 가져오도록 하여 개발자가 해당 loss term을 on/off 하기 편하게 만들어라.

## 완료된 작업 (2025-08-27)
1. **SMMD loss 모듈화**: `calculate_smmd_loss()` 함수로 분리
   - feature_map을 입력받아 SMMD loss 계산
   - `args.l2 > 0` 조건으로 on/off 제어 가능

2. **Regularization loss 모듈화**: `calculate_reg_loss()` 함수로 분리  
   - logits와 target_adj를 입력받아 regularization loss 계산
   - `args.l4 > 0` 조건으로 on/off 제어 가능

3. **중복 코드 제거**
   - training loop 밖에 있던 불필요한 weight_tensor 초기화 코드 제거
   - 실제로는 calculate_reg_loss 함수 내에서 매번 계산됨

1. SMMD loss term의 코드 라인
test_dataset = get_dataset(test_datapath, args.test_dataset)[0]
if is_reduction:
    feature_reduce = SVDFeatureReduction(out_channels=100)
    pretrain_dataset = feature_reduce(pretrain_dataset)
    test_dataset = feature_reduce(test_dataset)
pretrain_dataset.edge_index = add_remaining_self_loops(pretrain_dataset.edge_index)[0]
test_dataset.edge_index = add_remaining_self_loops(test_dataset.edge_index)[0]
pretrain_dataset = pretrain_dataset.to(device)
test_dataset = test_dataset.to(device)

ppr_weight = get_ppr_weight(test_dataset)

pretrain_graph_loader = DataLoader(pretrain_dataset.x, batch_size=128, shuffle=True)

feature_map = projector(test_dataset.x)

smmd_loss_f = batched_smmd_loss(feature_map, pretrain_graph_loader, SMMD, ppr_weight, 128)
loss = args.l1 * cls_loss + args.l2 * smmd_loss_f +  args.l3 * ct_loss + args.l4 * loss_reg

2. loss_reg
# feature_map 계산전까지는 SMMD loss term 계산 경로와 동일
feature_map = projector(test_dataset.x)
emb, emb1, emb2 = gnn2(feature_map, test_dataset.edge_index)
logits = logreg(emb)

reg_adj = torch.sigmoid(torch.matmul(torch.softmax(logits, dim=1), torch.softmax(logits, dim=1).T))
loss_reg = F.binary_cross_entropy(reg_adj.view(-1), target_adj.view(-1), weight=weight_tensor)

loss = args.l1 * cls_loss + args.l2 * smmd_loss_f +  args.l3 * ct_loss + args.l4 * loss_reg -->

<!-- 2025년 08월 28일
Task 1. batched_mmd_loss 함수 정의하기
Detail instruction: 
- util.py의 batched_smmd_loss와 거의 유사하게 동작하되, ppr을 사용한 structure-aware 요소가 사라진 버전, 이른바
SMMD가 아닌 Original MMD만을 사용하는 함수를 만들어라.
- 최대한 batched_smmd_loss와 함수 시그니쳐가 비슷하게 만들되 ppr_weight를 인자로 취하지 않으면 된다.

```
# batched_smmd_loss : Structure-aware Maximum MeanDiscrepancy using personalized pagerank
def batched_smmd_loss(z1: torch.Tensor, z2, MMD, ppr_weight, batch_size):
    device = z1.device
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    indices = torch.arange(0, num_nodes).to(device)
    losses = []

    for i in range(num_batches):
        mask = indices[i * batch_size:(i + 1) * batch_size]
        ppr = ppr_weight[mask][:, mask]
        target = next(iter(z2))
        losses.append(MMD(z1[mask], target, ppr))

    return torch.stack(losses).mean()

작업 완료: 25.08.28, 13시 20분.
``` -->


* 참고용
| Dataset | Nodes | Edges | Classes | Features |
|:---|---:|---:|---:|---:|
| Cora | 2,708 | 10,556 | 7 | 1,433 |
| CiteSeer | 3,327 | 9,104 | 6 | 3,703 |
| PubMed | 19,717 | 88,648 | 3 | 500 |
| Computers | 13,752 | 491,722 | 10 | 767 |
| Photo | 7,650 | 238,162 | 8 | 745 |
||
| Cornell | 183 | 295 | 5 | 1,703 |
| Texas | 183 | 309 | 5 | 1,703 |
| Wisconsin | 251 | 499 | 5 | 1,703 | 
| Chameleon | 2,277 | 36,051 | 5 | 2,325 |
| Squirrel | 5,201 | 216,933 | 5 | 2,089 | 
| Actor | 7,600 | 29,926 | 5 | 932 | 