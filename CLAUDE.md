## 1. First, add code to set a random seed so that the graph transfer learning called GraphLoRA can be reproduced every time the program is run.

## 2. config2.yaml 등의 파일에서 현재 데이터셋은 Cora, CiteSeer, PubMed, Computers, Photo밖에 지원하지 않는다.
다음 6종류의 heterophilic dataset도 지원할 수 있도록 프로그램 코드와 config.yaml, config2.yaml을 수정해라.

WebKb dataset(Cornell, Wisconsin, Texas) : https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.WebKB.html

WikipeidaNetwork(Chameleon, Squirrel) : https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.WikipediaNetwork.html

Actor : https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Actor.html

----
참고용: 

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