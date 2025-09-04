설계 의도
1. 이중 인코더 전략 (Dual Encoder Strategy)

- 구조 인코더 (Structural Encoder): GraphControl 스타일로 노드 특징 없이 위치 임베딩만 사용
- 특징 인코더 (Feature Encoder): GRACE 스타일로 노드 특징 활용
- 두 인코더를 독립적으로 사전학습한 후, 파인튜닝 시 결합

2. 공정한 비교 실험 프레임워크
GraphFourierFT + Heterophilic GraphControl
vs
GraphLoRA + Heterophilic GraphControl
- 동일한 사전학습 시작점
- 동일한 4개 손실함수 (cls, smmd, contrastive, struct_reg)
- 동일한 조건 생성 메커니즘

3. Heterophilic GraphControl 통합
- 조건 유형 3가지:
    * role: k-hop 차수 역할 유사성
    * inverse: 특징 코사인 유사도의 하위 분위수 (이종친화적)
    * ppr: PPR × 특징 거리
- ZeroMLP를 통한 FiLM(Feature-wise Linear Modulation) 적용

4. 어댑터 변조 메커니즘
- FourierFT: 스펙트럼 계수 변조 c' = γ ⊙ c + β
- LoRA: 저랭크 분기 진폭 변조

5. 메모리 효율성 비교
- 어댑터만의 파라미터 vs 전체 학습 가능 파라미터
- MB 단위로 정확한 메모리 사용량 추적