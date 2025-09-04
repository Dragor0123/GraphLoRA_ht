#!/bin/bash

# --- 여기를 수정하세요 ---
# 테스트를 위해 목록에 일부러 중복되는 데이터(Cora, PubMed)를 추가했습니다.
source_list="Computers"
target_list="Cora CiteSeer PubMed Computers Photo Chameleon Squirrel Actor"
# -------------------------

# 바깥쪽 루프
for source in $source_list
do
  # 안쪽 루프
  for target in $target_list
  do
    # 🔽 추가된 부분: source와 target이 같은지 확인
    if [[ "$source" == "$target" ]]; then
      echo "--- SKIPPING: $source and $target are the same. ---"
      continue  # 현재 루프를 중단하고 다음 target으로 넘어갑니다.
    fi

    # 이름이 다를 경우에만 아래 명령어가 실행됩니다.
    echo "--- Running for combination: $source and $target ---"
    ./run_script.sh "$source" "$target" 42 46
  done
done

echo "All script combinations finished!"