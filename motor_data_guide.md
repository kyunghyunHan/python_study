# 모터 진동 데이터를 하나의 CSV로 합치기

현재 폴더에서 실행합니다. 기존 `.venv`에는 필요한 numpy가 설치되어 있습니다.

```bash
.venv/bin/python merge_motor_data.py
```

`machine_data.csv`는 **진동 원본 CSV 하나당 한 행**인 특징 데이터입니다. `current` 파일은 제외합니다. 원본 파형은 보존되며, 결과에는 파형 전체가 아니라 진동 통계 특징이 들어갑니다. Random Forest나 SVM 같은 표 형식 모델에 사용할 수 있습니다. 1D CNN에 입력할 원본 시계열 통합 파일은 아닙니다.

다른 데이터 폴더를 사용할 때:

```bash
.venv/bin/python merge_motor_data.py --data-root "/실제/데이터/경로" --output machine_data.csv
```

`vibration/출력/모델/고장명/*.csv`만 찾습니다. 앞에 Training 또는 Validation 폴더가 있으면 `split`에 기록하며, 현재 샘플처럼 없으면 `unspecified`로 기록합니다.

| 열 | 의미 |
|---|---|
| sensor | vibration |
| power_kw | 모터 출력: 2.2 또는 55 |
| motor_model | 모터 모델명 |
| label | 예측할 정답: 정상, 베어링불량 등 |
| label_no | 원본 고장 코드. 정답 정보이므로 입력 특징에서 제외 |
| source_file | vibration/을 생략한 경로. 예: 55kW/R-PAHU-04S/정상/파일.csv |
| device_id | 파일명 앞부분의 식별자. 장비 구분에 활용하려면 실제 장비와의 관계 확인 필요 |
| recorded_at | CSV 내부 Date 값 |
| sample_rate_hz / n_samples / n_channels | 샘플링 주파수, 샘플 수, 채널 수 |
| ch1_mean / ch1_std / ch1_rms 등 | 1번 채널의 평균, 표준편차, RMS 등 |

진동 1채널의 mean, std(모집단 표준편차), min, max, abs_max, rms, peak_to_peak, crest_factor(최대절댓값/RMS)를 계산합니다. 사용하지 않는 ch2/ch3 열은 제외합니다.

주피터에서 결과를 읽는 예시:

```python
import pandas as pd

df = pd.read_csv("machine_data.csv")
display(df.head())
display(df.groupby(["sensor", "label"]).size())

# 모든 행이 진동 데이터입니다.
feature_columns = [column for column in df.columns if column.startswith("ch1_")]
X = df[feature_columns]
y = df["label"]
```

CSV 내부 RMS 대신 파형으로 직접 RMS를 계산합니다. 내부 고장명과 폴더 고장명이 다르거나, 샘플 수가 잘못된 파일 등은 `machine_data_errors.csv`에 기록하고 실행 결과에 실패 수를 표시합니다. 재실행하면 결과 파일을 갱신합니다.

학습 입력에는 `label`, `label_no`, `source_file`, `metadata_filename`을 넣지 마세요. 경로나 파일명이 정답 정보를 노출할 수 있습니다. 현재 샘플에는 학습/검증 구분이 없으므로 별도 분할이 필요합니다. 같은 장비·측정 세션의 유사 파일이 양쪽에 섞이지 않도록 장비 또는 시간 단위로 분할하세요. `motor_model`과 고장 종류가 연결되어 있을 수 있으므로 새 장비에 대한 성능은 별도로 확인해야 합니다. 이 CSV로 학습하는 기본 과제는 현재 정상/고장 상태 분류입니다.
