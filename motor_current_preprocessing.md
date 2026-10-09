# 전류 데이터 전처리

```bash
.venv/bin/python preprocess_motor_current.py
# 다른 데이터: Training과 Validation이 함께 들어 있는 상위 경로를 지정
.venv/bin/python preprocess_motor_current.py --data-root "/데이터/상위폴더"
```

`current/출력/모터모델/정상·고장폴더/*.csv`만 재귀 탐색합니다. Training, Validation이 있으면 둘 다 처리하고 `split`에 원래 구분을 기록합니다. 현재 샘플에는 두 폴더가 없어서 `unspecified`로 기록합니다. 원본 파일은 읽기만 하며, 학습·예측·데이터 분할·정확도 계산은 수행하지 않습니다.

결과는 `motor_current_features.csv`, 문제 목록은 `motor_current_features_issues.csv`, 집계는 `motor_current_features_summary.json`입니다. 재실행하면 결과만 갱신합니다. 원본 CSV당 결과 한 행이며, 읽지 못한 파일도 `status=error`인 행으로 남기고 특징은 비워 둡니다. 후속 학습에서는 error 행을 제외하고 warning 행의 원인을 확인하세요.

메타데이터와 첫 숫자 행부터 시작하는 파형을 구분합니다. UTF-8 BOM 및 CP949를 지원하며, 줄 끝의 빈 CSV 열을 제거합니다. NaN/무한대, 비정상 채널 수, 샘플 수 불일치, 불균일 시간 간격은 실패로 기록합니다. 누락된 메타데이터 이름은 `missing_metadata`에 기록합니다. 샘플링 주파수가 없으면 FFT를 계산할 수 없어 실패합니다. 존재 여부를 알 수 없는 원본 파일은 집계할 수 없으며, 누락 집계는 발견한 파일의 메타데이터 누락을 의미합니다.

전류 각 채널에 ch1/ch2/ch3 접두사를 붙여 특징을 계산합니다. 채널의 물리적 명칭은 데이터 문서로 확인하지 않았으므로 채널 번호를 사용합니다.

- Mean, Standard Deviation(모집단, ddof=0), Min, Max, RMS, Peak-to-Peak, Crest Factor(최대절댓값/RMS).
- Skewness는 표준화된 3차 모집단 모멘트, Kurtosis는 4차 모집단 모멘트에서 3을 뺀 Fisher 초과첨도입니다. 편향 보정은 적용하지 않습니다. 상수 신호의 왜도·첨도, 0 신호의 Crest Factor는 정의되지 않으므로 빈칸과 경고를 남깁니다.
- FFT는 평균을 제거하고 창 함수 없이 rFFT를 계산합니다. `fft_peak1_hz`~`fft_peak3_hz`는 DC를 제외한 스펙트럼 국소 최대점 중 에너지가 큰 순서의 주파수입니다. 고장 원인 주파수를 자동 판정한 결과는 아닙니다.
- 대역은 0–10, 10–50, 50–100, 100–200, 200–500, 500–1000Hz입니다. 하한 포함·상한 제외이며 마지막 가용 대역에는 Nyquist bin을 포함합니다. Nyquist 이상 대역은 빈칸, 일부만 측정 가능한 대역은 가용 구간의 에너지와 경고를 기록합니다.
- `fft_energy_*`는 단측 FFT에서 DC·Nyquist를 제외한 bin을 두 배로 보정한 `abs(FFT)**2 / N`의 합입니다. 단위는 원본 전류 단위의 제곱 × 샘플 수이며, 전체 `fft_ac_energy`는 `sum((x-mean)**2)`와 같습니다. 기록 길이가 달라지면 에너지 비교 시 샘플 수로 나누는 정규화가 필요합니다. FFT 주파수 분해능은 Sample Rate / N입니다.

`source_file`은 current를 포함한 원본 상대 경로입니다. `device_id`는 CSV 내부 Filename의 첫 밑줄 앞 식별자이며, Filename이 없으면 실제 파일명을 사용합니다. 식별자가 실제 장비와 어떻게 대응하는지는 별도 확인이 필요합니다. `recorded_at`은 내부 Date를 그대로 기록합니다.

`folder_label`, `csv_label`을 모두 보존하고 불일치는 `label_mismatch=True` 및 문제 목록으로 표시합니다. `label`은 CSV 내부 라벨을 우선 보존하며, 없으면 폴더 라벨을 사용하고 `label_source`에 출처를 기록합니다. 불일치 파일을 자동 삭제하거나 정상/고장 라벨을 재분류하지 않습니다. 폴더와 Motor Spec의 모델·출력이 다른 경우에도 경고와 원래 값을 보존합니다.

나중에 학습할 때에는 `ch*_` 특징을 입력으로 사용하고 라벨, 라벨 코드, 라벨이 포함된 파일 경로는 입력에서 제외하세요.
