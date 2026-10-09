"""모터 진동(vibration) CSV → 파일당 한 행인 머신러닝 특징 CSV.

실행: .venv/bin/python merge_motor_data.py
추가 패키지: numpy. 원본 파형 파일은 수정하지 않습니다.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path
import unicodedata

import numpy as np


def normalize(value):
    return unicodedata.normalize("NFC", value.strip())


def source_name(path, root):
    """출력 경로에서 vibration 폴더명만 제외합니다."""
    parts = list(path.relative_to(root).parts)
    parts.pop(parts.index("vibration"))
    return "/".join(parts)


STATS = ("mean", "std", "min", "max", "abs_max", "rms", "peak_to_peak", "crest_factor")
METADATA = (
    "source_file", "split", "sensor", "power_kw", "motor_model", "label",
    "label_no", "device_id", "recorded_at", "metadata_filename", "sample_rate_hz",
    "period", "n_samples", "n_channels",
)
FIELDS = list(METADATA) + [f"ch1_{stat}" for stat in STATS]


def read_sensor_file(path, root):
    """시간 열과 진동 측정 열을 읽고 폴더와 내부 라벨을 대조합니다."""
    parts = [normalize(part) for part in path.relative_to(root).parts]
    sensor_index = parts.index("vibration")
    if len(parts[sensor_index:]) != 5:
        raise ValueError("폴더 구조는 sensor/출력/모델/고장명/파일.csv 이어야 합니다")
    sensor, power, model, label, _ = parts[sensor_index:]
    split = parts[sensor_index - 1] if sensor_index else "unspecified"

    # 줄 수를 고정하지 않고 첫 숫자 행을 찾아 메타데이터와 파형을 구분합니다.
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError:
        lines = path.read_text(encoding="cp949").splitlines()
    metadata = {}
    for start, line in enumerate(lines):
        row = next(csv.reader([line]))
        if not row or not row[0].strip():
            continue
        try:
            float(row[0])
        except ValueError:
            metadata[normalize(row[0])] = [normalize(x) for x in row[1:] if x.strip()]
        else:
            break
    else:
        raise ValueError("측정값이 없습니다")

    def meta(key, default=""):
        return metadata.get(key, [default])[0]

    if meta("Data Label") != label:
        raise ValueError(f"고장명 불일치: 폴더={label}, Data Label={meta('Data Label')}")
    specs = metadata.get("Motor Spec", [])
    if specs and specs[0] != model:
        raise ValueError(f"모델 불일치: 폴더={model}, Motor Spec={specs[0]}")

    columns = next(csv.reader([lines[start]]))
    while columns and not columns[-1].strip():
        columns.pop()
    channels = len(columns) - 1
    if channels != 1:
        raise ValueError(f"진동 1채널 파일이 필요합니다. 실제 채널 수: {channels}")
    data = np.loadtxt(lines[start:], delimiter=",", usecols=range(channels + 1), ndmin=2)
    if not np.isfinite(data).all():
        raise ValueError("NaN 또는 무한대 측정값이 있습니다")
    if np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError("파일 내부 측정 시각이 증가하지 않습니다")
    if len(data) != int(meta("Data Length")):
        raise ValueError(f"Data Length={meta('Data Length')}, 실제 샘플={len(data)}")
    rate = float(meta("Sample Rate"))
    if rate <= 0:
        raise ValueError("Sample Rate는 양수여야 합니다")

    result = dict(zip(METADATA, (
        source_name(path, root), split, sensor,
        float(power.lower().removesuffix("kw")), model, label,
        meta("Label No", meta("Label_No")), path.stem.split("_")[0],
        meta("Date"), meta("Filename"), rate, meta("Period"), len(data), channels,
    )))
    for channel in range(channels):
        values = data[:, channel + 1]
        rms = float(np.sqrt(np.mean(values ** 2)))
        peak = float(np.max(np.abs(values)))
        features = (
            float(values.mean()), float(values.std(ddof=0)), float(values.min()),
            float(values.max()), peak, rms, float(np.ptp(values)), peak / rms if rms else 0.0,
        )
        result.update({f"ch{channel + 1}_{name}": value for name, value in zip(STATS, features)})
    return result


def merge_data(data_root, output):
    root = Path(data_root).resolve()
    output = Path(output).resolve()
    if not root.is_dir():
        raise ValueError(f"데이터 폴더가 없습니다: {root}")
    # 출력은 원본 폴더 밖에 두어 재실행 시 결과 파일이 입력에 섞이지 않게 합니다.
    if output.is_relative_to(root):
        raise ValueError("출력 CSV는 원본 데이터 폴더 밖에 지정하세요")
    files = sorted(p for p in root.rglob("*.csv") if "vibration" in p.relative_to(root).parts)
    if not files:
        raise ValueError("vibration 폴더 안에 CSV가 없습니다")
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_suffix(output.suffix + ".tmp")
    error_path = output.with_name(output.stem + "_errors.csv")
    counts, errors = Counter(), []
    with pending.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for index, path in enumerate(files, 1):
            try:
                result = read_sensor_file(path, root)
            except (ValueError, OSError, StopIteration, IndexError) as error:
                errors.append({"source_file": source_name(path, root), "error": str(error)})
            else:
                writer.writerow(result)
                counts[(result["sensor"], result["label"])] += 1
            if index % 500 == 0 or index == len(files):
                print(f"처리: {index:,}/{len(files):,}", flush=True)
    # 실패 목록은 빈 경우에도 헤더를 기록합니다. 오류를 조용히 무시하지 않습니다.
    with error_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_file", "error"])
        writer.writeheader()
        writer.writerows(errors)
    pending.replace(output)
    print(f"\n저장: {output}\n성공: {sum(counts.values()):,}, 실패: {len(errors):,}")
    for (sensor, label), count in sorted(counts.items()):
        print(f"  {sensor:10} {label}: {count:,}")
    if errors:
        print(f"실패한 파일을 확인하세요: {error_path}")
    return sum(counts.values()), len(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="샘플 폴더 또는 Training/Validation의 상위 폴더")
    parser.add_argument("--output", type=Path, default=Path("machine_data.csv"))
    args = parser.parse_args()
    root = args.data_root
    if root is None:
        candidates = list(Path(__file__).resolve().parent.glob("2020-02-125*sample"))
        if len(candidates) != 1:
            parser.error("--data-root 옵션으로 데이터 폴더를 지정하세요")
        root = candidates[0]
    _, failures = merge_data(root, args.output)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
