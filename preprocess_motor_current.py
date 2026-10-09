"""전류 CSV 전처리만 수행합니다. 모델 학습·예측·데이터 분할 없음."""

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np


BANDS = ((0, 10), (10, 50), (50, 100), (100, 200), (200, 500), (500, 1000))
STATS = (
    "mean", "std", "min", "max", "rms", "peak_to_peak", "crest_factor",
    "skewness", "kurtosis", "fft_peak1_hz", "fft_peak2_hz", "fft_peak3_hz",
    "fft_ac_energy",
) + tuple(f"fft_energy_{low}_{high}_hz" for low, high in BANDS)
META_FIELDS = (
    "source_file", "split", "sensor", "power_kw", "motor_model", "device_id",
    "recorded_at", "sample_rate_hz", "label", "folder_label", "csv_label",
    "label_mismatch", "label_source", "label_no", "metadata_filename",
    "metadata_motor_model", "metadata_power_kw", "period", "declared_n_samples",
    "n_samples", "n_channels", "encoding", "status", "missing_metadata", "warnings", "error",
)
FIELDS = list(META_FIELDS) + [f"ch{ch}_{stat}" for ch in range(1, 4) for stat in STATS]
ISSUE_FIELDS = ("source_file", "status", "label_mismatch", "missing_metadata", "warnings", "error")


def norm(text):
    return unicodedata.normalize("NFC", text.strip())


def features(values, rate):
    """첨도는 Fisher 초과첨도, 모멘트는 모집단 기준. FFT는 DC 제거 후 계산."""
    n = len(values)
    centered = values - values.mean()
    std = float(values.std(ddof=0))
    rms = float(np.sqrt(np.mean(values ** 2)))
    result = dict(zip(STATS[:9], (
        float(values.mean()), std, float(values.min()), float(values.max()), rms,
        float(np.ptp(values)), float(np.max(np.abs(values))) / rms if rms else "",
        float(np.mean((centered / std) ** 3)) if std else "",
        float(np.mean((centered / std) ** 4) - 3) if std else "",
    )))
    frequencies = np.fft.rfftfreq(n, d=1 / rate)
    # Parseval: one-sided bin energy sum == sum(centered**2).
    energy = np.abs(np.fft.rfft(centered)) ** 2 / n
    if n % 2 == 0:
        energy[1:-1] *= 2
    else:
        energy[1:] *= 2
    energy[0] = 0
    # 서로 인접한 FFT bin을 세 번 고르지 않도록 국소 최대점에서 순위를 정합니다.
    peaks = [i for i in range(1, len(energy))
             if energy[i] > 0 and energy[i] > energy[i - 1]
             and (i == len(energy) - 1 or energy[i] >= energy[i + 1])]
    peaks.sort(key=lambda i: (-energy[i], frequencies[i]))
    for rank in range(1, 4):
        result[f"fft_peak{rank}_hz"] = float(frequencies[peaks[rank - 1]]) if len(peaks) >= rank else ""
    result["fft_ac_energy"] = float(energy.sum())
    nyquist = rate / 2
    for low, high in BANDS:
        key = f"fft_energy_{low}_{high}_hz"
        if low >= nyquist:
            result[key] = ""  # 샘플링 범위 밖을 0 에너지로 오인하지 않게 합니다.
            continue
        mask = (frequencies >= low) & (frequencies < high)
        if low < nyquist <= high:
            mask |= frequencies == nyquist
        result[key] = float(energy[mask].sum())
    return result


def process_file(path, root):
    result = {field: "" for field in META_FIELDS}
    result.update(source_file=path.relative_to(root).as_posix(), sensor="current", status="error")
    warnings = []
    try:
        parts = [norm(x) for x in path.relative_to(root).parts]
        index = parts.index("current")
        if len(parts[index:]) != 5:
            raise ValueError("폴더 구조가 current/출력/모델/고장명/파일.csv가 아닙니다")
        _, power, model, folder_label, _ = parts[index:]
        splits = [x for x in parts[:index] if x in {"Training", "Validation"}]
        result.update(split=splits[-1] if splits else "unspecified", power_kw=float(power.lower().removesuffix("kw")),
                      motor_model=model, folder_label=folder_label, label=folder_label, label_source="folder")
        try:
            lines = path.read_text(encoding="utf-8-sig").splitlines()
            result["encoding"] = "utf-8-sig"
        except UnicodeDecodeError:
            lines = path.read_text(encoding="cp949").splitlines()
            result["encoding"] = "cp949"
        metadata, samples = {}, []
        started = False
        for line_no, row in enumerate(csv.reader(lines), 1):
            if not row or not any(x.strip() for x in row):
                continue
            cells = [x.strip() for x in row]
            while cells and not cells[-1]:
                cells.pop()
            try:
                float(cells[0])
            except ValueError:
                if started:
                    raise ValueError(f"{line_no}행: 파형 도중 숫자가 아닌 행")
                metadata[norm(cells[0])] = cells[1:]
            else:
                started = True
                if len(cells) < 2:
                    raise ValueError(f"{line_no}행: 측정값 누락")
                try:
                    samples.append([float(x) for x in cells])
                except ValueError as error:
                    raise ValueError(f"{line_no}행: 숫자가 아닌 값 또는 빈 측정값") from error

        def get(key):
            return metadata.get(key, [""])[0] if metadata.get(key) else ""

        csv_label = get("Data Label")  # 원문 라벨 보존, 비교할 때만 유니코드 정규화
        result.update(csv_label=csv_label, label=csv_label or folder_label,
                      label_source="csv" if csv_label else "folder",
                      label_mismatch=bool(csv_label and norm(csv_label) != folder_label),
                      label_no=get("Label No") or get("Label_No"), recorded_at=get("Date"),
                      metadata_filename=get("Filename"), period=get("Period"),
                      declared_n_samples=get("Data Length"))
        result["device_id"] = Path(get("Filename") or path.name).stem.split("_")[0]
        spec = metadata.get("Motor Spec", [])
        result["metadata_motor_model"] = spec[0] if spec else ""
        result["metadata_power_kw"] = spec[2] if len(spec) > 2 else ""
        if spec and norm(spec[0]) != model:
            warnings.append("motor_model_mismatch")
        if len(spec) > 2 and spec[2] and float(spec[2]) != result["power_kw"]:
            warnings.append("power_kw_mismatch")
        missing = [key for key in ("Date", "Filename", "Data Label", "Sample Rate", "Data Length", "Motor Spec") if not get(key)]
        if not result["label_no"]:
            missing.append("Label No/Label_No")
        result["missing_metadata"] = "|".join(missing)
        if result["label_mismatch"]:
            warnings.append("label_mismatch")
        if not samples:
            raise ValueError("측정값이 없습니다")
        data = np.asarray(samples, dtype=float)
        n, columns = data.shape
        result.update(n_samples=n, n_channels=columns - 1)
        if n < 2 or not 1 <= columns - 1 <= 3:
            raise ValueError("샘플이 2개 미만이거나 채널 수가 1~3 범위를 벗어납니다")
        if not np.isfinite(data).all():
            raise ValueError("NaN 또는 무한대가 있습니다")
        rate = float(get("Sample Rate"))
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("샘플링 주파수가 없거나 유효하지 않습니다")
        result["sample_rate_hz"] = rate
        intervals = np.diff(data[:, 0])
        if not np.allclose(intervals, 1 / rate, rtol=1e-3, atol=1e-8):
            raise ValueError("시간 간격이 Sample Rate와 일치하지 않습니다. FFT에 균일 샘플링이 필요합니다")
        if get("Data Length") and float(get("Data Length")) != n:
            raise ValueError(f"Data Length={get('Data Length')}, 실제 샘플={n}")
        for ch in range(1, columns):
            values = data[:, ch]
            result.update({f"ch{ch}_{key}": value for key, value in features(values, rate).items()})
            if np.ptp(values) == 0:
                warnings.append(f"ch{ch}_constant_signal")
        if rate / 2 < 1000:
            warnings.append("fft_bands_clipped_at_nyquist")
        if rate / 2 > 1000:
            warnings.append("fft_energy_above_1000_hz_not_in_bands")
        result["status"] = "warning" if warnings or missing else "ok"
    except (OSError, ValueError, IndexError, OverflowError) as error:
        result["error"] = str(error)
        # 실패한 파형의 일부 특징이 남지 않도록 메타데이터만 보존합니다.
        for key in list(result):
            if re.match(r"ch\d+_", key):
                del result[key]
    result["warnings"] = "|".join(warnings)
    return result


def preprocess(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not root.is_dir():
        raise ValueError(f"데이터 폴더가 없습니다: {root}")
    if output.is_relative_to(root):
        raise ValueError("출력 파일은 원본 폴더 밖에 지정하세요")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".csv"
                   and "current" in p.relative_to(root).parts and "vibration" not in p.relative_to(root).parts)
    if not files:
        raise ValueError("current 폴더 안에 CSV가 없습니다")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.tmp")
    issues_path = output.with_name(output.stem + "_issues.csv")
    statuses, labels, models, splits = Counter(), Counter(), Counter(), Counter()
    missing_count, mismatches, issues = 0, 0, []
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for i, path in enumerate(files, 1):
            row = process_file(path, root)
            writer.writerow(row)
            statuses[row["status"]] += 1
            labels[row["label"] or "unknown"] += 1
            models[f"{row['power_kw']}kW/{row['motor_model']}"] += 1
            splits[row["split"] or "unknown"] += 1
            missing_count += bool(row["missing_metadata"])
            mismatches += bool(row["label_mismatch"])
            if row["status"] != "ok":
                issues.append({key: row[key] for key in ISSUE_FIELDS})
            if i % 500 == 0 or i == len(files):
                print(f"처리 {i:,}/{len(files):,}", flush=True)
    temporary.replace(output)
    with issues_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ISSUE_FIELDS)
        writer.writeheader()
        writer.writerows(issues)
    summary = {
        "data_root": str(root), "output": str(output), "discovered_files": len(files),
        "output_rows": len(files), "successful_files": statuses["ok"] + statuses["warning"],
        "failed_files": statuses["error"], "missing_metadata_files": missing_count,
        "label_mismatch_files": mismatches, "status_counts": dict(statuses),
        "label_counts": dict(sorted(labels.items())), "model_counts": dict(sorted(models.items())),
        "split_counts": dict(splits),
    }
    output.with_name(output.stem + "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="current를 포함한 샘플 폴더 또는 Training/Validation의 상위 폴더")
    parser.add_argument("--output", type=Path, default=Path("motor_current_features.csv"))
    args = parser.parse_args()
    root = args.data_root
    if root is None:
        candidates = list(Path(__file__).resolve().parent.glob("2020-02-125*sample"))
        if len(candidates) != 1:
            parser.error("--data-root로 데이터 폴더를 지정하세요")
        root = candidates[0]
    summary = preprocess(root, args.output)
    if summary["failed_files"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
