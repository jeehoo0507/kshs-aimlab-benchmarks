# ImageNet-1K (ILSVRC2012)

[English](../../../../datasets/imagenet/README.md) · 한국어

<p align="center">
  <img src="../../../../assets/imagenet.png" alt="ImageNet 이미지 모자이크" width="750">
</p>

라벨이 있는 **train**과 **validation** split을 준비합니다. 이후 처음부터 학습하거나 ImageNet 사전학습 모델을 평가할 때 사용합니다. 라벨이 공개되지 않은 공식 test split은 이 과정에서 다운로드하거나 준비하지 않습니다.

## 준비

[공식 ImageNet 다운로드 페이지](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php)에 로그인하고 다음 아카이브 3개를 Git에서 제외된 `data/imagenet/archives/`에 넣으세요.

```bash
mkdir -p data/imagenet/archives
```

```text
data/imagenet/archives/
  ILSVRC2012_img_train.tar
  ILSVRC2012_img_val.tar
  ILSVRC2012_devkit_t12.tar.gz
```

저장소 루트에서 실행하세요.

```bash
bash datasets/imagenet/setup.sh
```

이 명령은 공식 아카이브의 MD5를 확인하고 Git에서 제외된 `data/imagenet/`에 이미지를 풀어 devkit 정답으로 validation 라벨을 지정한 뒤 결과를 검증합니다. 원본 아카이브를 자동 다운로드하지는 않습니다. Train 압축 해제에는 넉넉한 디스크 공간이 필요합니다. 아카이브와 준비된 이미지를 다른 드라이브에 두려면 `bash datasets/imagenet/setup.sh /mnt/archives /mnt/data/imagenet`을 실행하세요. 중단 후 재실행하면 완료된 train 클래스는 재사용합니다.

```text
data/imagenet/
  manifest.json
  train/<wnid>/*.JPEG   # 라벨이 있는 이미지 1,281,167장
  val/<wnid>/*.JPEG     # 라벨이 있는 이미지 50,000장; 클래스당 50장
```

`manifest.json`에는 정렬된 1,000개 WordNet ID, split별 이미지 수, 원본 아카이브의 SHA-256이 기록됩니다. Test 이미지나 라벨은 포함되지 않습니다. 로컬 top-1/top-5 평가는 validation split을 사용하고 결과는 **ImageNet-1K validation**으로 표기하세요. 공식 test 정확도는 [ImageNet 평가 서버](https://www.image-net.org/challenges/LSVRC/)에 예측을 제출해야 확인할 수 있습니다.

## 검증

```bash
python3 scripts/check_assets.py imagenet data/imagenet
sha256sum data/imagenet/manifest.json
```

첫 명령은 클래스 디렉터리, 파일명, split별 이미지 수를 확인합니다. 결과를 보고할 때 manifest 해시도 함께 남기세요. 공식 아카이브 이름과 validation 라벨 연결 방식은 [TorchVision의 ImageNet 구현](https://github.com/pytorch/vision/blob/main/torchvision/datasets/imagenet.py)을 참고할 수 있습니다.
