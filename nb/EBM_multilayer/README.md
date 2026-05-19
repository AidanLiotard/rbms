# EBM multilayer MNIST trainings

Chaque sous-dossier contient un notebook `01_training_analysis.ipynb` configure pour un training distant.

| folder | host | model | layers | hidden dims | run name |
|---|---:|---|---:|---|---|
| `binary_L01_1024` | `ablette` | `BEBM` | 1 | `1024` | `bebm_binary_mnist_L1_1024_ch2048_steps100_lr1e-3_100k` |
| `binary_L02_1024_512` | `anchois` | `BEBM` | 2 | `1024 512` | `bebm_binary_mnist_L2_1024_512_ch2048_steps100_lr1e-3_100k` |
| `binary_L03_1024_768_512` | `anguille` | `BEBM` | 3 | `1024 768 512` | `bebm_binary_mnist_L3_1024_768_512_ch2048_steps100_lr1e-3_100k` |
| `binary_L04_1024_768_512_256` | `barbeau` | `BEBM` | 4 | `1024 768 512 256` | `bebm_binary_mnist_L4_1024_768_512_256_ch2048_steps100_lr1e-3_100k` |
| `binary_L05_1024_768_512_256_128` | `barbue` | `BEBM` | 5 | `1024 768 512 256 128` | `bebm_binary_mnist_L5_1024_768_512_256_128_ch2048_steps100_lr1e-3_100k` |
| `binary_L10_1024_1024_768_768_512_512_384_256_128_64` | `baudroie` | `BEBM` | 10 | `1024 1024 768 768 512 512 384 256 128 64` | `bebm_binary_mnist_L10_1024_1024_768_768_512_512_384_256_128_64_ch2048_steps100_lr1e-3_100k` |
| `continuous_L01_1024` | `brochet` | `CEBM` | 1 | `1024` | `cebm_cont_mnist_L1_1024_ch2048_steps10_lr1e-3_100k` |
| `continuous_L02_1024_512` | `carrelet` | `CEBM` | 2 | `1024 512` | `cebm_cont_mnist_L2_1024_512_ch2048_steps10_lr1e-3_100k` |
| `continuous_L03_1024_768_512` | `gymnote` | `CEBM` | 3 | `1024 768 512` | `cebm_cont_mnist_L3_1024_768_512_ch2048_steps10_lr1e-3_100k` |
| `continuous_L04_1024_768_512_256` | `labre` | `CEBM` | 4 | `1024 768 512 256` | `cebm_cont_mnist_L4_1024_768_512_256_ch2048_steps10_lr1e-3_100k` |
| `continuous_L05_1024_768_512_256_128` | `lieu` | `CEBM` | 5 | `1024 768 512 256 128` | `cebm_cont_mnist_L5_1024_768_512_256_128_ch2048_steps10_lr1e-3_100k` |
| `continuous_L10_1024_1024_768_768_512_512_384_256_128_64` | `lotte` | `CEBM` | 10 | `1024 1024 768 768 512 512 384 256 128 64` | `cebm_cont_mnist_L10_1024_1024_768_768_512_512_384_256_128_64_ch2048_steps10_lr1e-3_100k` |
