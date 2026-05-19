# EBM multilayer MNIST trainings

Chaque sous-dossier contient un notebook `01_training_analysis.ipynb` configure pour inspecter un training. Les notebooks actifs ci-dessous pointent vers les runs Adam avec initialisation Xavier, champ visible initialise sur les marginales, output bias retire, et calibration de la derniere couche.

| folder | host | model | layers | hidden dims | run stem |
|---|---:|---|---:|---|---|
| `binary_L01_1024` | `carmor` | `BEBM` | 1 | `448` | `bebm_adam_initcalib_mnist_L1_448_ch2048_steps100_lr3e-4_100k_nsave1000` |
| `binary_L02_1024_512` | `charente` | `BEBM` | 2 | `256 128` | `bebm_adam_initcalib_mnist_L2_256_128_ch2048_steps100_lr3e-4_100k_nsave1000` |
| `binary_L03_1024_768_512` | `corvette` | `BEBM` | 3 | `256 128 64` | `bebm_adam_initcalib_mnist_L3_256_128_64_ch2048_steps100_lr3e-4_100k_nsave1000` |
| `binary_L05_1024_768_512_384_256` | `ferrari` | `BEBM` | 5 | `160 128 96 64 32` | `bebm_adam_initcalib_mnist_L5_160_128_96_64_32_ch2048_steps100_lr3e-4_100k_nsave1000` |
| `continuous_L01_1024` | `barbue` | `CEBM` | 1 | `448` | `cebm_adam_initcalib_mnist_L1_448_ch2048_steps50_lr5e-4_100k_nsave1000` |
| `continuous_L02_1024_512` | `bengali` | `CEBM` | 2 | `256 128` | `cebm_adam_initcalib_mnist_L2_256_128_ch2048_steps50_lr5e-4_100k_nsave1000` |
| `continuous_L03_1024_768_512` | `brochet` | `CEBM` | 3 | `256 128 64` | `cebm_adam_initcalib_mnist_L3_256_128_64_ch2048_steps50_lr5e-4_100k_nsave1000` |
| `continuous_L05_1024_768_512_384_256` | `labre` | `CEBM` | 5 | `160 128 96 64 32` | `cebm_adam_initcalib_mnist_L5_160_128_96_64_32_ch2048_steps50_lr5e-4_100k_nsave1000` |

Les noms de certains dossiers gardent l'ancienne convention `1024/...`, mais les notebooks et les fichiers `.h5` qu'ils chargent correspondent bien aux nouvelles petites architectures.
