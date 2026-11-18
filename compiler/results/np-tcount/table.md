# T-count for the RQ1 T slide

Written by `python -m pysimcount.pftcount results/np-tcount/config.json`. Mean over seeds [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], merged, ct (eps_trotter = eps_synth = 5e-4). Measured r at every n in the runs; fitted r from each seed's own power-law fit ({'fit_sizes': [6, 7, 8, 10], 'sizes': [13, 20]}).

## T per rotation

Least squares over 2880 synthesised circuits: T/rotation = 3.042 log2(1/eps_rot) +2.13 (max |residual| 6.87); T/rotation / log2(1/eps_rot) in [2.899, 3.270].

## Equal-order T ratios at n = 10 (measured)

| Suzuki | optimised | T Suzuki | T optimised | ratio |
|---|---|---:|---:|---:|
| suzuki4 | mo-q6 | 3,814,178 | 2,579,888 | 1.48x |
| suzuki6 | mo-q14 | 4,012,921 | 1,637,724 | 2.45x |
| suzuki6 | bm-q10 | 4,012,921 | 2,178,800 | 1.84x |
| suzuki8 | mo-o8-q21 | 9,141,777 | 2,548,933 | 3.59x |
| suzuki8 | morales-y8m10 | 9,141,777 | 2,817,990 | 3.24x |
| suzuki8 | morales-m8 | 9,141,777 | 2,446,100 | 3.74x |

## Mean T-count

| formula | n=4 | n=5 | n=6 | n=7 | n=8 | n=9 | n=10 | n=13 (fit) | n=20 (fit) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| suzuki4 | 410,227 | 535,141 | 956,087 | 1,398,590 | 2,052,152 | 2,712,542 | 3,814,178 | 7,745,687 | 24,911,670 |
| mo-q6 | 298,204 | 454,337 | 719,342 | 1,005,067 | 1,389,715 | 1,875,411 | 2,579,888 | 4,878,507 | 14,282,311 |
| suzuki6 | 498,655 | 683,636 | 1,257,090 | 1,716,813 | 2,463,415 | 3,147,913 | 4,012,921 | 7,544,299 | 20,321,667 |
| mo-q14 | 199,644 | 295,456 | 517,803 | 700,110 | 982,626 | 1,292,116 | 1,637,724 | 3,004,156 | 7,987,145 |
| bm-q10 | 287,233 | 408,923 | 658,784 | 901,238 | 1,239,448 | 1,624,357 | 2,178,800 | 3,990,068 | 10,946,177 |
| suzuki8 | 1,147,832 | 1,564,733 | 2,967,788 | 4,126,735 | 5,725,360 | 7,091,892 | 9,141,777 | 16,991,266 | 44,173,940 |
| morales-m8 | 283,483 | 480,273 | 790,926 | 1,062,267 | 1,480,735 | 1,892,882 | 2,446,100 | 4,424,857 | 11,407,316 |
