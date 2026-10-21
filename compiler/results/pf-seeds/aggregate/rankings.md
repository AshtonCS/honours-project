# Cost-order reversals relative to seed 1 (merged counts)

A row means: on the reference seed formula a is cheaper than b at this n; on the listed seed it is more expensive. Ties are not counted.

94 reversals in total; 10 between formulas of equal order.

## Cheapest formula per seed

| metric | gate set | n | r from | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
|---|---|---:|---|---|---|---|---|---|
| cnot | cz | 4 | measured | mo-q14 | mo-q14 | mo-q14 | mo-q14 | mo-q14 |
| cnot | cz | 5 | measured | mo-q14 | mo-q14 | mo-q14 | mo-q14 | mo-q14 |
| cnot | cz | 6 | measured | mo-q14 | mo-q14 | mo-q14 | mo-q14 | mo-q14 |
| cnot | cz | 7 | measured | mo-q14 | mo-q14 | mo-q14 | mo-q14 | mo-q14 |
| cnot | cz | 8 | measured | mo-q14 | mo-q14 | mo-q14 | mo-q14 | mo-q14 |
| cnot | cz | 10 | measured | mo-q14 | mo-q14 | mo-q14 | mo-q14 | mo-q14 |
| cnot | cz | 13 | fit | mo-q14 | mo-q14 | mo-q14 | mo-q14 | mo-q14 |
| cnot | cz | 20 | fit | mo-q14 | mo-q14 | mo-q14 | mo-q14 | mo-q14 |
| total | cz | 4 | measured | mo-q14 | mo-q14 | mo-q14 | mo-q14 | mo-q14 |
| total | cz | 5 | measured | mo-q14 | mo-q14 | mo-q14 | mo-q14 | mo-q14 |
| total | cz | 6 | measured | mo-q14 | mo-q14 | mo-q14 | mo-q14 | mo-q14 |
| total | cz | 7 | measured | mo-q14 | mo-q14 | mo-q14 | mo-q14 | mo-q14 |
| total | cz | 8 | measured | mo-q14 | mo-q14 | mo-q14 | mo-q14 | mo-q14 |
| total | cz | 10 | measured | mo-q14 | mo-q14 | mo-q14 | mo-q14 | mo-q14 |
| total | cz | 13 | fit | mo-q14 | mo-q14 | mo-q14 | mo-q14 | mo-q14 |
| total | cz | 20 | fit | mo-q14 | mo-q14 | mo-q14 | mo-q14 | mo-q14 |
| t | ct | 4 | measured | mo-q14 | mo-q14 | mo-q14 | mo-q14 | mo-q14 |
| t | ct | 5 | measured | mo-q14 | mo-q14 | mo-q14 | mo-q14 | mo-q14 |
| t | ct | 6 | measured | mo-q14 | mo-q14 | mo-q14 | mo-q14 | mo-q14 |
| t | ct | 7 | measured | mo-q14 | mo-q14 | mo-q14 | mo-q14 | mo-q14 |
| t | ct | 8 | measured | mo-q14 | mo-q14 | mo-q14 | mo-q14 | mo-q14 |
| t | ct | 10 | measured | mo-q14 | mo-q14 | mo-q14 | mo-q14 | mo-q14 |
| t | ct | 13 | fit | mo-q14 | mo-q14 | mo-q14 | mo-q14 | mo-q14 |
| t | ct | 20 | fit | mo-q14 | mo-q14 | mo-q14 | mo-q14 | mo-q14 |

## Equal-order reversals

| metric | gate set | n | r from | cheaper on seed 1 | other | orders | seed | seed 1: a / b | seed: a / b |
|---|---|---:|---|---|---|---|---:|---|---|
| cnot | cz | 6 | measured | morales-m8 | mo-o8-q21 | 8/8 | 4 | 15,732 / 16,320 | 15,732 / 15,300 |
| cnot | cz | 7 | measured | morales-m8 | mo-o8-q21 | 8/8 | 3 | 21,252 / 22,610 | 20,286 / 20,230 |
| total | cz | 6 | measured | morales-m8 | mo-o8-q21 | 8/8 | 4 | 80,256 / 83,328 | 80,256 / 78,120 |
| total | cz | 7 | measured | morales-m8 | mo-o8-q21 | 8/8 | 3 | 108,416 / 115,444 | 103,488 / 103,292 |
| t | ct | 6 | measured | morales-y8m10 | morales-m8 | 8/8 | 2 | 825,520 / 829,206 | 925,642 / 786,280 |
| t | ct | 6 | measured | morales-y8m10 | morales-m8 | 8/8 | 3 | 825,520 / 829,206 | 825,112 / 786,880 |
| t | ct | 6 | measured | morales-y8m10 | morales-m8 | 8/8 | 4 | 825,520 / 829,206 | 873,468 / 786,560 |
| t | ct | 6 | measured | morales-y8m10 | morales-m8 | 8/8 | 5 | 825,520 / 829,206 | 873,612 / 786,360 |
| t | ct | 8 | measured | morales-m8 | mo-o8-q21 | 8/8 | 4 | 1,529,304 / 1,603,008 | 1,529,080 / 1,472,372 |
| t | ct | 8 | measured | morales-m8 | mo-o8-q21 | 8/8 | 5 | 1,529,304 / 1,603,008 | 1,530,200 / 1,469,952 |

## All reversals

| metric | gate set | n | r from | cheaper on seed 1 | other | orders | seed | seed 1: a / b | seed: a / b |
|---|---|---:|---|---|---|---|---:|---|---|
| cnot | cz | 4 | measured | morales-m8 | bm-q10 | 8/6 | 2 | 5,520 / 5,576 | 6,072 / 5,904 |
| cnot | cz | 4 | measured | morales-m8 | bm-q10 | 8/6 | 3 | 5,520 / 5,576 | 6,072 / 5,576 |
| cnot | cz | 4 | measured | morales-m8 | bm-q10 | 8/6 | 4 | 5,520 / 5,576 | 6,072 / 5,576 |
| cnot | cz | 4 | measured | morales-m8 | mo-q6 | 8/4 | 2 | 5,520 / 5,800 | 6,072 / 6,000 |
| cnot | cz | 4 | measured | morales-m8 | mo-q6 | 8/4 | 3 | 5,520 / 5,800 | 6,072 / 6,000 |
| cnot | cz | 4 | measured | morales-m8 | mo-q6 | 8/4 | 4 | 5,520 / 5,800 | 6,072 / 6,000 |
| cnot | cz | 4 | measured | morales-y8m10 | suzuki4 | 8/4 | 2 | 7,480 / 7,560 | 8,160 / 7,728 |
| cnot | cz | 4 | measured | morales-y8m10 | suzuki4 | 8/4 | 3 | 7,480 / 7,560 | 7,480 / 7,392 |
| cnot | cz | 4 | measured | morales-y8m10 | suzuki4 | 8/4 | 5 | 7,480 / 7,560 | 7,480 / 7,392 |
| cnot | cz | 5 | measured | mo-q6 | mo-o8-q21 | 4/8 | 3 | 8,750 / 9,350 | 8,750 / 8,500 |
| cnot | cz | 5 | measured | mo-q6 | mo-o8-q21 | 4/8 | 4 | 8,750 / 9,350 | 8,750 / 8,500 |
| cnot | cz | 5 | measured | mo-q6 | mo-o8-q21 | 4/8 | 5 | 8,750 / 9,350 | 8,750 / 8,500 |
| cnot | cz | 5 | measured | morales-m8 | suzuki4 | 8/4 | 3 | 10,350 / 11,130 | 9,660 / 9,450 |
| cnot | cz | 5 | measured | morales-y8m10 | suzuki4 | 8/4 | 3 | 11,050 / 11,130 | 10,200 / 9,450 |
| cnot | cz | 5 | measured | morales-y8m10 | suzuki4 | 8/4 | 4 | 11,050 / 11,130 | 10,200 / 9,870 |
| cnot | cz | 5 | measured | morales-y8m10 | suzuki4 | 8/4 | 5 | 11,050 / 11,130 | 10,200 / 9,870 |
| cnot | cz | 6 | measured | morales-m8 | mo-o8-q21 | 8/8 | 4 | 15,732 / 16,320 | 15,732 / 15,300 |
| cnot | cz | 6 | measured | morales-y8m10 | suzuki4 | 8/4 | 3 | 17,340 / 17,388 | 16,320 / 15,876 |
| cnot | cz | 7 | measured | morales-m8 | mo-o8-q21 | 8/8 | 3 | 21,252 / 22,610 | 20,286 / 20,230 |
| cnot | cz | 10 | measured | mo-q6 | mo-o8-q21 | 4/8 | 5 | 44,500 / 47,600 | 45,000 / 44,200 |
| cnot | cz | 10 | measured | morales-m8 | mo-q6 | 8/4 | 2 | 44,160 / 44,500 | 46,920 / 44,500 |
| cnot | cz | 10 | measured | morales-m8 | mo-q6 | 8/4 | 3 | 44,160 / 44,500 | 44,160 / 42,500 |
| cnot | cz | 10 | measured | morales-m8 | mo-q6 | 8/4 | 4 | 44,160 / 44,500 | 44,160 / 43,000 |
| cnot | cz | 13 | fit | mo-o8-q21 | mo-q6 | 8/4 | 2 | 83,980 / 85,150 | 90,610 / 83,200 |
| cnot | cz | 13 | fit | mo-o8-q21 | mo-q6 | 8/4 | 3 | 83,980 / 85,150 | 81,770 / 74,100 |
| cnot | cz | 13 | fit | mo-o8-q21 | mo-q6 | 8/4 | 4 | 83,980 / 85,150 | 81,770 / 77,350 |
| cnot | cz | 13 | fit | morales-m8 | mo-q6 | 8/4 | 3 | 77,142 / 85,150 | 78,936 / 74,100 |
| cnot | cz | 20 | fit | mo-o8-q21 | mo-q6 | 8/4 | 3 | 204,000 / 241,000 | 210,800 / 198,000 |
| cnot | cz | 20 | fit | mo-o8-q21 | mo-q6 | 8/4 | 4 | 204,000 / 241,000 | 207,400 / 207,000 |
| cnot | cz | 20 | fit | morales-y8m10 | mo-q6 | 8/4 | 2 | 221,000 / 241,000 | 275,400 / 242,000 |
| cnot | cz | 20 | fit | morales-y8m10 | mo-q6 | 8/4 | 3 | 221,000 / 241,000 | 221,000 / 198,000 |
| cnot | cz | 20 | fit | morales-y8m10 | mo-q6 | 8/4 | 4 | 221,000 / 241,000 | 227,800 / 207,000 |
| cnot | cz | 20 | fit | suzuki6 | suzuki4 | 6/4 | 5 | 355,520 / 371,280 | 363,600 / 336,000 |
| total | cz | 4 | measured | morales-m8 | bm-q10 | 8/6 | 2 | 28,160 / 28,356 | 30,976 / 30,024 |
| total | cz | 4 | measured | morales-m8 | bm-q10 | 8/6 | 3 | 28,160 / 28,356 | 30,976 / 28,356 |
| total | cz | 4 | measured | morales-m8 | bm-q10 | 8/6 | 4 | 28,160 / 28,356 | 30,976 / 28,356 |
| total | cz | 4 | measured | morales-m8 | mo-q6 | 8/4 | 2 | 28,160 / 29,348 | 30,976 / 30,360 |
| total | cz | 4 | measured | morales-m8 | mo-q6 | 8/4 | 3 | 28,160 / 29,348 | 30,976 / 30,360 |
| total | cz | 4 | measured | morales-m8 | mo-q6 | 8/4 | 4 | 28,160 / 29,348 | 30,976 / 30,360 |
| total | cz | 5 | measured | mo-q6 | mo-o8-q21 | 4/8 | 3 | 44,275 / 47,740 | 44,275 / 43,400 |
| total | cz | 5 | measured | mo-q6 | mo-o8-q21 | 4/8 | 4 | 44,275 / 47,740 | 44,275 / 43,400 |
| total | cz | 5 | measured | mo-q6 | mo-o8-q21 | 4/8 | 5 | 44,275 / 47,740 | 44,275 / 43,400 |
| total | cz | 5 | measured | morales-m8 | suzuki4 | 8/4 | 3 | 52,800 / 56,180 | 49,280 / 47,700 |
| total | cz | 5 | measured | suzuki4 | morales-y8m10 | 4/8 | 2 | 56,180 / 56,420 | 53,000 / 52,080 |
| total | cz | 6 | measured | morales-m8 | mo-o8-q21 | 8/8 | 4 | 80,256 / 83,328 | 80,256 / 78,120 |
| total | cz | 6 | measured | suzuki4 | morales-y8m10 | 4/8 | 2 | 87,768 / 88,536 | 90,312 / 88,536 |
| total | cz | 6 | measured | suzuki4 | morales-y8m10 | 4/8 | 4 | 87,768 / 88,536 | 85,224 / 83,328 |
| total | cz | 6 | measured | suzuki4 | morales-y8m10 | 4/8 | 5 | 87,768 / 88,536 | 83,952 / 83,328 |
| total | cz | 7 | measured | morales-m8 | mo-o8-q21 | 8/8 | 3 | 108,416 / 115,444 | 103,488 / 103,292 |
| total | cz | 7 | measured | morales-y8m10 | suzuki4 | 8/4 | 3 | 121,520 / 139,496 | 115,444 / 114,268 |
| total | cz | 10 | measured | mo-q6 | mo-o8-q21 | 4/8 | 5 | 225,170 / 243,040 | 227,700 / 225,680 |
| total | cz | 10 | measured | mo-q6 | morales-m8 | 4/8 | 5 | 225,170 / 225,280 | 227,700 / 218,240 |
| total | cz | 13 | fit | mo-o8-q21 | mo-q6 | 8/4 | 2 | 428,792 / 430,859 | 462,644 / 420,992 |
| total | cz | 13 | fit | mo-o8-q21 | mo-q6 | 8/4 | 3 | 428,792 / 430,859 | 417,508 / 374,946 |
| total | cz | 13 | fit | mo-o8-q21 | mo-q6 | 8/4 | 4 | 428,792 / 430,859 | 417,508 / 391,391 |
| total | cz | 13 | fit | morales-m8 | mo-q6 | 8/4 | 3 | 393,536 / 430,859 | 402,688 / 374,946 |
| total | cz | 20 | fit | mo-o8-q21 | mo-q6 | 8/4 | 3 | 1,041,600 / 1,219,460 | 1,076,320 / 1,001,880 |
| total | cz | 20 | fit | mo-o8-q21 | mo-q6 | 8/4 | 4 | 1,041,600 / 1,219,460 | 1,058,960 / 1,047,420 |
| total | cz | 20 | fit | morales-y8m10 | mo-q6 | 8/4 | 2 | 1,128,400 / 1,219,460 | 1,406,160 / 1,224,520 |
| total | cz | 20 | fit | morales-y8m10 | mo-q6 | 8/4 | 3 | 1,128,400 / 1,219,460 | 1,128,400 / 1,001,880 |
| total | cz | 20 | fit | morales-y8m10 | mo-q6 | 8/4 | 4 | 1,128,400 / 1,219,460 | 1,163,120 / 1,047,420 |
| total | cz | 20 | fit | suzuki6 | suzuki4 | 6/4 | 5 | 1,816,320 / 1,874,080 | 1,857,600 / 1,696,000 |
| t | ct | 5 | measured | mo-o8-q21 | mo-q6 | 8/4 | 4 | 472,728 / 479,136 | 472,704 / 468,384 |
| t | ct | 5 | measured | mo-q6 | morales-m8 | 4/8 | 3 | 479,136 / 508,064 | 478,632 / 476,550 |
| t | ct | 5 | measured | mo-q6 | morales-m8 | 4/8 | 5 | 479,136 / 508,064 | 478,464 / 476,550 |
| t | ct | 6 | measured | morales-y8m10 | morales-m8 | 8/8 | 2 | 825,520 / 829,206 | 925,642 / 786,280 |
| t | ct | 6 | measured | morales-y8m10 | morales-m8 | 8/8 | 3 | 825,520 / 829,206 | 825,112 / 786,880 |
| t | ct | 6 | measured | morales-y8m10 | morales-m8 | 8/8 | 4 | 825,520 / 829,206 | 873,468 / 786,560 |
| t | ct | 6 | measured | morales-y8m10 | morales-m8 | 8/8 | 5 | 825,520 / 829,206 | 873,612 / 786,360 |
| t | ct | 7 | measured | morales-m8 | mo-q6 | 8/4 | 2 | 1,074,192 / 1,085,184 | 1,073,640 / 943,920 |
| t | ct | 7 | measured | morales-m8 | mo-q6 | 8/4 | 3 | 1,074,192 / 1,085,184 | 1,017,676 / 943,236 |
| t | ct | 7 | measured | morales-m8 | mo-q6 | 8/4 | 4 | 1,074,192 / 1,085,184 | 1,073,318 / 1,018,944 |
| t | ct | 8 | measured | mo-q6 | mo-o8-q21 | 4/8 | 5 | 1,535,040 / 1,603,008 | 1,514,436 / 1,469,952 |
| t | ct | 8 | measured | morales-m8 | mo-o8-q21 | 8/8 | 4 | 1,529,304 / 1,603,008 | 1,529,080 / 1,472,372 |
| t | ct | 8 | measured | morales-m8 | mo-o8-q21 | 8/8 | 5 | 1,529,304 / 1,603,008 | 1,530,200 / 1,469,952 |
| t | ct | 8 | measured | morales-m8 | mo-q6 | 8/4 | 2 | 1,529,304 / 1,535,040 | 1,528,408 / 1,419,120 |
| t | ct | 8 | measured | morales-m8 | mo-q6 | 8/4 | 3 | 1,529,304 / 1,535,040 | 1,469,664 / 1,342,188 |
| t | ct | 8 | measured | morales-m8 | mo-q6 | 8/4 | 4 | 1,529,304 / 1,535,040 | 1,529,080 / 1,435,896 |
| t | ct | 8 | measured | morales-m8 | mo-q6 | 8/4 | 5 | 1,529,304 / 1,535,040 | 1,530,200 / 1,514,436 |
| t | ct | 10 | measured | mo-q6 | mo-o8-q21 | 4/8 | 3 | 2,628,360 / 2,639,650 | 2,534,292 / 2,479,790 |
| t | ct | 10 | measured | mo-q6 | mo-o8-q21 | 4/8 | 4 | 2,628,360 / 2,639,650 | 2,527,152 / 2,477,992 |
| t | ct | 10 | measured | mo-q6 | mo-o8-q21 | 4/8 | 5 | 2,628,360 / 2,639,650 | 2,710,096 / 2,481,066 |
| t | ct | 10 | measured | mo-q6 | morales-y8m10 | 4/8 | 5 | 2,628,360 / 2,865,324 | 2,710,096 / 2,667,426 |
| t | ct | 13 | fit | mo-o8-q21 | mo-q6 | 8/4 | 2 | 4,947,150 / 5,223,988 | 5,324,408 / 5,145,504 |
| t | ct | 13 | fit | mo-o8-q21 | mo-q6 | 8/4 | 3 | 4,947,150 / 5,223,988 | 4,582,080 / 4,557,904 |
| t | ct | 13 | fit | mo-q6 | morales-y8m10 | 4/8 | 5 | 5,223,988 / 5,456,042 | 5,092,626 / 4,730,416 |
| t | ct | 13 | fit | suzuki4 | suzuki6 | 4/6 | 3 | 7,874,944 / 7,911,792 | 7,344,590 / 7,163,868 |
| t | ct | 13 | fit | suzuki4 | suzuki6 | 4/6 | 4 | 7,874,944 / 7,911,792 | 7,444,696 / 7,422,650 |
| t | ct | 20 | fit | morales-m8 | bm-q10 | 8/6 | 2 | 10,671,120 / 11,440,128 | 12,314,760 / 11,822,636 |
| t | ct | 20 | fit | morales-m8 | bm-q10 | 8/6 | 3 | 10,671,120 / 11,440,128 | 10,671,984 / 9,818,616 |
| t | ct | 20 | fit | morales-m8 | bm-q10 | 8/6 | 4 | 10,671,120 / 11,440,128 | 11,590,810 / 10,773,840 |
| t | ct | 20 | fit | morales-y8m10 | mo-q6 | 8/4 | 2 | 14,967,520 / 15,456,984 | 16,233,532 / 15,586,560 |
| t | ct | 20 | fit | morales-y8m10 | mo-q6 | 8/4 | 3 | 14,967,520 / 15,456,984 | 13,071,100 / 12,974,808 |
| t | ct | 20 | fit | morales-y8m10 | mo-q6 | 8/4 | 4 | 14,967,520 / 15,456,984 | 13,241,216 / 13,194,808 |
