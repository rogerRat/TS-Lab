# TIMERXL.md — v5-timerxl 正式结果（timerxl-84m, 15 臂 × 9 任务 × 3 seeds）

- Run: `/root/autodl-tmp/TS3/outputs/main-rank-volume-timerxl-20260920`（protocol v5-timerxl，387 unique cells，405 cell 文件含一次中断重跑）
- 模型: thuml/timer-base-84m（84,142,080 参数，bf16，96 点 patch）
- 冻结配方: lr 1e-4, epochs 4, schedule_epochs 6, batch 32, wd 0.01, cosine, betas .9/.999, eps 1e-8, max_grad_norm 1, warmup 0, min_lr_ratio 0（safe-history 两阶段标定 + 预注册扩展确认：ext6 0.91860 / ext8 0.92070 均差于 4ep 0.91699）
- MASE 为 3 seeds 均值（seed0/1/2）；uni = 轨迹内均匀化修订版（与 V5-uni 同 revision rags-intra-trajectory-uniform-v1，但为 TimerXL 独立 run 内同配方对比）

## 全局均值（9 任务平均 MASE）

| arm | mean MASE | vs original |
|---|---|---|
| orig | 1.0802 | +0.00% |
| lp-mirror | 1.1141 | -3.14% |
| rags-block-zeroClipSum | 1.1555 | -6.97% |
| rags-block-expSum-uni | 1.1578 | -7.18% |
| rags-block-maxSum | 1.1600 | -7.39% |
| rand | 1.1600 | -7.39% |
| rags-block-linearSum | 1.1605 | -7.43% |
| rags-block-expSum | 1.1617 | -7.55% |
| rags-traj-maxSum | 1.1699 | -8.31% |
| rags-block-linearSum-uni | 1.1721 | -8.51% |
| rags-traj-expSum | 1.1728 | -8.57% |
| rags-traj-linearSum | 1.1769 | -8.96% |
| rags-traj-zeroClipSum | 1.1794 | -9.19% |
| less-main | 1.1832 | -9.53% |
| tsds | 1.2901 | -19.44% |

## Timer block-vs-uni（同 run 直接对比，逐任务 uni−block，负值=uni更差/MASE更高取正? 见注）

注：Δ = mean(uni) − mean(block)，Δ>0 表示 uni 更差（MASE 更高）。

- **rags-block-expSum**: uni 更差 3/9 任务，平均 ΔMASE -0.0039
- **rags-block-linearSum**: uni 更差 5/9 任务，平均 ΔMASE +0.0117

## 逐任务结果

### time_azure_i_cpu_5t

| arm | seed0 | seed1 | seed2 | mean |
|---|---|---|---|---|
| orig | 1.2575 | 1.2575 | 1.2575 | 1.2575 |
| lp-mirror | 1.2768 | 1.2583 | 1.2644 | 1.2665 |
| tsds | 1.2476 | 1.2778 | 1.2846 | 1.2700 |
| rand | 1.2681 | 1.2798 | 1.2701 | 1.2727 |
| rags-block-maxSum | 1.2786 | 1.2865 | 1.2672 | 1.2774 |
| rags-block-zeroClipSum | 1.2669 | 1.2899 | 1.2803 | 1.2791 |
| rags-block-linearSum | 1.2871 | 1.2794 | 1.2775 | 1.2814 |
| rags-block-linearSum-uni | 1.2962 | 1.2769 | 1.2727 | 1.2819 |
| rags-block-expSum-uni | 1.2870 | 1.2812 | 1.2947 | 1.2876 |
| rags-traj-expSum | 1.2861 | 1.2933 | 1.2868 | 1.2887 |
| less-main | 1.2821 | 1.2917 | 1.3003 | 1.2914 |
| rags-traj-zeroClipSum | 1.2960 | 1.2980 | 1.2815 | 1.2918 |
| rags-block-expSum | 1.2791 | 1.2843 | 1.3131 | 1.2922 |
| rags-traj-linearSum | 1.2919 | 1.3041 | 1.2890 | 1.2950 |
| rags-traj-maxSum | 1.2851 | 1.2987 | 1.3042 | 1.2960 |

### time_cphl_h

| arm | seed0 | seed1 | seed2 | mean |
|---|---|---|---|---|
| rags-block-zeroClipSum | 1.4819 | 1.4936 | 1.4754 | 1.4836 |
| rags-block-expSum | 1.4959 | 1.4890 | 1.4875 | 1.4908 |
| orig | 1.4947 | 1.4947 | 1.4947 | 1.4947 |
| rags-block-linearSum | 1.4754 | 1.4960 | 1.5265 | 1.4993 |
| rand | 1.4971 | 1.5058 | 1.4961 | 1.4997 |
| lp-mirror | 1.5185 | 1.4945 | 1.5002 | 1.5044 |
| rags-block-maxSum | 1.5028 | 1.5075 | 1.5089 | 1.5064 |
| rags-block-linearSum-uni | 1.5243 | 1.5183 | 1.4920 | 1.5116 |
| rags-block-expSum-uni | 1.5061 | 1.5376 | 1.4918 | 1.5118 |
| rags-traj-maxSum | 1.5153 | 1.5258 | 1.5098 | 1.5170 |
| rags-traj-expSum | 1.5194 | 1.5175 | 1.5193 | 1.5187 |
| rags-traj-zeroClipSum | 1.5082 | 1.5323 | 1.5248 | 1.5217 |
| less-main | 1.5012 | 1.5460 | 1.5197 | 1.5223 |
| rags-traj-linearSum | 1.5095 | 1.5309 | 1.5287 | 1.5230 |
| tsds | 1.5742 | 1.5502 | 1.5123 | 1.5456 |

### time_darwin_water_15t

| arm | seed0 | seed1 | seed2 | mean |
|---|---|---|---|---|
| orig | 1.2582 | 1.2582 | 1.2582 | 1.2582 |
| lp-mirror | 1.2876 | 1.2721 | 1.2668 | 1.2755 |
| rags-block-maxSum | 1.5959 | 1.5936 | 1.5951 | 1.5948 |
| rand | 1.5826 | 1.6096 | 1.6089 | 1.6004 |
| rags-block-expSum | 1.6162 | 1.5987 | 1.6173 | 1.6107 |
| rags-block-linearSum-uni | 1.6046 | 1.5968 | 1.6319 | 1.6111 |
| rags-block-linearSum | 1.6132 | 1.6241 | 1.6004 | 1.6126 |
| rags-block-expSum-uni | 1.5926 | 1.6282 | 1.6214 | 1.6141 |
| rags-block-zeroClipSum | 1.6227 | 1.6306 | 1.6104 | 1.6213 |
| tsds | 1.6722 | 1.6608 | 1.6298 | 1.6543 |
| rags-traj-expSum | 1.6652 | 1.6856 | 1.6734 | 1.6747 |
| less-main | 1.6806 | 1.6748 | 1.6891 | 1.6815 |
| rags-traj-linearSum | 1.6596 | 1.6951 | 1.6942 | 1.6829 |
| rags-traj-maxSum | 1.7226 | 1.6498 | 1.6811 | 1.6845 |
| rags-traj-zeroClipSum | 1.7098 | 1.6969 | 1.7130 | 1.7066 |

### time_eweld_15t

| arm | seed0 | seed1 | seed2 | mean |
|---|---|---|---|---|
| orig | 0.9472 | 0.9472 | 0.9472 | 0.9472 |
| rags-traj-maxSum | 1.0171 | 1.0227 | 1.0393 | 1.0263 |
| rags-block-zeroClipSum | 1.0117 | 1.0375 | 1.0452 | 1.0314 |
| rags-block-expSum | 1.0137 | 1.0633 | 1.0404 | 1.0391 |
| lp-mirror | 1.0041 | 1.0520 | 1.0616 | 1.0392 |
| less-main | 1.0690 | 1.0073 | 1.0435 | 1.0400 |
| rags-block-maxSum | 1.0488 | 1.0439 | 1.0513 | 1.0480 |
| rags-traj-expSum | 1.0846 | 1.0667 | 1.0255 | 1.0590 |
| rags-traj-zeroClipSum | 1.0591 | 1.0763 | 1.0472 | 1.0609 |
| rags-traj-linearSum | 1.0839 | 1.0379 | 1.0781 | 1.0666 |
| rags-block-expSum-uni | 1.0973 | 1.0752 | 1.0552 | 1.0759 |
| rags-block-linearSum | 1.0828 | 1.1434 | 1.0216 | 1.0826 |
| rand | 1.0700 | 1.0851 | 1.1117 | 1.0890 |
| rags-block-linearSum-uni | 1.1245 | 1.1456 | 1.1229 | 1.1310 |
| tsds | 1.1005 | 1.1878 | 1.1411 | 1.1431 |

### time_finland_15t

| arm | seed0 | seed1 | seed2 | mean |
|---|---|---|---|---|
| rags-block-linearSum | 0.9252 | 0.8769 | 0.9353 | 0.9125 |
| rags-block-expSum-uni | 0.9217 | 0.9500 | 0.9045 | 0.9254 |
| rags-block-zeroClipSum | 0.9028 | 0.9082 | 1.0311 | 0.9474 |
| rags-traj-expSum | 0.9221 | 1.0327 | 0.9063 | 0.9537 |
| rags-traj-linearSum | 0.9221 | 1.0327 | 0.9063 | 0.9537 |
| rags-traj-maxSum | 0.9221 | 1.0327 | 0.9063 | 0.9537 |
| rags-traj-zeroClipSum | 0.9932 | 0.9519 | 0.9290 | 0.9580 |
| rand | 0.9272 | 0.9441 | 1.0288 | 0.9667 |
| rags-block-expSum | 0.9874 | 0.9953 | 0.9248 | 0.9692 |
| rags-block-linearSum-uni | 0.9792 | 0.8655 | 1.0916 | 0.9788 |
| rags-block-maxSum | 0.9478 | 1.0716 | 0.9405 | 0.9866 |
| orig | 1.0148 | 1.0148 | 1.0148 | 1.0148 |
| less-main | 0.9937 | 1.1133 | 1.0261 | 1.0444 |
| lp-mirror | 1.3828 | 1.1693 | 1.1509 | 1.2343 |
| tsds | 1.8821 | 1.8793 | 1.7916 | 1.8510 |

### time_nem_demand_5t

| arm | seed0 | seed1 | seed2 | mean |
|---|---|---|---|---|
| orig | 0.9597 | 0.9597 | 0.9597 | 0.9597 |
| lp-mirror | 1.0007 | 0.9329 | 0.9924 | 0.9753 |
| rand | 1.1816 | 1.1682 | 1.1766 | 1.1754 |
| rags-block-expSum-uni | 1.1860 | 1.1701 | 1.1795 | 1.1785 |
| rags-block-maxSum | 1.1975 | 1.2031 | 1.1668 | 1.1892 |
| rags-block-expSum | 1.1815 | 1.2220 | 1.1897 | 1.1977 |
| rags-block-zeroClipSum | 1.2209 | 1.1835 | 1.2054 | 1.2033 |
| rags-traj-expSum | 1.2265 | 1.1777 | 1.2075 | 1.2039 |
| rags-block-linearSum-uni | 1.2301 | 1.1843 | 1.2039 | 1.2061 |
| rags-traj-maxSum | 1.2157 | 1.1991 | 1.2077 | 1.2075 |
| rags-traj-linearSum | 1.2235 | 1.2033 | 1.2156 | 1.2141 |
| rags-traj-zeroClipSum | 1.2379 | 1.1959 | 1.2096 | 1.2145 |
| less-main | 1.2183 | 1.2239 | 1.2057 | 1.2160 |
| rags-block-linearSum | 1.2391 | 1.2242 | 1.1876 | 1.2169 |
| tsds | 1.2855 | 1.2979 | 1.2854 | 1.2896 |

### time_nem_price_5t

| arm | seed0 | seed1 | seed2 | mean |
|---|---|---|---|---|
| orig | 0.7723 | 0.7723 | 0.7723 | 0.7723 |
| lp-mirror | 0.7847 | 0.7867 | 0.7847 | 0.7854 |
| less-main | 0.8978 | 0.9015 | 0.8833 | 0.8942 |
| rand | 0.8953 | 0.8961 | 0.8933 | 0.8949 |
| rags-block-linearSum-uni | 0.9098 | 0.8890 | 0.8894 | 0.8961 |
| rags-traj-maxSum | 0.9049 | 0.8883 | 0.8986 | 0.8973 |
| rags-block-maxSum | 0.9020 | 0.9008 | 0.8944 | 0.8990 |
| rags-block-expSum-uni | 0.8967 | 0.8988 | 0.9031 | 0.8995 |
| rags-block-expSum | 0.9066 | 0.8973 | 0.8952 | 0.8997 |
| rags-block-linearSum | 0.8954 | 0.9022 | 0.9026 | 0.9001 |
| rags-block-zeroClipSum | 0.9032 | 0.9004 | 0.9048 | 0.9028 |
| rags-traj-expSum | 0.9028 | 0.9026 | 0.9113 | 0.9056 |
| tsds | 0.9050 | 0.9108 | 0.9009 | 0.9056 |
| rags-traj-zeroClipSum | 0.9018 | 0.9012 | 0.9188 | 0.9072 |
| rags-traj-linearSum | 0.8981 | 0.9031 | 0.9215 | 0.9076 |

### time_port_activity_d

| arm | seed0 | seed1 | seed2 | mean |
|---|---|---|---|---|
| rags-block-maxSum | 0.7829 | 0.7921 | 0.7864 | 0.7871 |
| rags-traj-maxSum | 0.7866 | 0.7924 | 0.7895 | 0.7895 |
| rags-block-linearSum | 0.7907 | 0.7917 | 0.7887 | 0.7903 |
| less-main | 0.7865 | 0.7962 | 0.7898 | 0.7908 |
| orig | 0.7918 | 0.7918 | 0.7918 | 0.7918 |
| rags-block-expSum-uni | 0.7887 | 0.7907 | 0.7962 | 0.7919 |
| rags-block-zeroClipSum | 0.7932 | 0.7907 | 0.7920 | 0.7919 |
| rags-traj-linearSum | 0.7961 | 0.7914 | 0.7887 | 0.7921 |
| rand | 0.7929 | 0.7906 | 0.7947 | 0.7927 |
| rags-traj-zeroClipSum | 0.7976 | 0.7926 | 0.7887 | 0.7930 |
| rags-traj-expSum | 0.7954 | 0.7956 | 0.7890 | 0.7933 |
| rags-block-linearSum-uni | 0.7915 | 0.7955 | 0.7932 | 0.7934 |
| lp-mirror | 0.7967 | 0.7931 | 0.7912 | 0.7937 |
| rags-block-expSum | 0.7974 | 0.7958 | 0.7915 | 0.7949 |
| tsds | 0.8002 | 0.8072 | 0.7959 | 0.8011 |

### time_solar_h

| arm | seed0 | seed1 | seed2 | mean |
|---|---|---|---|---|
| rags-block-expSum-uni | 1.1324 | 1.1583 | 1.1148 | 1.1352 |
| rags-block-zeroClipSum | 1.1381 | 1.1489 | 1.1293 | 1.1388 |
| rags-block-linearSum-uni | 1.1298 | 1.1242 | 1.1637 | 1.1392 |
| rags-block-linearSum | 1.1507 | 1.1479 | 1.1470 | 1.1485 |
| rand | 1.1690 | 1.1395 | 1.1384 | 1.1490 |
| tsds | 1.1293 | 1.1755 | 1.1482 | 1.1510 |
| rags-block-maxSum | 1.1359 | 1.1698 | 1.1483 | 1.1513 |
| lp-mirror | 1.1507 | 1.1709 | 1.1362 | 1.1526 |
| rags-traj-expSum | 1.1747 | 1.1462 | 1.1513 | 1.1574 |
| rags-traj-linearSum | 1.1747 | 1.1462 | 1.1513 | 1.1574 |
| rags-traj-maxSum | 1.1747 | 1.1462 | 1.1513 | 1.1574 |
| rags-block-expSum | 1.1503 | 1.1634 | 1.1694 | 1.1610 |
| rags-traj-zeroClipSum | 1.1772 | 1.1478 | 1.1586 | 1.1612 |
| less-main | 1.1670 | 1.1597 | 1.1770 | 1.1679 |
| orig | 1.2255 | 1.2255 | 1.2255 | 1.2255 |

## 与 chronos/timesfm 的同臂全局对比（9 任务平均 MASE）

来源：V5 主 run（main-rank-volume-20260918）与 V5-uni run（main-rank-volume-uni-20260920）cells 重新聚合；orig 每任务单 seed 广播；跨模型 MASE 水平受模型能力影响，应比较**臂间排序形态**而非绝对值。

| arm | chronos-base (V5) | timesfm-2.5 (V5) | timerxl-84m (本 run) |
|---|---|---|---|
| orig | 1.4424 | 0.8918 | **1.0802** |
| rand | 1.2806 | 0.8665 | 1.1600 |
| tsds | 1.2608 | 0.8690 | 1.2901 |
| less-main | 1.2702 | 0.8626 | 1.1832 |
| lp-mirror | 1.3965 | 0.8896 | **1.1141** |
| rags-block-expSum | **1.2563** | **0.8544** | 1.1617 |
| rags-block-linearSum | 1.2712 | 0.8553 | 1.1605 |
| rags-block-maxSum | 1.2852 | 0.8629 | 1.1600 |
| rags-block-zeroClipSum | 1.2658 | 0.8598 | 1.1555 |
| rags-traj-expSum | 1.2711 | 0.8635 | 1.1728 |
| rags-traj-linearSum | 1.2814 | 0.8618 | 1.1769 |
| rags-traj-maxSum | 1.2767 | 0.8614 | 1.1699 |
| rags-traj-zeroClipSum | 1.2884 | 0.8607 | 1.1794 |
| rags-block-expSum-uni | 1.3037 (UNI run) | 0.8691 (UNI run) | 1.1578 |
| rags-block-linearSum-uni | 1.3328 (UNI run) | 0.8675 (UNI run) | 1.1721 |

观察（描述性，非因果）：
- chronos/timesfm 上 orig 为全局最差、RAGS block 臂最优；**TimerXL 上形态反转——orig 最优，全部选择臂差于 orig**，lp-mirror 为最佳选择臂（-3.14% vs orig）。三模型间排序形态不一致，是机制实验（reference-census H1–H7）待解释的异质性。
- TimerXL 同 run block-vs-uni：expSum uni 平均 −0.0039（uni 略好，与 chronos/timesfm 的 V5-uni 结论方向相反），linearSum uni 平均 +0.0117（uni 更差，同向）。逐任务见上节。
