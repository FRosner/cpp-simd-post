```
----------------------------------------------------------------------------------
Benchmark                        Time             CPU   Iterations UserCounters...
----------------------------------------------------------------------------------
BM_DdotOpenBLAS/1024           135 ns          135 ns      5355777 items_per_second=7.56875G/s
BM_DdotOpenBLAS/2048           257 ns          257 ns      2815791 items_per_second=7.98207G/s
BM_DdotOpenBLAS/4096           483 ns          483 ns      1460975 items_per_second=8.47495G/s
BM_DdotOpenBLAS/8192          1251 ns         1250 ns       579269 items_per_second=6.55139G/s
BM_DdotOpenBLAS/16384        45638 ns        14307 ns        48076 items_per_second=1.14515G/s
BM_DdotOpenBLAS/32768        47031 ns        14538 ns        45665 items_per_second=2.25401G/s
BM_DdotOpenBLAS/65536        47323 ns        14208 ns        55291 items_per_second=4.61276G/s
BM_DdotOpenBLAS/131072       47919 ns        14625 ns        48657 items_per_second=8.96249G/s
BM_DdotOpenBLAS/262144       54794 ns        17069 ns        41808 items_per_second=15.3579G/s
BM_DdotOpenBLAS/524288       62707 ns        21132 ns        34234 items_per_second=24.8098G/s
BM_DdotOpenBLAS/1048576      79825 ns        31166 ns        22118 items_per_second=33.6444G/s
BM_DdotOpenBLAS/2097152     271153 ns       161546 ns         4224 items_per_second=12.9818G/s
BM_DdotOpenBLAS/4194304     640566 ns       459757 ns         1450 items_per_second=9.12287G/s
```

```
------------------------------------------------------------------------------------
Benchmark                          Time             CPU   Iterations UserCounters...
------------------------------------------------------------------------------------
BM_DdotAccelerate/1024           106 ns          106 ns      6584393 items_per_second=9.62218G/s
BM_DdotAccelerate/2048           128 ns          128 ns      5368633 items_per_second=15.9664G/s
BM_DdotAccelerate/4096           168 ns          168 ns      4228636 items_per_second=24.4352G/s
BM_DdotAccelerate/8192           248 ns          248 ns      2815078 items_per_second=33.0268G/s
BM_DdotAccelerate/16384          407 ns          407 ns      1714107 items_per_second=40.2533G/s
BM_DdotAccelerate/32768          722 ns          722 ns       966704 items_per_second=45.3601G/s
BM_DdotAccelerate/65536         1368 ns         1368 ns       511924 items_per_second=47.9184G/s
BM_DdotAccelerate/131072        2640 ns         2640 ns       265949 items_per_second=49.6561G/s
BM_DdotAccelerate/262144        5220 ns         5219 ns       134200 items_per_second=50.2301G/s
BM_DdotAccelerate/524288       10346 ns        10346 ns        67692 items_per_second=50.6731G/s
BM_DdotAccelerate/1048576      56345 ns        56345 ns        12598 items_per_second=18.6098G/s
BM_DdotAccelerate/2097152     321763 ns       321760 ns         2242 items_per_second=6.51774G/s
BM_DdotAccelerate/4194304     661364 ns       661364 ns         1057 items_per_second=6.3419G/s
```