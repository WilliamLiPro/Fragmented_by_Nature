# Fragmented_by_Nature
Data sources for three novel worldwide urban indexes--the share of natural barriers, average dyadic nonconvexity, and the average road detour.

Research Paper: Fragmented by Nature: Metropolitan Geography,Inner Urban Connectivity, and Environmental Outcomes

Author: Luyao Wang, Albert Saiz, Weipeng Li

* Please run "pip install -r requirements.txt" to install the dependencies before executing any python file.

## Share of barrier
Share of barrier: The share of geographic barriers in certain region.

## Nonconvexity
Definition: 

* You can run the "[nonconvexity/demo_nonconvexity.py](nonconvexity/demo_nonconvexity.py)" directly. 

Here is the [output](result/global_nonconvexity/demo_data-r10km nonconvexity summary.csv):
<details>
<summary>demo_data-r10km nonconvexity summary.csv</summary>

| areaID | share_of_barrier | nonconvexity | number of commuting nodes |
|-------:|------------------|--------------|---------------------------|
|       1| 0.2185 | 0.0136 | 974     		|
|       2| 0.009491| 0.004986| 1233           	|
|       3| 0.0006421 | 8.3345e-07 | 1246       	|

</details>

* For other experiments, please download and prepocess the datasets according to our Appendix.

## Detour
Definition: 

* You can run the "[detour/demo_detour.py](detour/demo_detour.py)" directly.

Here is the [output](result/detour/demo-r10km detour summary.csv):
<details>
<summary>demo-r10km detour summary</summary>
| areaID | share_of_barrier | detour | distance_max | distance_mean | distance_std | number of commuting nodes | number of road nodes |
|-----:|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
|     1| 0.2185 | 0.2665 | 27.5856 | 10.5261 | 5.0514 | 948 | 31498|
|     2| 0.009491 | 0.3590 | 23.8956 | 7.8960 | 3.9693 | 313 | 6301 |
|     3| 0.0006421 | 0.1925 | 26.5877 | 10.7287 | 5.0226 | 1224 | 68350       |

* For other experiments, please download and prepocess the datasets according to our Appendix.