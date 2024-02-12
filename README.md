# Fragmented by Nature
Data sources for three novel worldwide urban indexes--the share of natural barriers, average dyadic nonconvexity, and the average road detour.

Research Paper: Fragmented by Nature: Metropolitan Geography,Inner Urban Connectivity, and Environmental Outcomes

Author: Luyao Wang, Albert Saiz, Weipeng Li

* This code should be run in a python environment.
* Please run "pip install -r requirements.txt" to install the dependencies before executing any python file.

## Share of barrier
Share of barrier: The share of geographic barriers in certain region. It is calculated in preprocessing of both "Nonconvexity" and "Detour".

## Nonconvexity
The nonconvexity index is designed to assess the impact of geographic barriers on the direct-line connectivity between various locations within an urban area. To calculate this index, we initially generate a set of random points, evenly distributed across the urban footprint, excluding those points that fall within barrier areas. These points are spaced at intervals greater than 200 meters, with each city having approximately 1,000 to 3,000 points to ensure comprehensive coverage of the study area. Subsequently, we create straight lines connecting each pair of points. These lines are then overlaid with the geographic barriers within the designated i km buffer zone. This process allows us to calculate the length of each line that intersects with barriers. We then determine the proportion of intersected length for each line. The average proportion of line lengths intersected by barriers is calculated to establish the nonconvexity index for each urban area.

$$NC=\frac{1}{n^{2} } \sum \sum \frac{SLDI_{pipj} }{SLD_{pipj} },$$

where $SLD_{pipj}$ denotes the length of straight line distance between point $p_i$ and $p_j$, $SLDI_{pipj}$ denotes the length of lines intersected by geographic barriers between point $p_i$ and $p_j$, n denotes the number of points located within the urban center at an interval of 5 or 10 km, and NC denotes the nonconvexity index of the urban center.


* You can run the "[nonconvexity/demo_nonconvexity.py](nonconvexity/demo_nonconvexity.py)" directly. 

Here is the output of demo_nonconvexity.py:

<summary>demo_data-r10km nonconvexity summary.csv</summary>

| areaID | share_of_barrier | nonconvexity | number of commuting nodes |
|-------:|------------------|--------------|---------------------------|
|       1| 0.2185 | 0.0136 | 974     		|
|       2| 0.009491| 0.004986| 1233           	|
|       3| 0.0006421 | 8.3345e-07 | 1246       	|

* For other experiments, please download and prepocess the datasets according to our Appendix.

## Detour
The detour index is designed to estimate the minimum additional driving distances incurred due to geographic barriers and the layout of the street/road network. To calculate this index, we start by randomly generating points within the urban area at 500-meter intervals, excluding areas classified as barriers. For each pair of points, we calculate both the Euclidean distance (the straight-line distance) and the minimum road distance. The minimum road distance is defined as the length of the shortest possible route via roads (such as streets, bridges, and tunnels) connecting the origin and destination points. The difference between the straight-line distance and the minimum road distance represents the detour for each pair of points. The detour index for a city is then determined by calculating the average detour across all pairs of points:


$$detour_{ij}=\frac{d_{ij}}{D_{ij}}-1,$$

$$Detour=\frac{\sum detour_{ij}}{n^{2}},$$

where $d_{ij}$ is the minimum road distance, $D_{ij}$ is the Euclidean distance between them, and $detour_{ij}$ is the detour index between $p_i$ and $p_j$. n represents the number of points within 10 km (or 5 km) of the city center and Detour represent the average detour index of each city.


Notice: since the detour is defined on existing road map, here the commuting nodes are distributed near the roads. So, they are fewer than in nonconvexity.

* You can run the "[detour/demo_detour.py](detour/demo_detour.py)" directly.

Here is the output of demo_detour.py:

<summary>demo-r10km detour summary.csv</summary>

| areaID | share_of_barrier | detour | distance_max | distance_mean | distance_std | number of commuting nodes | number of road nodes |
|-----:|--------|--------|--------|---------|---------|-------|-------|
|     1| 0.2185 | 0.2665 | 27.5856 | 10.5261 | 5.0514 | 948 | 31498 |
|     2| 0.009491 | 0.3590 | 23.8956 | 7.8960 | 3.9693 | 313 | 6301 |
|     3| 0.0006421 | 0.1925 | 26.5877 | 10.7287 | 5.0226 | 1224 | 68350 |

##  Regression analysis

Description: 

* For other experiments, please download and prepocess the datasets according to our Paper.

## Citation

```BibTeX
@journal{WilliamLiPro/Fragmented_by_Nature,
  author = {Luyao Wang, Albert Saiz, Weipeng Li},
  title = {},
  journal = {},
  year = {},
  doi = {},
}
```
