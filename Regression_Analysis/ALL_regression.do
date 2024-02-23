
/* The earliest compatible version we used is version 14 */
version 14

clear
eststo clear
set more off

/* TO USER: PLEASE INSTALL STATA PACKAGES PER BELOW */
*capture ssc install reghdfe
*capture ssc install ivreghdfe
*capture ssc install outreg2
*capture ssc install asdoc
*capture ssc install estout



/* regressions of UCDB dataset*/
/* IMPORT UCDB 2019 Dataset with GHSL 2023 UPDATES*/
clear
use "DATAGEN/UCDB data/UCDB_allvariables_for regressions.dta"

/*Table 1-DETOUR*/
xi:regress detour_10km_b share_barrier_10km_b 
est store T1_1
xi:regress detour_10km_b nonconvexity_10km_b 
est store T1_2
xi:reghdfe detour_10km_b share_barrier_10km_b coastal capital dry log_precip log_temp log_elev log_gdppc  log_pop_new log_buildup_new , absorb (ctr_mn_iso soilYes climate biome )
est store T1_3
xi:reghdfe detour_10km_b nonconvexity_10km_b coastal capital dry log_precip log_temp log_elev log_gdppc  log_pop_new log_buildup_new , absorb (ctr_mn_iso soilYes climate biome )
est store T1_4
xi:reghdfe detour_10km_b share_barrier_10km_b nonconvexity_10km_b coastal capital dry log_precip log_temp log_elev log_gdppc  log_pop_new log_buildup_new , absorb (ctr_mn_iso soilYes climate biome )
est store T1_5
xi:reghdfe detour_10km_b nonconvexity_10km_b coastal capital dry log_precip log_temp log_elev log_gdppc  log_pop_new log_buildup_new UrbanPopRatio averagePR POP014 Literacy GovFCE log_country_gdppc , absorb (soilYes climate biome )
est store T1_6
outreg2[T1_1 T1_2 T1_3 T1_4 T1_5 T1_6] using Table1_DETOUR_UCDB.xls, stats(coef,se) addstat(Ajusted R2,`e(r2_a)') replace word label
drop _est_T1_*

/* Table 2-UCDB-old*/
xi:reghdfe log_pop nonconvexity_10km_b coastal capital dry log_precip log_temp log_elev, absorb (ctr_mn_iso soilYes climate biome )
est store T2_1_1
xi:reghdfe lodensb nonconvexity_10km_b coastal capital dry log_precip log_temp log_elev, absorb (ctr_mn_iso soilYes climate biome )
est store T2_1_2
xi:reghdfe sharebu nonconvexity_10km_b coastal capital dry log_precip log_temp log_elev, absorb (ctr_mn_iso soilYes climate biome)
est store T2_1_3
xi:reghdfe logdpcap15 nonconvexity_10km_b coastal capital dry log_precip log_temp log_elev log_pop, absorb (ctr_mn_iso soilYes climate biome )
est store T2_1_4
xi:reghdfe lodensb nonconvexity_10km_b coastal capital dry log_precip log_temp log_elev logdpcap15 log_pop, absorb (ctr_mn_iso soilYes climate biome)
est store T2_1_5
xi:reghdfe sharebu nonconvexity_10km_b coastal capital dry log_precip log_temp log_elev logdpcap15 log_pop, absorb (ctr_mn_iso soilYes climate biome)
est store T2_1_6
xi:reghdfe log_height nonconvexity_10km_b coastal capital dry log_precip log_temp log_elev logdpcap15 log_pop, absorb (ctr_mn_iso soilYes climate biome)
est store T2_1_7
outreg2 [*] using Table2_UCDB_old.xls, stats(coef,se) addstat(Adjusted R-squared, e(r2_a))
drop _est_T2_1_*

/* Table 2-UCDB-new and repeated with all control coefficients at S.7.*/
xi:reghdfe log_pop_new nonconvexity_10km_b coastal capital dry log_precip log_temp log_elev , absorb (ctr_mn_iso soilYes climate biome )
est store T2_1_1
xi:reghdfe log_densb_new nonconvexity_10km_b coastal capital dry log_precip log_temp log_elev , absorb (ctr_mn_iso soilYes climate biome )
est store T2_1_2
xi:reghdfe sharebu_new nonconvexity_10km_b coastal capital dry log_precip log_temp log_elev , absorb (ctr_mn_iso soilYes climate biome)
est store T2_1_3
xi:reghdfe log_gdppc nonconvexity_10km_b  coastal capital dry log_precip log_temp  log_elev log_pop_new, absorb(ctr_mn_iso soilYes climate biome)
est store T2_1_4
xi:reghdfe log_densb_new nonconvexity_10km_b coastal capital dry log_precip log_temp log_elev log_gdppc log_pop_new , absorb (ctr_mn_iso soilYes climate biome )
est store T2_1_5
xi:reghdfe sharebu_new nonconvexity_10km_b coastal capital dry log_precip log_temp log_elev log_gdppc log_pop_new , absorb (ctr_mn_iso soilYes climate biome)
est store T2_1_6
xi:reghdfe log_height_new nonconvexity_10km_b coastal capital dry log_precip log_temp log_elev log_gdppc log_pop_new , absorb (ctr_mn_iso soilYes climate biome)
est store T2_1_7
outreg2[T2_1_1 T2_1_2 T2_1_3 T2_1_4 T2_1_5 T2_1_6 T2_1_7] using Table2_UCDB_new.xls, stats(coef,se) addstat(Ajusted R2,`e(r2_a)') replace word label
drop _est_T2_1_*

/* Table 2-GHUB */
clear
use "DATAGEN/AUE_GHUB_GUB Data/GHUB_allvariables_for regressions.dta", clear
xi:reghdfe log_pop nonconvexity_10km_b costal capital dry log_precip log_temp log_elev , absorb(countr_iso soil_ climate_ biom_)
est store T2_2_1
xi:reghdfe  log_densb nonconvexity_10km_b costal capital dry log_precip log_temp log_elev , absorb(countr_iso soil_ climate_ biom_)
est store T2_2_2
xi:reghdfe  sharebu nonconvexity_10km_b costal capital dry log_precip log_temp log_elev , absorb(countr_iso soil_ climate_ biom_)
est store T2_2_3
xi:reghdfe log_gdppc nonconvexity_10km_b costal capital dry log_precip log_temp  log_elev log_pop  , absorb(countr_iso soil_ climate_ biom_)
est store T2_2_4
xi:reghdfe  log_densb nonconvexity_10km_b costal capital dry log_precip log_temp log_elev log_pop log_gdppc , absorb(countr_iso soil_ climate_ biom_)
est store T2_2_5
xi:reghdfe sharebu nonconvexity_10km_b costal capital dry log_precip log_temp log_elev log_pop log_gdppc , absorb(countr_iso soil_ climate_ biom_)
est store T2_2_6
xi:reghdfe log_height_new nonconvexity_10km_b costal capital dry log_precip log_temp log_elev log_pop log_gdppc , absorb(countr_iso soil_ climate_ biom_)
est store T2_1_7
outreg2 [*] using Table2_GHUB.xls, stats(coef,se) addstat(Adjusted R-squared, e(r2_a)) replace word label
drop _est_T2_2_*

/* Table 2-GUB */
clear
use "DATAGEN/AUE_GHUB_GUB Data/GUB_allvariables_for regressions.dta"
xi:reghdfe log_pop nonconvexity_10km_b costal capital dry log_precip log_temp  log_elev , absorb(country_iso soil_ climate_ biom_)
est store T2_3_1
xi:reghdfe  log_densb nonconvexity_10km_b costal capital dry log_precip log_temp  log_elev , absorb(country_iso soil_ climate_ biom_)
est store T2_3_2
xi:reghdfe  sharebu nonconvexity_10km_b costal capital dry log_precip log_temp  log_elev , absorb(country_iso soil_ climate_ biom_)
est store T2_3_3
xi:reghdfe log_gdppc nonconvexity_10km_b costal capital dry log_precip log_temp  log_elev log_pop , absorb(country_iso soil_ climate_ biom_)
est store T2_3_4
xi:reghdfe  log_densb nonconvexity_10km_b costal capital dry log_precip log_temp  log_elev log_pop log_gdppc , absorb(country_iso soil_ climate_ biom_)
est store T2_3_5
xi:reghdfe sharebu nonconvexity_10km_b costal capital dry log_precip log_temp  log_elev log_pop log_gdppc , absorb(country_iso soil_ climate_ biom_)
est store T2_3_6
xi:reghdfe log_height_new nonconvexity_10km_b costal capital dry log_precip log_temp log_elev log_pop log_gdppc , absorb(country_iso soil_ climate_ biom_)
est store T2_1_7
outreg2 [*] using Table2_GUB.xls, stats(coef,se) addstat(Adjusted R-squared, e(r2_a)) replace word label
drop _est_T2_3_*

/* Table 2-AUE */
clear
use "DATAGEN/AUE_GHUB_GUB Data/AUE_allvariables_for regressions.dta"
xi:regress log_pop nonconvexity_10km_b costal capital dry log_precip log_temp  log_elev
est store T2_4_1
xi:regress log_densb nonconvexity_10km_b costal capital dry log_precip log_temp  log_elev
est store T2_4_2
xi:regress frgmentationsaturationbuiltupare nonconvexity_10km_b costal capital dry log_precip log_temp  log_elev
est store T2_4_3
xi:regress  log_gdppc nonconvexity_10km_b costal capital dry log_precip log_temp  log_elev log_pop
est store T2_4_4
xi:regress log_densb nonconvexity_10km_b costal capital dry log_precip log_temp  log_elev log_pop log_gdppc
est store T2_4_5
xi:regress frgmentationsaturationbuiltupare nonconvexity_10km_b costal capital dry log_precip log_temp  log_elev log_pop log_gdppc
est store T2_4_6
xi:regress log_height_new nonconvexity_10km_b costal capital dry log_precip log_temp  log_elev log_pop log_gdppc
est store T2_4_7

outreg2 [*] using Table2_AUE.xls, stats(coef,se) addstat(Adjusted R-squared, e(r2_a)) replace word label
drop _est_T2_4_*


/* Table 3- Environmental Variables*/
/*Use 2019 UCDB as derived variables were calculated on per capita basis using 2019 estimates*/
clear
use "DATAGEN/UCDB data/UCDB_allvariables_for regressions.dta"

xi:reghdfe log_light  nonconvexity_10km_b coastal capital dry log_precip log_temp log_elev log_pop, absorb (ctr_mn_iso soilYes climate biome )  
est store T3_1
xi:reghdfe e_gr_av14  nonconvexity_10km_b coastal capital dry log_precip log_temp log_elev log_pop  logdpcap15 , absorb (ctr_mn_iso soilYes climate biome )
est store T3_2
xi:reghdfe loco2r  nonconvexity_10km_b coastal capital dry log_precip log_temp log_elev log_pop  logdpcap15 , absorb (ctr_mn_iso soilYes climate biome )
est store T3_3
xi:reghdfe loco2t  nonconvexity_10km_b coastal capital dry log_precip log_temp log_elev log_pop logdpcap15 , absorb (ctr_mn_iso soilYes climate biome )
est store T3_4
outreg2[T3_1 T3_2 T3_3 T3_4] using Table3_UCDB_new.xls, stats(coef,se) addstat(Ajusted R2,`e(r2_a)') replace word label
drop _est_T3_*

/* APENDIXES AND ROBUSTNESS*/
/* S.5. (Table 1 for other three datasets with 10 km radiuses) Detour regressions at 10 kilometres from other datasets*/
/* S.6-1. (Complete Table 2 with 10 km radiuses) regressions shown in above 'Table 2'*/
/* S.6-2. (Complete Table 3) regressions shown in above 'Table 3'*/
/* S.7. (Table 1 with 5 km radius) Detour regressions at 5 kilometres from center*/
/* S.8. (TABLE 2 WITH 5 KM RADIUSES) regressions of nonconvexity from the four datasets with 5 km radiuses*/
do "DATAGEN/Table S5-S8.do"



clear
use "DATAGEN/UCDB data/UCDB_allvariables_for regressions.dta"
/* S.9-1 GHSL23-UCDB Density: trimmed by "outliers" in ratios */
gen pop_dens_new=pop_new/buildup_new
summ pop_dens_new, d
summ pop_dens_new if pop_new>1000000, d
summ pop_dens_new if pop_new>1000000 & pop_dens_new>55000 , d
xtile pct_pop_dens_new =pop_dens_new, nq(10)

xi:reghdfe log_pop_new nonconvexity_10km_b coastal capital dry log_precip log_temp  log_elev if pct_pop_dens_new>1 & pct_pop_dens_new<10 , absorb (ctr_mn_iso soilYes climate biome )
est store TS9_1_1
xi:reghdfe log_densb_new nonconvexity_10km_b coastal capital dry log_precip log_temp  log_elev if pct_pop_dens_new>1 & pct_pop_dens_new<10 , absorb (ctr_mn_iso soilYes climate biome )
est store TS9_1_2
xi:reghdfe sharebu_new nonconvexity_10km_b coastal capital dry log_precip log_temp  log_elev if pct_pop_dens_new>1 & pct_pop_dens_new<10 , absorb (ctr_mn_iso soilYes climate biome)
est store TS9_1_3
xi:reghdfe log_gdppc nonconvexity_10km_b coastal capital dry log_precip log_temp log_elev log_pop_new if pct_pop_dens_new>1 & pct_pop_dens_new<10  , absorb (ctr_mn_iso soilYes climate biome )
est store TS9_1_4
xi:reghdfe log_densb_new nonconvexity_10km_b coastal capital dry log_precip log_temp  log_elev log_gdppc log_pop_new if pct_pop_dens_new>1 & pct_pop_dens_new<10  , absorb (ctr_mn_iso soilYes climate biome )
est store TS9_1_5
xi:reghdfe sharebu_new nonconvexity_10km_b coastal capital dry log_precip log_temp  log_elev log_gdppc log_pop_new if pct_pop_dens_new>1 & pct_pop_dens_new<10  , absorb (ctr_mn_iso soilYes climate biome)
est store TS9_1_6
outreg2[*] using TableS9_trimmed_UCDB.xls, stats(coef,se) addstat(Ajusted R2,`e(r2_a)') replace word label
drop _est_TS9_1_*

/* S.10 GHSL23-UCDB Ignore International Borders */
xi:reghdfe log_pop_new nonconvexity_10km_nb coastal capital dry log_precip log_temp  log_elev , absorb (ctr_mn_iso soilYes climate biome )
est store TS10_1_1
xi:reghdfe log_densb_new nonconvexity_10km_nb coastal capital dry log_precip log_temp  log_elev , absorb (ctr_mn_iso soilYes climate biome )
est store TS10_1_2
xi:reghdfe sharebu_new nonconvexity_10km_nb coastal capital dry log_precip log_temp  log_elev , absorb (ctr_mn_iso soilYes climate biome)
est store TS10_1_3
xi:reghdfe log_gdppc nonconvexity_10km_nb coastal capital dry log_precip log_temp log_elev log_pop_new   , absorb (ctr_mn_iso soilYes climate biome )
est store TS10_1_4
xi:reghdfe log_densb_new nonconvexity_10km_nb coastal capital dry log_precip log_temp  log_elev log_gdppc log_pop_new , absorb (ctr_mn_iso soilYes climate biome )
est store TS10_1_5
xi:reghdfe sharebu_new nonconvexity_10km_nb coastal capital dry log_precip log_temp  log_elev log_gdppc log_pop_new , absorb (ctr_mn_iso soilYes climate biome)
est store TS10_1_6
outreg2[*] using TableS10_ignore_border_UCDB.xls, stats(coef,se) addstat(Ajusted R2,`e(r2_a)') replace word label
drop _est_TS10_1_*

/* TS11: REGRESIONS WITH RANDOM RADIUSES*/
forvalues i=0(1)4 {

xi:reghdfe log_pop_new nonconvexityratio_`i' coastal capital dry log_precip log_temp log_elev , absorb (ctr_mn_iso soilYes climate biome )
est store T2_1_1
xi:reghdfe log_densb_new nonconvexityratio_`i' coastal capital dry log_precip log_temp log_elev , absorb (ctr_mn_iso soilYes climate biome )
est store T2_1_2
xi:reghdfe sharebu_new nonconvexityratio_`i' coastal capital dry log_precip log_temp log_elev , absorb (ctr_mn_iso soilYes climate biome)
est store T2_1_3
xi:reghdfe log_gdppc nonconvexityratio_`i'  coastal capital dry log_precip log_temp  log_elev log_pop_new, absorb(ctr_mn_iso soilYes climate biome)
est store T2_1_4
xi:reghdfe log_densb_new nonconvexityratio_`i' coastal capital dry log_precip log_temp log_elev log_gdppc log_pop_new , absorb (ctr_mn_iso soilYes climate biome )
est store T2_1_5
xi:reghdfe sharebu_new nonconvexityratio_`i' coastal capital dry log_precip log_temp log_elev log_gdppc log_pop_new , absorb (ctr_mn_iso soilYes climate biome)
est store T2_1_6
xi:reghdfe log_height_new nonconvexityratio_`i' coastal capital dry log_precip log_temp log_elev log_gdppc log_pop_new , absorb (ctr_mn_iso soilYes climate biome)
est store T2_1_7
outreg2[T2_1_1 T2_1_2 T2_1_3 T2_1_4 T2_1_5 T2_1_6 T2_1_7] using TableS11_`i'.xls, stats(coef,se) addstat(Ajusted R2,`e(r2_a)') replace word label
drop _est_T2_1_*
}


/* TS12: IV REGRESIONS WITH RANDOM RADIUSES AS IVs*/

xi:ivreghdfe log_pop_new (nonconvexity_10km_b=nonconvexityratio_0 nonconvexityratio_1 nonconvexityratio_2 nonconvexityratio_3 nonconvexityratio_4) coastal capital dry log_precip log_temp log_elev , absorb (ctr_mn_iso soilYes climate biome ) first
est store TS12_1
xi:ivreghdfe log_densb_new (nonconvexity_10km_b=nonconvexityratio_0 nonconvexityratio_1 nonconvexityratio_2 nonconvexityratio_3 nonconvexityratio_4) coastal capital dry log_precip log_temp log_elev , absorb (ctr_mn_iso soilYes climate biome )
est store TS12_2
xi:ivreghdfe sharebu_new (nonconvexity_10km_b=nonconvexityratio_0 nonconvexityratio_1 nonconvexityratio_2 nonconvexityratio_3 nonconvexityratio_4) coastal capital dry log_precip log_temp log_elev , absorb (ctr_mn_iso soilYes climate biome)
est store TS12_3
xi:ivreghdfe log_gdppc (nonconvexity_10km_b=nonconvexityratio_0 nonconvexityratio_1 nonconvexityratio_2 nonconvexityratio_3 nonconvexityratio_4)  coastal capital dry log_precip log_temp  log_elev log_pop_new, absorb(ctr_mn_iso soilYes climate biome)
est store TS12_4
xi:ivreghdfe log_densb_new (nonconvexity_10km_b=nonconvexityratio_0 nonconvexityratio_1 nonconvexityratio_2 nonconvexityratio_3 nonconvexityratio_4) coastal capital dry log_precip log_temp log_elev log_gdppc log_pop_new , absorb (ctr_mn_iso soilYes climate biome )
est store TS12_5
xi:ivreghdfe sharebu_new (nonconvexity_10km_b=nonconvexityratio_0 nonconvexityratio_1 nonconvexityratio_2 nonconvexityratio_3 nonconvexityratio_4) coastal capital dry log_precip log_temp log_elev log_gdppc log_pop_new , absorb (ctr_mn_iso soilYes climate biome)
est store TS12_6
xi:ivreghdfe log_height_new (nonconvexity_10km_b=nonconvexityratio_0 nonconvexityratio_1 nonconvexityratio_2 nonconvexityratio_3 nonconvexityratio_4) coastal capital dry log_precip log_temp log_elev log_gdppc log_pop_new , absorb (ctr_mn_iso soilYes climate biome)
est store TS12_7
outreg2[TS12_1 TS12_2 TS12_3 TS12_4 TS12_5 TS12_6 TS12_7] using TableS12_UCDB_IV.xls, stats(coef,se) addstat(Ajusted R2,`e(r2_a)') replace word label
drop _est_TS12_*






