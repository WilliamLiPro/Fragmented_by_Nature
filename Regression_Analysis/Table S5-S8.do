/* Table 1-GHUB with 10 km radius at S.5.*/
clear
use "DATAGEN/AUE_GHUB_GUB Data/GHUB_allvariables_for regressions.dta", clear
xi:regress detour_10km_b share_of_barrier_10km_b 
est store TS5_1
xi:regress detour_10km_b nonconvexity_10km_b 
est store TS5_2
xi:reghdfe detour_10km_b share_of_barrier_10km_b costal capital dry log_precip log_temp log_elev log_gdppc  log_pop log_bu, absorb ( countr_iso soil_ climate_ biom_ )
est store TS5_3
xi:reghdfe detour_10km_b nonconvexity_10km_b costal capital dry log_precip log_temp log_elev log_gdppc  log_pop log_bu , absorb ( countr_iso soil_ climate_ biom_ )
est store TS5_4
xi:reghdfe detour_10km_b share_of_barrier_10km_b nonconvexity_10km_b costal capital dry log_precip log_temp log_elev log_gdppc  log_pop log_bu , absorb ( countr_iso soil_ climate_ biom_ )
est store TS5_5
outreg2[TS5_1 TS5_2 TS5_3 TS5_4 TS5_5] using TableS5_DETOUR_GHUB.xls, stats(coef,se) addstat(Ajusted R2,`e(r2_a)') replace word label
drop _est_TS5_*
/* Table 1-GUB with 10 km radius at S.5.*/
clear
use "DATAGEN/AUE_GHUB_GUB Data/GUB_allvariables_for regressions.dta", clear
xi:regress detour_10km_b share_of_barrier_10km_b 
est store TS5_1
xi:regress detour_10km_b nonconvexity_10km_b 
est store TS5_2
xi:reghdfe detour_10km_b share_of_barrier_10km_b costal capital dry log_precip log_temp log_elev log_gdppc  log_pop log_bu, absorb ( country_iso soil_ climate_ biom_ )
est store TS5_3
xi:reghdfe detour_10km_b nonconvexity_10km_b costal capital dry log_precip log_temp log_elev log_gdppc  log_pop log_bu , absorb ( country_iso soil_ climate_ biom_ )
est store TS5_4
xi:reghdfe detour_10km_b share_of_barrier_10km_b nonconvexity_10km_b costal capital dry log_precip log_temp log_elev log_gdppc  log_pop log_bu , absorb ( country_iso soil_ climate_ biom_ )
est store TS5_5
outreg2[TS5_1 TS5_2 TS5_3 TS5_4 TS5_5] using TableS5_DETOUR_GUB.xls, stats(coef,se) addstat(Ajusted R2,`e(r2_a)') replace word label
drop _est_TS5_*
/* Table 1-AUE with 10 km radius at S.5.*/
clear
use "DATAGEN/AUE_GHUB_GUB Data/AUE_allvariables_for regressions.dta"
gen log_bu=log(bu_areakm)
label var log_bu "Log of build-up area"
xi:regress detour_10km_b share_of_barrier_10km_b 
est store TS5_1
xi:regress detour_10km_b nonconvexity_10km_b 
est store TS5_2
xi:regress detour_10km_b share_of_barrier_10km_b costal capital dry log_precip log_temp log_elev log_gdppc  log_pop log_bu
est store TS5_3
xi:regress detour_10km_b nonconvexity_10km_b costal capital dry log_precip log_temp log_elev log_gdppc  log_pop log_bu
est store TS5_4
xi:regress detour_10km_b share_of_barrier_10km_b nonconvexity_10km_b costal capital dry log_precip log_temp log_elev log_gdppc  log_pop log_bu
est store TS5_5
outreg2[TS5_1 TS5_2 TS5_3 TS5_4 TS5_5] using TableS5_DETOUR_AUE.xls, stats(coef,se) addstat(Ajusted R2,`e(r2_a)') replace word label
drop _est_TS5_*


/* S.6-1(Complete Table 2 with 10 km radiuses) CODE Omitted: regressions shown in above 'Table 2'*/
/* S.6-2(Complete Table 3) CODE Omitted: regressions shown in above 'Table 3'*/

/* Table 1-UCDB with 5 km radius at S.7.*/
clear
use "DATAGEN/UCDB data/UCDB_allvariables_for regressions.dta"
xi:regress detour_5km_b share_barrier_5km_b 
est store TS7_1
xi:regress detour_5km_b nonconvexity_5km_b 
est store TS7_2
xi:reghdfe detour_5km_b share_barrier_5km_b coastal capital dry log_precip log_temp log_elev log_gdppc  log_pop_new log_buildup_new , absorb (ctr_mn_iso soilYes climate biome )
est store TS7_3
xi:reghdfe detour_5km_b nonconvexity_5km_b coastal capital dry log_precip log_temp log_elev log_gdppc  log_pop_new log_buildup_new , absorb (ctr_mn_iso soilYes climate biome )
est store TS7_4
xi:reghdfe detour_5km_b share_barrier_5km_b nonconvexity_5km_b coastal capital dry log_precip log_temp log_elev log_gdppc  log_pop_new log_buildup_new , absorb (ctr_mn_iso soilYes climate biome )
est store TS7_5
xi:reghdfe detour_5km_b nonconvexity_5km_b coastal capital dry log_precip log_temp log_elev log_gdppc  log_pop_new log_buildup_new UrbanPopRatio averagePR POP014 Literacy GovFCE log_country_gdppc , absorb (soilYes climate biome )
est store TS7_6
outreg2[TS7_1 TS7_2 TS7_3 TS7_4 TS7_5 TS7_6] using TableS7_DETOUR_UCDB.xls, stats(coef,se) addstat(Ajusted R2,`e(r2_a)') replace word label
drop _est_TS7_*

/* Table 1-GHUB with 5 km radius at S.7.*/
clear
use "DATAGEN/AUE_GHUB_GUB Data/GHUB_allvariables_for regressions.dta", clear
xi:regress detour_5km_b share_of_barrier_5km_b 
est store TS7_1
xi:regress detour_5km_b nonconvexity_5km_b 
est store TS7_2
xi:reghdfe detour_5km_b share_of_barrier_5km_b costal capital dry log_precip log_temp log_elev log_gdppc  log_pop log_bu, absorb ( countr_iso soil_ climate_ biom_ )
est store TS7_3
xi:reghdfe detour_5km_b nonconvexity_5km_b costal capital dry log_precip log_temp log_elev log_gdppc  log_pop log_bu , absorb ( countr_iso soil_ climate_ biom_ )
est store TS7_4
xi:reghdfe detour_5km_b share_of_barrier_5km_b nonconvexity_5km_b costal capital dry log_precip log_temp log_elev log_gdppc  log_pop log_bu , absorb ( countr_iso soil_ climate_ biom_ )
est store TS7_5
outreg2[TS7_1 TS7_2 TS7_3 TS7_4 TS7_5] using TableS7_DETOUR_GHUB.xls, stats(coef,se) addstat(Ajusted R2,`e(r2_a)') replace word label
drop _est_TS7_*

/* Table 1-GUB with 5 km radius at S.7.*/
clear
use "DATAGEN/AUE_GHUB_GUB Data/GUB_allvariables_for regressions.dta", clear
xi:regress detour_5km_b share_of_barrier_5km_b 
est store TS7_1
xi:regress detour_5km_b nonconvexity_5km_b 
est store TS7_2
xi:reghdfe detour_5km_b share_of_barrier_5km_b costal capital dry log_precip log_temp log_elev log_gdppc  log_pop log_bu, absorb ( country_iso soil_ climate_ biom_ )
est store TS7_3
xi:reghdfe detour_5km_b nonconvexity_5km_b costal capital dry log_precip log_temp log_elev log_gdppc  log_pop log_bu , absorb ( country_iso soil_ climate_ biom_ )
est store TS7_4
xi:reghdfe detour_5km_b share_of_barrier_5km_b nonconvexity_5km_b costal capital dry log_precip log_temp log_elev log_gdppc  log_pop log_bu , absorb ( country_iso soil_ climate_ biom_ )
est store TS7_5
outreg2[TS7_1 TS7_2 TS7_3 TS7_4 TS7_5] using TableS7_DETOUR_GUB.xls, stats(coef,se) addstat(Ajusted R2,`e(r2_a)') replace word label
drop _est_TS7_*

/* Table 1-AUE with 5 km radius at S.7.*/
clear
use "DATAGEN/AUE_GHUB_GUB Data/AUE_allvariables_for regressions.dta"
gen log_bu=log(bu_areakm)
label var log_bu "Log of build-up area"
xi:regress detour_5km_b share_of_barrier_5km_nb 
est store TS7_1
xi:regress detour_5km_b nonconvexity_5km_nb 
est store TS7_2
xi:regress detour_5km_b share_of_barrier_5km_nb costal capital dry log_precip log_temp log_elev log_gdppc  log_pop log_bu
est store TS7_3
xi:regress detour_5km_b nonconvexity_5km_nb costal capital dry log_precip log_temp log_elev log_gdppc  log_pop log_bu
est store TS7_4
xi:regress detour_5km_b share_of_barrier_5km_nb nonconvexity_5km_nb costal capital dry log_precip log_temp log_elev log_gdppc  log_pop log_bu
est store TS7_5
outreg2[TS7_1 TS7_2 TS7_3 TS7_4 TS7_5] using TableS7_DETOUR_AUE.xls, stats(coef,se) addstat(Ajusted R2,`e(r2_a)') replace word label
drop _est_TS7_*





/* Table 2-UCDB-new with 5 km radius and repeated with all control coefficients at S.8.*/
clear
use "DATAGEN/UCDB data/UCDB_allvariables_for regressions.dta"
xi:reghdfe log_pop_new nonconvexity_5km_b coastal capital dry log_precip log_temp log_elev , absorb (ctr_mn_iso soilYes climate biome )
est store ST8_1_1
xi:reghdfe log_densb_new nonconvexity_5km_b coastal capital dry log_precip log_temp log_elev, absorb ( ctr_mn_iso soilYes climate biome )
est store ST8_1_2
xi:reghdfe sharebu_new nonconvexity_5km_b coastal capital dry log_precip log_temp log_elev , absorb ( ctr_mn_iso soilYes climate biome)
est store ST8_1_3
xi:reghdfe log_gdppc nonconvexity_5km_b  coastal capital dry log_precip log_temp  log_elev log_pop_new , absorb( ctr_mn_iso soilYes climate biome)
est store ST8_1_4
xi:reghdfe log_densb_new nonconvexity_5km_b coastal capital dry log_precip log_temp log_elev log_gdppc log_pop_new , absorb ( ctr_mn_iso soilYes climate biome )
est store ST8_1_5
xi:reghdfe sharebu_new nonconvexity_5km_b coastal capital dry log_precip log_temp log_elev log_gdppc log_pop_new , absorb ( ctr_mn_iso soilYes climate biome)
est store ST8_1_6
xi:reghdfe log_height_new nonconvexity_5km_b coastal capital dry log_precip log_temp log_elev log_gdppc log_pop_new , absorb ( ctr_mn_iso soilYes climate biome)
est store ST8_1_7
outreg2[ST8_1_1 ST8_1_2 ST8_1_3 ST8_1_4 ST8_1_5 ST8_1_6 ST8_1_7] using TableS8_UCDB_new.xls, stats(coef,se) addstat(Ajusted R2,`e(r2_a)') replace word label
drop _est_ST8_1_*

/* Table 2-GHUB */
clear
use "DATAGEN/AUE_GHUB_GUB Data/GHUB_allvariables_for regressions.dta", clear
xi:reghdfe log_pop nonconvexity_5km_b costal capital dry log_precip log_temp log_elev  , absorb(countr_iso soil_ climate_ biom_)
est store ST8_2_1
xi:reghdfe  log_densb nonconvexity_5km_b costal capital dry log_precip log_temp log_elev , absorb(countr_iso soil_ climate_ biom_)
est store ST8_2_2
xi:reghdfe  sharebu nonconvexity_5km_b costal capital dry log_precip log_temp log_elev , absorb(countr_iso soil_ climate_ biom_)
est store ST8_2_3
xi:reghdfe log_gdppc nonconvexity_5km_b costal capital dry log_precip log_temp  log_elev log_pop  , absorb(countr_iso soil_ climate_ biom_)
est store ST8_2_4
xi:reghdfe  log_densb nonconvexity_5km_b costal capital dry log_precip log_temp log_elev log_pop log_gdppc , absorb(countr_iso soil_ climate_ biom_)
est store ST8_2_5
xi:reghdfe sharebu nonconvexity_5km_b costal capital dry log_precip log_temp log_elev log_pop log_gdppc , absorb(countr_iso soil_ climate_ biom_)
est store ST8_2_6
xi:reghdfe log_height_new nonconvexity_5km_b costal capital dry log_precip log_temp log_elev log_pop log_gdppc , absorb(countr_iso soil_ climate_ biom_)
est store ST8_1_7
outreg2 [*] using TableS8_GHUB.xls, stats(coef,se) addstat(Adjusted R-squared, e(r2_a)) replace word label
drop _est_ST8_2_*

/* Table 2-GUB */
clear
use "DATAGEN/AUE_GHUB_GUB Data/GUB_allvariables_for regressions.dta"
xi:reghdfe log_pop nonconvexity_5km_b  costal capital dry log_precip log_temp  log_elev , absorb(country_iso soil_ climate_ biom_)
est store ST8_3_1
xi:reghdfe  log_densb nonconvexity_5km_b costal capital dry log_precip log_temp  log_elev , absorb(country_iso soil_ climate_ biom_)
est store ST8_3_2
xi:reghdfe  sharebu nonconvexity_5km_b costal capital dry log_precip log_temp  log_elev , absorb(country_iso soil_ climate_ biom_)
est store ST8_3_3
xi:reghdfe log_gdppc nonconvexity_5km_b costal capital dry log_precip log_temp  log_elev log_pop , absorb(country_iso soil_ climate_ biom_)
est store ST8_3_4
xi:reghdfe  log_densb nonconvexity_5km_b costal capital dry log_precip log_temp  log_elev log_pop log_gdppc  , absorb(country_iso soil_ climate_ biom_)
est store ST8_3_5
xi:reghdfe sharebu nonconvexity_5km_b costal capital dry log_precip log_temp  log_elev log_pop log_gdppc , absorb(country_iso soil_ climate_ biom_)
est store ST8_3_6
xi:reghdfe log_height_new nonconvexity_5km_b costal capital dry log_precip log_temp log_elev log_pop log_gdppc , absorb(country_iso soil_ climate_ biom_)
est store ST8_1_7
outreg2 [*] using TableS8_GUB.xls, stats(coef,se) addstat(Adjusted R-squared, e(r2_a)) replace word label
drop _est_ST8_3_*

/* Table 2-AUE */
clear
use "DATAGEN/AUE_GHUB_GUB Data/AUE_allvariables_for regressions.dta"
xi:regress log_pop nonconvexity_5km_nb costal capital dry log_precip log_temp  log_elev
est store ST8_4_1
xi:regress log_densb nonconvexity_5km_nb costal capital dry log_precip log_temp  log_elev
est store ST8_4_2
xi:regress frgmentationsaturationbuiltupare nonconvexity_5km_nb costal capital dry log_precip log_temp  log_elev
est store ST8_4_3
xi:regress  log_gdppc nonconvexity_5km_nb costal capital dry log_precip log_temp  log_elev log_pop
est store ST8_4_4
xi:regress log_densb nonconvexity_5km_nb costal capital dry log_precip log_temp  log_elev log_pop log_gdppc
est store ST8_4_5
xi:regress frgmentationsaturationbuiltupare nonconvexity_5km_nb costal capital dry log_precip log_temp  log_elev log_pop log_gdppc
est store ST8_4_6
xi:regress log_height_new nonconvexity_5km_nb costal capital dry log_precip log_temp  log_elev log_pop log_gdppc
est store ST8_4_7
outreg2 [*] using TableS8_AUE.xls, stats(coef,se) addstat(Adjusted R-squared, e(r2_a)) replace word label
drop _est_ST8_4_*
