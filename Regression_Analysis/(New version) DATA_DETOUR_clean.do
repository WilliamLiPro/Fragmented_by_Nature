clear
eststo clear
set more off

/* The earliest compatible version we used is version 14 */
version 14

/* Albert's folder*/
cd "/Users/saiz/Dropbox (MIT)/RESEARCH/BOSTONGEOGRAPHY/WORLD/regressions_DETOUR/DATAGEN/UCDB data"
/* Luyao's folder*/
*cd "UCDB data"


/* Import data from original UCDB datasets */
import delimited "GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.csv", encoding(ISO-8859-1)

keep ïid_hdc_g0 area ctr_mn_nm ctr_mn_iso grgn_l1 grgn_l2 uc_nm_mn el_av_als e_rb_nm_lst e_wr_p_14 e_wr_t_14 b15 p15 ntl_av gdp15_sm e_gr_av14 e_epm2_r15 e_epm2_i15 e_epm2_t15 tt2cc e_sl_lst e_kg_nm_lst e_bm_nm_lst e_ec2e_r15 e_ec2e_i15 e_ec2e_t15 

rename area area15
rename ïid_hdc_g0  ucid

gen dry=1 if e_rb_nm_lst=="NAN"
replace dry=0 if dry==.

split e_sl_lst, parse(;)
rename e_sl_lst1 soilYes
drop e_sl_lst*

split e_kg_nm_lst, parse(,)
split e_kg_nm_lst1, parse(;)
rename e_kg_nm_lst11 climate
drop e_kg_nm_lst*

split e_bm_nm_lst, parse(,)
split e_bm_nm_lst1, parse(;)
rename e_bm_nm_lst11 biome
drop e_bm_nm_lst*

replace el_av_als="" if el_av_als=="NAN"
destring el_av_als , force replace
replace el_av_als=1 if el_av_als<1

replace e_gr_av14 ="" if e_gr_av14 =="NAN"
replace e_ec2e_r15 ="" if e_ec2e_r15 =="NAN"
replace e_ec2e_i15 ="" if e_ec2e_i15 =="NAN"
replace e_ec2e_t15 ="" if e_ec2e_t15 =="NAN"
replace e_epm2_r15 ="" if e_epm2_r15 =="NAN"
replace e_epm2_t15 ="" if e_epm2_t15 =="NAN"
destring e_gr_av14 e_ec2e_r15 e_ec2e_i15 e_ec2e_t15 e_epm2_r15 e_epm2_t15 , force replace
destring e_wr* e_ec2* e_gr*, force replace


gen log_precip=log(e_wr_p_14+1)
gen log_temp=log(e_wr_t_14+10)
gen log_elev=log(el_av_als)
gen log_pop=log(p15)
gen log_light=log(ntl_av)
gen loco2r=log(e_ec2e_r15)
gen loco2i=log(e_ec2e_i15)
gen loco2t=log(e_ec2e_t15) 
gen lopm25r=log(e_epm2_r15) 
gen lopm25i=log(e_epm2_i15) 
gen lopm25t=log(e_epm2_t15)

gen densityb=p15/b15
gen lodensb=log(densityb)
gen sharebu=b15/area15
gen logdpcap15=log(gdp15_sm/p15)

gen capital=1 if tt2cc==0
replace capital=0 if capital==.
destring el_av_als, force replace

label var ctr_mn_nm "Country Name"
label var ctr_mn_iso "Country Code"
label var uc_nm_mn "City Name"
label var climate "Climate Class"
label var biome "Biome Class" 
label var soilYes "Soil Group"
label var el_av_als "Average Elevation"
label var log_elev "Log Average Elevation"
label var dry "City not in Major River Watershed"
label var e_wr_p_14 "Average Precipitation for Epoch 2014"
label var e_wr_t_14 "Average Temperature for Epoch 1990"
label var b15 "(Old version) Total built-up area in 2015"
label var p15 "(Old aversion) Total Resident Population in 2015"
label var log_pop "Log of Total Resident Population in 2015"
label var ucid "Urban Area's Unique Identifier in Urban Centre Database(UCD)"
label var ntl_av "Average Night Time Light Emission in 2015"
label var log_light "Log Average Night Time Light Emission in 2015"
label var gdp15_sm "Sum of GDP PPP Values for Year 2015"
label var tt2cc "Travel Time to Country Capital(min)"
label var e_ec2e_r15 "Residential CO2 Emissions non-organic 2015"
label var e_ec2e_i15 "Industrial CO2 Emissions non-organic 2015"
label var e_ec2e_t15 "Transportation CO2 Emissions non-organic 2015"
label var e_epm2_r15 "Residential PM2.5 Emissions 2015"
label var e_epm2_i15 "Industrial PM2.5 Emissions 2015"
label var e_epm2_t15 "Transportation PM2.5 Emissions 2015"
label var loco2r "Log Residential CO2 Emissions non-organic 2015"
label var loco2i "Log Industrial CO2 Emissions non-organic 2015"
label var loco2t "Log Transportation CO2 Emissions non-organic 2015"
label var lopm25r "Log Residential PM2.5 Emissions 2015"
label var lopm25i "Log Industrial PM2.5 Emissions 2015"
label var lopm25t "Log Transportation PM2.5 Emissions 2015"
label var area "Area of Urban Centre in 2015"
label var log_precip "Log of Average Precipitations in 2015"
label var log_temp "Log of Average Temperature in 2015"
label var capital "Dummy of Capital City" 
label var densityb "Density of Resident Population in Total Built-up area in 2015 (UCDB 2019 Data Vintage)"
label var lodensb "Log of Density of Resident Population in Total Built-up area in 2015 (UCDB 2019 Data Vintage)"
label var sharebu "Share of Total Built-up area of the Area of Urban Centre in 2015 (UCDB 2019 Data Vintage)"
label var logdpcap15 "(Old version) Log GDP per Capita in 2015"
label variable e_rb_nm_lst "Major river bashin"
label variable e_gr_av14 "Average greeness"
sort ucid
save "UCDB_R2019A.dta", replace
clear

/* Geographic indicators of UCDB (share of barriers, nonconvexity and detour) */ 
import delimited "UCDB_GeographicIndexes.csv", encoding(ISO-8859-1)
gen coastal=1 if water_share_10km>=0.05
replace coastal=0 if water_share_10km<0.05
drop water_share_5km water_share_10km
label variable nonconvexity_10km_nb "Average Local Non-convexity within 10km of City Center(no boundary)"
label variable nonconvexity_5km_nb "Average Local Non-convexity within 5km of City Center(no boundary)"
label variable nonconvexity_10km_b "Average Local Non-convexity within 10km of City Center(with boundary)"
label variable nonconvexity_5km_b "Average Local Non-convexity within 5km of City Center(with boundary)"
label variable detour_5km_nb "Average detour within 5km of City Center(no boundary)"
label variable nonconvexity_10km_nb "Average Local Non-convexity within 10km of City Center (no boundary)"
label variable nonconvexity_10km_b "Average Local Non-convexity within 10km of City Center (with boundary)"
label variable nonconvexity_5km_nb "Average Local Non-convexity within 5km of City Center (no boundary)"
label variable detour_10km_nb "Average detour within 10km of city center (no boundary)"
label variable detour_5km_nb "Average detour within 5km of city center (no boundary)"
label variable detour_5km_b "Average detour within 5km of City Center(with boundary)"
label variable detour_10km_b "Average detour within 10km of City Center(with boundary)"
label variable nonconvexity_elastic_nb "Average Local Non-convexity with elastic ratio (no boundary)"
label variable share_barrier_elastic_nb "Share of barriers with elastic ratio (no boundary)"
label variable share_barrier_elastic_b "Share of barriers with elastic ratio (with boundary)"
label variable nonconvexity_elastic_b "Average Local Non-convexity with elastic ratio (with boundary)"
label variable share_barrier_5km_nb "Share of barriers within 5km of City Center (no boundary)"
label variable share_barrier_10km_b "Share of barriers within 10km of City Center (with boundary)"
label variable share_barrier_10km_nb "Share of barriers within 10km of City Center (no boundary)"
label variable share_barrier_5km_b "Share of barriers within 5km of City Center(with boundary)"
label variable share_barrier_5km_nb "Share of barriers within 5km of City Center (no boundary)"
label variable nonconvexityratio_0 "(Random R0) nonconvexity "
label variable share_of_barrierratio_1 "(Random R1) share of barriers"
label variable share_of_barrierratio_2 "(Random R2) share of barriers"
label variable share_of_barrierratio_3 "(Random R3) share of barriers"
label variable share_of_barrierratio_4 "(Random R4) share of barriers"
label variable nonconvexityratio_1 "(Random R1) nonconvexity "
label variable nonconvexityratio_2 "(Random R2) nonconvexity "
label variable nonconvexityratio_3 "(Random R3) nonconvexity "
label variable nonconvexityratio_4 "(Random R4) nonconvexity "
label variable coastal "Dummy for Ocean or Lake Intersection"
label variable detour_40km_b "Average detour within 40km of City Center(with boundary)"
label variable share_barrier_40km_b "Share of barriers within 40km of City Center (with boundary)"
label variable nonconvexity_40km_b "Average Local Non-convexity within 40km of City Center (with boundary)"
sort ucid
save "UCDB_GeographicIndexes.dta", replace
clear

/* New variables from the 2023 vintage: population, build-up area, and build-up volumns of UCDB, which is derived from GHSL 2023 data package*/ 
import delimited "UCDB_2023vintage.csv", encoding(ISO-8859-1)
label variable buildup100 "(New version) Total built-up area in 2015 "
label variable pop100 "(New version) Total Resident Population in 2015"
label variable bvall "Build-up volume (m3)"
label variable bvnr "Build-up volume of non-residential area (m3)"
label variable gdp_new2 "GDP 2015 (UCDB 2023 vintage_weighted version)"
gen buildup_new=buildup100
gen pop_new=pop100
keep buildup_new pop_new bvall bvnr ucid worldpop gdp_new2
sort ucid
save "UCDB_2023vintage.dta", replace
clear


/*----------------------------------Country_level indicators---------------------------------------------*/
/*Country_level:World Development Indicators (World Bank)*/
import delimited "WDIData.csv", encoding(ISO-8859-1)
egen indicatorid = group(indicatorname)
order indicatorid,before(indicatorname)
rename countrycode ctr_mn_iso 
keep if indicatorid==427 | indicatorid==478 | indicatorid==508 |indicatorid==717 |indicatorid==1026 |indicatorid==1444 
egen meanvalue=rowmean( year2000- year2015)
keep ctr_mn_iso indicatorid meanvalue
reshape wide meanvalue ,i(ctr_mn_iso) j(indicatorid)
label variable meanvalue427 "COUNTRY: Fertility rate, total (births per woman)"
label variable meanvalue478 "COUNTRY: GDP per capita (constant 2015 US$)"
label variable meanvalue508 "COUNTRY: General government final consumption expenditure (% of GDP)"
label variable meanvalue717 "COUNTRY: Literacy rate, adult total (% of people ages 15 and above)"
label variable meanvalue1026 "COUNTRY: Population ages 0-14 (% of total population)"
label variable meanvalue1444 "COUNTRY: Urban population (% of total population)"
rename meanvalue427 FertilityRate
rename meanvalue478 GDPPC
rename meanvalue508 GovFCE
rename meanvalue717 Literacy
rename meanvalue1026 POP014
rename meanvalue1444 UrbanPopRatio
sort ctr_mn_iso
save "Country_level_WDIData.dta", replace
clear

/*Country_level:Data from freedomhouse*/
import delimited "0519_Country_Ratings_final.csv", encoding(ISO-8859-1)
egen averagePR=rmean(pr1973-pr2015)
rename country_iso ctr_mn_iso
keep ctr_mn_iso averagePR
label var averagePR "COUNTRY:Average freehouse index from 1973-2015"
sort ctr_mn_iso
save "Country_level_freehouse.dta", replace
clear
/*--------------------------------------------------------------------------------------------------------*/




/*Combined all variables*/
use UCDB_R2019A.dta
merge ucid using "UCDB_GeographicIndexes.dta"
drop _merge
sort ucid
merge ucid using "UCDB_2023vintage.dta"
drop _merge
sort ctr_mn_iso
merge ctr_mn_iso using "Country_level_WDIData.dta"
drop if _merge==2
drop _merge
sort ctr_mn_iso
merge ctr_mn_iso using "Country_level_freehouse.dta"
drop if _merge==2
drop _merge
drop grgn_l1 grgn_l2

egen sum_countryPOP=sum(pop_new), by ( ctr_mn_iso )
egen sum_countryUrbanArea=sum(area15), by ( ctr_mn_iso )
gen endogenous_radius=2*sqrt((pop_new*( sum_countryUrbanArea- area15)/(sum_countryPOP- pop_new))/_pi)
gen log_radius=log(endogenous_radius)
drop sum_countryPOP sum_countryUrbanArea

gen log_random_0=log(random_ratio_0)
gen log_random_1=log(random_ratio_1)
gen log_random_2=log(random_ratio_2)
gen log_random_3=log(random_ratio_3)
gen log_random_4=log(random_ratio_4)
gen log_pop_new=log(pop_new)
gen log_buildup_new=log(buildup_new)
gen log_gdppc_new=log(gdp_new2/pop_new)
gen log_densb_new=log(pop_new/buildup_new)
gen sharebu_new=buildup_new/area15
gen log_country_gdppc=log(GDPPC)
gen height=bvall/(buildup_new*1000000)
gen log_height_new=log(height)

label var log_random_0 "Log of random ratio 0"
label var log_random_1 "Log of random ratio 1"
label var log_random_2 "Log of random ratio 2"
label var log_random_3 "Log of random ratio 3"
label var log_random_4 "Log of random ratio 4"
label var endogenous_radius "Endogenous Circular Radius"
label var log_radius "Log of Endogenous Circular Radius"
label var log_pop_new " Log of Total Resident Population in 2015 (UCDB 2023 vintage)"
label var sharebu_new "Share of build-up area within urban unit (UCDB 2023 vintage)"
label var log_buildup_new "Log of build-up area within urban unit (UCDB 2023 vintage)"
label var log_densb_new "Log of resident population density in build-up area (UCDB 2023 vintage)"
label var log_gdppc_new "Log of GDPPC (UCDB 2023 vintage_weighted version)"
label var log_country_gdppc "COUNTRY: GDPPC"
label var height "Average build-up area height (m)"
label var log_height_new "Log of average build-up area height (m)"
save "UCDB_allvariables_for regressions", replace
clear



