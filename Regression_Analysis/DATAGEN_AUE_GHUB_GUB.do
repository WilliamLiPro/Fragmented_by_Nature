/* Albert's folder*/
*cd "/Users/saiz/Dropbox (MIT)/RESEARCH/BOSTONGEOGRAPHY/WORLD/regressions_DETOUR/DATAGEN/"
/* Luyao's folder*/
cd "AUE_GHUB_GUB Data"



/*-----------------------------------------------------------------Creation of AUE datasets----------------------------------------------------------*/
/*Import the external variables (Gridded data)*/
import delimited "AUE_ExternalSources.csv", clear
replace elev =1 if elev <1
gen log_precip=log(prep2015+1)
gen log_temp=log(temp2015+10)
gen log_elev=log(elev)
label variable prep2015 "Average Precipitations in 2015"
label variable temp2015 "Average Temperature in 2015"
label var costal "Dummy for Ocean or Lake Intersection"
label var log_precip "Log of Average Precipitations in 2015"
label var log_temp "Log of Average Temperature in 2015"
label var log_elev "Log Average Elevation"
label var capital "Dummy of Capital City" 
label var dry "City not in Major River Watershed"
label var elev "Average Elevation (m)"
 // drop the old version gdp dataset
// drop gdp2015
label var gdp2 "GDP of 2015 (2023 vintage)"
sort areaid
save "AUE_ExternalSources.dta", replace
clear

/*Import the geographic indexes of AUE (AUE 5km does not touth with any boundary, so the detour_5km_nb=sdetour_5km_b )*/
import delimited "AUE_GeographicIndicators.csv", clear
label variable share_of_barrier_10km_b "Share of barriers within 10km of City Center(with boundary)"
label variable share_of_barrier_10km_nb "Share of barriers within 10km of City Center(no boundary)"
label variable share_of_barrier_5km_nb "Share of barriers within 5km of City Center(no boundary)"
label variable nonconvexity_10km_nb "Average Local Non-convexity within 10km of City Center(no boundary)"
label variable nonconvexity_5km_nb "Average Local Non-convexity within 5km of City Center(no boundary)"
label variable nonconvexity_10km_b "Average Local Non-convexity within 10km of City Center(with boundary)"
label variable detour_10km_nb "Average detour within 10km of City Center(no boundary)"
label variable detour_5km_nb "Average detour within 5km of City Center(no boundary)"
label variable detour_10km_b "Average detour within 10km of City Center(with boundary)"
label variable detour_5km_b "Average detour within 5km of City Center(with boundary)"
label variable nonconvexity_elastic_b "Average Local Non-convexity within elastic ratio of City Center(with boundary)"
label variable nonconvexity_elastic_nb "Average Local Non-convexity within elastic ratio of City Center(no boundary)"
label variable share_barrier_elastic_nb "Share of barriers within elastic ratio of City Center(no boundary)"
label variable share_barrier_elastic_b "Share of barriers within elastic ratio of City Center(with boundary)"
label variable share_barrier_40km_b "Share of barriers within 40km of City Center(with boundary)"
sort areaid
save "AUE_GeographicIndicators.dta", replace
clear

/*Combine the external sources, geographic indexes with the original attributes from AUE*/
import delimited "AUE_OriginalData.csv", clear
// Transform the unit from 'ha' to 'km2'//
gen bu_areakm= bu_areaha/100 
gen densbkm= population/ bu_areakm
gen areakm=urbanextentha/100
keep areaid country population frgmentationsaturationbuiltupare bu_areakm densbkm areakm
label variable areakm "Total area of urban extend (km2)"
label variable population "Total Resident Population in 2014"
label variable densbkm "Density of Resident Population in Total Built-up area in 2014"
label variable bu_areakm "Total built-up area in 2014 (km2)"
label variable frgmentationsaturationbuiltupare "Share of build-up area within urban unit"
sort areaid
merge areaid using "AUE_ExternalSources.dta"
drop _merge
sort areaid
merge areaid using "AUE_GeographicIndicators.dta"
drop _merge
gen log_pop=log(population)
gen log_gdppc=log(gdp2/population)
gen log_densb=log(densbkm)
gen height=buvolu_tot/(bu_areakm*1000000)
gen log_height_new=log(height)

egen sum_countryPOP=sum(population), by ( country )
egen sum_countryUrbanArea=sum(areakm), by ( country )
gen endogenous_radius=2*sqrt((population*( sum_countryUrbanArea- areakm)/(sum_countryPOP- population))/_pi)
gen log_radius=log(endogenous_radius)
drop sum_countryPOP sum_countryUrbanArea

label var endogenous_radius "Endogenous radius"
label var log_radius "Log of Endogenous radius"
label var height "Average build-up area height (m)"
label var log_height_new "Log of average build-up area height (m)"
label var buvolu_tot "Total Build-up area volume (m3)"
label var buvolu_non "Non-residential Build-up area volume (m3)"
label variable log_pop "Log of Total Resident Population in 2014"
label variable log_gdppc "Log of GDPPC in 2015 (2023 vintage)"
label variable log_densb "Log of Density of Resident Population in Total Built-up area in 2014"
save "AUE_allvariables_for regressions.dta", replace

/*----------------------------------------------------------------------------------------------------------------------------------------------------*/











/*-----------------------------------------------------------------Creation of GHUB datasets----------------------------------------------------------*/
/*Import the external variables (Gridded data)*/
import delimited "GHUB_ExternalSources.csv", clear
gen dry=1 if riverba == "0" | riverba == ""
replace dry=0 if dry==.
drop riverba

split soil , parse(;)
rename soil1 soil_
drop soil soil2
split climate, parse(,)
split climate1 , parse(;)
rename climate11 climate_
drop climate1 climate2 climate3 climate4 climate12 climate
split biom , parse(,)
rename biom1 biom_
drop biom biom2

replace elevation =1 if elevation <1

drop gdp
replace bu100=bu100/1000000

gen log_precip=log(precip2015+1)
gen log_temp=log(temp2015+10)
gen log_pop=log(pop100)
gen log_gdppc=log(gdp2/pop100)
gen log_densb=log(pop100/bu100)
gen log_elev=log(elevation)
gen log_bu=log(bu100)
gen sharebu=bu100/ue_area
gen height=vol_tot/(bu100*1000000)
gen log_height_new=log(height)

label var vol_tot "Total Build-up area volume (m3)"
label var vol_non "Non-residential Build-up area volume (m3)"
label variable gdp2 "GDP of 2015 (2023 vintage)"
label variable precip2015 "Average Precipitations in 2015"
label variable temp2015 "Average Temperature in 2015"
label variable pop100 "Total Resident Population in 2015 (2023 vintage)"
label variable bu100 "Total Built-up area within urban unit (2023 vintage)"
label variable ue_area "Total area of urban unit (km2)"
label var elevation "Average Elevation (m)"
label variable continent "Continent of the city"
label variable country "Country of the city"
label variable countr_iso "Country iso"
label variable log_pop "Log of total Resident Population in 2015"
label variable sharebu "Share of build-up area within urban unit"
label variable log_densb "Log of Build-up area population density"
label variable log_precip "Log of Average Precipitations in 2015"
label variable log_temp "Log of Average Temperature in 2015"
label var log_gdppc "Log of GDP per capital (2023 vintage)"
label variable log_bu "Log of Total Built-up area within urban unit (km2)"
label var log_elev "Log of Average Elevation (m)"
label var costal "Dummy Variable for Areas with Substantial Ocean or Lake Intersection"
label var dry "City not in Major River Watershed"
label var capital "Dummy of Capital City" 
label var height "Average build-up area height (m)"
label var log_height_new "Log of average build-up area height (m)"
sort ghub_id
save "GHUB_ExternalSources.dta", replace
clear


/*Import the geographic indexes of GHUB*/
import delimited "GHUB_GeographicIndicators.csv"
label variable ghub_id "Urban unit ID in GHUB dataset"
label variable share_of_barrier_10km_b "Share of barriers within 10km of City Center(with boundary)"
label variable share_of_barrier_5km_b "Share of barriers within 5km of City Center(with boundary)"
label variable share_of_barrier_10km_nb "Share of barriers within 10km of City Center(no boundary)"
label variable share_of_barrier_5km_nb "Share of barriers within 5km of City Center(no boundary)"
label variable nonconvexity_10km_nb "Average Local Non-convexity within 10km of City Center(no boundary)"
label variable nonconvexity_5km_nb "Average Local Non-convexity within 5km of City Center(no boundary)"
label variable nonconvexity_10km_b "Average Local Non-convexity within 10km of City Center(with boundary)"
label variable nonconvexity_5km_b "Average Local Non-convexity within 5km of City Center(with boundary)"
label variable detour_10km_nb "Average detour within 10km of City Center(no boundary)"
label variable detour_5km_nb "Average detour within 5km of City Center(no boundary)"
label variable detour_10km_b "Average detour within 10km of City Center(with boundary)"
label variable detour_5km_b "Average detour within 5km of City Center(with boundary)"
label variable nonconvexity_elastic_b "Average Local Non-convexity within elastic ratio of City Center(with boundary)"
label variable nonconvexity_elastic_nb "Average Local Non-convexity within elastic ratio of City Center(no boundary)"
label variable share_barrier_elastic_nb "Share of barriers within elastic ratio of City Center(no boundary)"
label variable share_barrier_elastic_b "Share of barriers within elastic ratio of City Center(with boundary)"
label variable detour_40km_b "Average detour within 40km of City Center(with boundary)"
label variable share_barrier_40km_b "Share of barriers within 40km of City Center(with boundary)"
label variable nonconvexity_40km_b "Average Local Non-convexity within 40km of City Center(with boundary)"
rename share_barrier_40km_b share_of_barrier_40km_b
sort ghub_id
save "GHUB_GeographicIndicators.dta", replace
clear

/*Combine the external sources with geographic indexes*/
use "GHUB_ExternalSources.dta"
merge ghub_id using "GHUB_GeographicIndicators.dta"
drop if _merge==2
drop _merge

egen sum_countryPOP=sum(pop100), by ( country )
egen sum_countryUrbanArea=sum(bu100), by ( country )
gen endogenous_radius=2*sqrt((pop100*( sum_countryUrbanArea- bu100)/(sum_countryPOP- pop100))/_pi)
gen log_radius=log(endogenous_radius)
drop sum_countryUrbanArea sum_countryUrbanArea
label var endogenous_radius "Endogenous radius"
label var log_radius "Log of Endogenous radius"
save "GHUB_allvariables_for regressions.dta", replace
/*----------------------------------------------------------------------------------------------------------------------------------------------------*/

















/*-----------------------------------------------------------------Creation of GUB datasets----------------------------------------------------------*/


/*Import the external variables (Gridded data)*/
import delimited "GUB_ExternalSources.csv", clear
gen dry=1 if riverba == "0" | riverba == ""
replace dry=0 if dry==.
drop riverba

split soil , parse(;)
rename soil1 soil_
drop soil soil2
split climate, parse(,)
split climate1 , parse(;)
rename climate11 climate_
drop climate1 climate2 climate3 climate4 climate12 climate
split biom , parse(,)
rename biom1 biom_
drop biom biom2
replace elevation =1 if elevation <1
drop gdp 
replace bu100=bu100/1000000

gen log_precip=log(precip2015+1)
gen log_temp=log(temp2015+10)
gen log_pop=log(pop100)
gen log_gdppc=log(gdp2/pop100)
gen log_densb=log(pop100/bu100)
gen log_elev=log(elevation)
gen log_bu=log(bu100)
gen sharebu=bu100/ue_area
gen densb= pop100/bu100

gen height=vol_tot/(bu100*1000000)
gen log_height_new=log(height)

label var densb "Population density of build-up area"
label variable precip2015 "Average Precipitations in 2015"
label variable temp2015 "Average Temperature in 2015"
label variable pop100 "Total Resident Population in 2015 (2023 vintage)"
label variable bu100 "Total Built-up area within urban unit (km2)"
label variable ue_area "Total area of urban unit (km2)"
label var elevation "Average Elevation (m)"
label variable country "Country of the city"
label variable country_iso "Country iso"
label variable log_pop "Log of total Resident Population in 2015 (2023 vintage)"
label variable sharebu "Share of build-up area within urban unit"
label variable log_densb "Log of Build-up area population density"
label variable log_precip "Log of Average Precipitations in 2015"
label variable log_temp "Log of Average Temperature in 2015"
label var log_gdppc "Log of GDP per capital (2023 vintage)"
label variable log_bu "Log of Total Built-up area within urban unit (km2)"
label var log_elev "Log of Average Elevation (m)"
label var costal "Dummy Variable for Areas with Substantial Ocean or Lake Intersection"
label var dry "City not in Major River Watershed"
label var capital "Dummy of Capital City" 
label var height "Average build-up area height (m)"
label var log_height_new "Log of average build-up area height (m)"
label var vol_tot "Total Build-up area volume (m3)"
label var vol_non "Non-residential Build-up area volume (m3)"
label variable gdp2 "GDP of 2015 (2023 vintage)"
sort gub_id
drop fid_ urbanare_1 gub_id_1 urbanarea new_capi
save "GUB_ExternalSources.dta", replace


/*Import the geographic indexes of GUB*/
import delimited "GUB_GeographicIndicators.csv", clear
label variable gub_id "Urban unit ID in GUB dataset"
label variable share_of_barrier_10km_b "Share of barriers within 10km of City Center(with boundary)"
label variable share_of_barrier_5km_b "Share of barriers within 5km of City Center(with boundary)"
label variable share_of_barrier_10km_nb "Share of barriers within 10km of City Center(no boundary)"
label variable share_of_barrier_5km_nb "Share of barriers within 5km of City Center(no boundary)"
label variable nonconvexity_10km_nb "Average Local Non-convexity within 10km of City Center(no boundary)"
label variable nonconvexity_5km_nb "Average Local Non-convexity within 5km of City Center(no boundary)"
label variable nonconvexity_10km_b "Average Local Non-convexity within 10km of City Center(with boundary)"
label variable nonconvexity_5km_b "Average Local Non-convexity within 5km of City Center(with boundary)"
label variable detour_10km_nb "Average detour within 10km of City Center(no boundary)"
label variable detour_5km_nb "Average detour within 5km of City Center(no boundary)"
label variable detour_10km_b "Average detour within 10km of City Center(with boundary)"
label variable detour_5km_b "Average detour within 5km of City Center(with boundary)"
label variable nonconvexity_elastic_b "Average Local Non-convexity within elastic ratio of City Center(with boundary)"
label variable nonconvexity_elastic_nb "Average Local Non-convexity within elastic ratio of City Center(no boundary)"
label variable share_barrier_elastic_nb "Share of barriers within elastic ratio of City Center(no boundary)"
label variable share_barrier_elastic_b "Share of barriers within elastic ratio of City Center(with boundary)"
label variable detour_40km_b "Average detour within 40km of City Center(with boundary)"
label variable share_barrier_40km_b "Share of barriers within 40km of City Center(with boundary)"
label variable nonconvexity_40km_b "Average Local Non-convexity within 40km of City Center(with boundary)"
rename share_barrier_40km_b share_of_barrier_40km_b
sort gub_id
save "GUB_GeographicIndicators.dta", replace
clear


/*Combine the external sources with geographic indexes*/
use "GUB_ExternalSources.dta"
merge gub_id using "GUB_GeographicIndicators.dta"
drop if _merge==2
drop _merge
egen sum_countryPOP=sum(pop100), by ( country )
egen sum_countryUrbanArea=sum(bu100), by ( country )
gen endogenous_radius=2*sqrt((pop100*( sum_countryUrbanArea- bu100)/(sum_countryPOP- pop100))/_pi)
gen log_radius=log(endogenous_radius)
drop sum_countryUrbanArea sum_countryUrbanArea
label var endogenous_radius "Endogenous radius"
label var log_radius "Log of Endogenous radius"
save "GUB_allvariables_for regressions.dta", replace
clear




/*Name
label var el_av_als "Average Elevation"
label var log_elev "Log Average Elevation"
label var log_precip "Log of Average Precipitations in 2015"
label var log_temp "Log of Average Temperature in 2015"
label var capital "Dummy of Capital City" 
label var densityb "Density of Resident Population in Total Built-up area in 2015 (UCDB 2019 Data Vintage)"
label var lodensb "Log of Density of Resident Population in Total Built-up area in 2015 (UCDB 2019 Data Vintage)"
label var sharebu "Share of Total Built-up area of the Area of Urban Centre in 2015 (UCDB 2019 Data Vintage)"
label var logdpcap15 "(Old version) Log GDP per Capita in 2015"
label variable e_rb_nm_lst "Major river bashin"
label variable e_gr_av14 "Average greeness"
label var e_ec2e_r15 "Residential CO2 Emissions non-organic 2015"
label var e_ec2e_i15 "Industrial CO2 Emissions non-organic 2015"
label var e_ec2e_t15 "Transportation CO2 Emissions non-organic 2015"
label var dry "City not in Major River Watershed"
label var climate "Climate Class"
label var biome "Biome Class" 
label var soilYes "Soil Group"
label variable coastal "Dummy for Ocean or Lake Intersection"
Total Resident Population in 2015
Log of Total Resident Population in 2015
Sum of GDP PPP Values for Year 2015
Total built-up area in 2015
Log of Density of Resident Population in Total Built-up area in 2014
Share of build-up area within urban unit
*/


