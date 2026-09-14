import math
import numpy as np
# overloaded version to work for the individual decays 
def return_signal_SF(year,mass_point,decay, y_uu = 2.0, y_x = 2.0, WB_BR = 0.50, ZT_BR = 0.25, HT_BR = 0.25):    ## parameters are theory couplings
	## input signal mass point in the format Suu$Mass_chi$Mass
	## can be imported into other modules when needed

	if( abs(WB_BR+ZT_BR+HT_BR - 1.0) > 1e-3    ): 
		print("ERROR: branching fractions do not add to 1.0: WB_BR = %s, ZT_BR = %s, HT_BR = %s."%(WB_BR,ZT_BR,HT_BR))

	Suu_mass = 4000
	if "Suu5" in mass_point: Suu_mass = 5000
	elif "Suu6" in mass_point: Suu_mass = 6000
	elif "Suu7" in mass_point: Suu_mass = 7000
	elif "Suu8" in mass_point: Suu_mass = 8000

	chi_mass = 1000
	if "chi1p5" in mass_point: chi_mass = 1500
	if "chi2" in mass_point: chi_mass = 2000
	if "chi2p5" in mass_point: chi_mass = 2500
	if "chi3" in mass_point: chi_mass = 3000

	# the production xs of Suu depends on y_uu^2, all values below are reference values for y_uu = 2
	
	### These were estimates previously
	"""Suu_prod_xs = { "4000": 1000.  * (pow(1.0*y_uu,2) / pow(2.0,2) ),   #fb
					"5000": 500.   * (pow(1.0*y_uu,2) / pow(2.0,2) ),   #fb
					"6000": 200.   * (pow(1.0*y_uu,2) / pow(2.0,2) ),   #fb  
					"7000": 28     * (pow(1.0*y_uu,2) / pow(2.0,2) ),   #fb
					"8000": 3.5    * (pow(1.0*y_uu,2) / pow(2.0,2) )    #fb
	    } """

	### these are the "real" number estimates, they use reference y_uu of 0.2
	Suu_prod_xs = { "4000": 32.0  * (pow(1.0*y_uu,2) / pow(0.2,2) ),   #fb
					"5000": 6.95   * (pow(1.0*y_uu,2) / pow(0.2,2) ),   #fb
					"6000": 1.37   * (pow(1.0*y_uu,2) / pow(0.2,2) ),   #fb  
					"7000": 0.231     * (pow(1.0*y_uu,2) / pow(0.2,2) ),   #fb
					"8000": 0.0303    * (pow(1.0*y_uu,2) / pow(0.2,2) )    #fb
	    }

	r""" #this is the reference to Bogdan's new paper
	\bibitem{Dobrescu:2024mdl}
	B.~A.~Dobrescu,
	``TeV-scale particles and LHC events with dijet pairs,''
	[arXiv:2411.04121 [hep-ph]]. 

	"""

	collected_data = {"2015":19.52,"2016":16.81, "2017":41.48, "2018":59.83}  # collected luminosity per year 

	Z_had_BR = 0.6991
	W_had_BR = 0.6741
	H_had_BR = 0.58
	t_had_BR = 0.6741

	frac_of_events_used = 1.0

	nEvents = 0

	if decay in ["WBWB","ZTZT","HTHT"]:
		if "Suu8" in mass_point or "Suu7" in mass_point or "Suu6" in mass_point:
			nEvents = 30000
		else: nEvents = 60000
	elif decay in ["WBHT","WBZT","HTZT"]:
		if "Suu8" in mass_point or "Suu7" in mass_point or "Suu6" in mass_point:
			nEvents = 50000
		else: nEvents = 100000




 	### 2016preAPV and 2016postAPV have fewer stats (about half) than 2017/2018
	### work for how these fractions came about:
		#### 2016preAPV for pure: 32223  --> 54%
		#### 2016postAPV for pure: 27600 --> 46
		#### 2016preAPV  for mixed: 54000
		#### 2016postapV for mixed: 46000

	if year == "2015":
		nEvents *= 0.54  
	elif year == "2016":
		nEvents *= 0.46   

	lumi_eff = nEvents*frac_of_events_used / (  Suu_prod_xs[ "%s"%Suu_mass ] * calculate_Suu_to_chi_chi_BR( Suu_mass, chi_mass, y_uu, y_x)    )
	lumi_year = collected_data[year]

	#print("decay is %s"%decay)
	#print("Suu_prod_xs is %f"%Suu_prod_xs[mass_point][year])
	#print("nevents is %i"%nEvents)

	#print("Effective luminosity is %f"%lumi_eff)
	SF_fullBR = lumi_year/lumi_eff

	#if not decay:
	#	return SF_fullBR

	WBWB_had_BR = (WB_BR*WB_BR)*(W_had_BR)*(W_had_BR)   # the Suu -> chi chi -> WBWB -> all had branching fraction
	HTHT_had_BR = (HT_BR*HT_BR)*(H_had_BR*t_had_BR)*(H_had_BR*t_had_BR)
	ZTZT_had_BR = (ZT_BR*ZT_BR)*(Z_had_BR*t_had_BR)*(Z_had_BR*t_had_BR)
	WBHT_had_BR = 2*(WB_BR*HT_BR)*(W_had_BR)*(H_had_BR*t_had_BR)
	WBZT_had_BR = 2*(WB_BR*ZT_BR)*(W_had_BR)*(Z_had_BR*t_had_BR)
	HTZT_had_BR = 2*(HT_BR*ZT_BR)*(H_had_BR*t_had_BR)*(Z_had_BR*t_had_BR)


	if decay == "WBWB":
		return SF_fullBR*WBWB_had_BR
	elif decay == "HTHT":
		return SF_fullBR*HTHT_had_BR
	elif decay == "ZTZT":
		return SF_fullBR*ZTZT_had_BR
	elif decay == "WBHT":
		return SF_fullBR*WBHT_had_BR
	elif decay == "WBZT":
		return SF_fullBR*WBZT_had_BR
	elif decay == "HTZT":
		return SF_fullBR*HTZT_had_BR
	else:
		print("ERROR: decay %s did not match any of the accepted SuuToChiChi decays (WBWB,HTHT,ZTZT,WBHT,WBZT,HTZT)."%decay)
		return None


def return_Suu_to_chi_chi_xs(mass_point,decay, y_uu = 2.0, y_x = 2.0, WB_BR = 0.50, ZT_BR = 0.25, HT_BR = 0.25, quiet = False):
	## input signal mass point in the format Suu$Mass_chi$Mass
	## can be imported into other modules when needed


	Suu_mass = 4000
	if "Suu5" in mass_point: Suu_mass = 5000
	elif "Suu6" in mass_point: Suu_mass = 6000
	elif "Suu7" in mass_point: Suu_mass = 7000
	elif "Suu8" in mass_point: Suu_mass = 8000

	chi_mass = 1000
	if "chi1p5" in mass_point: chi_mass = 1500
	if "chi2" in mass_point: chi_mass = 2000
	if "chi2p5" in mass_point: chi_mass = 2500
	if "chi3" in mass_point: chi_mass = 3000

	# the production xs of Suu depends on y_uu^2, all values below are reference values for y_uu = 2
	Suu_prod_xs = { "4000": 32.0  * (pow(1.0*y_uu,2) / pow(0.2,2) ),   #fb
					"5000": 6.95   * (pow(1.0*y_uu,2) / pow(0.2,2) ),   #fb
					"6000": 1.37   * (pow(1.0*y_uu,2) / pow(0.2,2) ),   #fb  
					"7000": 0.231     * (pow(1.0*y_uu,2) / pow(0.2,2) ),   #fb
					"8000": 0.0303    * (pow(1.0*y_uu,2) / pow(0.2,2) )    #fb
	}


	collected_data = {"2015":19.5,"2016":16.80, "2017":41.4, "2018":59.8}  # collected luminosity per year 

	#WB_BR = 0.50
	#ZT_BR = 0.25
	#HT_BR = 0.25

	Z_had_BR = 0.6991
	W_had_BR = 0.6741
	H_had_BR = 0.58
	t_had_BR = 0.6741

	WBWB_had_BR = (WB_BR*WB_BR)*(W_had_BR)*(W_had_BR)   # the Suu -> chi chi -> WBWB -> all had branching fraction
	HTHT_had_BR = (HT_BR*HT_BR)*(H_had_BR*t_had_BR)*(H_had_BR*t_had_BR)
	ZTZT_had_BR = (ZT_BR*ZT_BR)*(Z_had_BR*t_had_BR)*(Z_had_BR*t_had_BR)
	WBHT_had_BR = 2*(WB_BR*HT_BR)*(W_had_BR)*(H_had_BR*t_had_BR)
	WBZT_had_BR = 2*(WB_BR*ZT_BR)*(W_had_BR)*(Z_had_BR*t_had_BR)
	HTZT_had_BR = 2*(HT_BR*ZT_BR)*(H_had_BR*t_had_BR)*(Z_had_BR*t_had_BR)

	Suu_to_chi_chi_to_VLQs_xs = Suu_prod_xs[ "%s"%Suu_mass ] * calculate_Suu_to_chi_chi_BR( Suu_mass, chi_mass, y_uu, y_x) 




	if decay == "WBWB":
		if not quiet: print( "mass point is %s, decay is %s, Suu production xs is %s, Suu -> chi chi BR is %s, hadronic BR is %s,  hadronic xs is %s"%(mass_point,decay,Suu_prod_xs[ "%s"%Suu_mass ], calculate_Suu_to_chi_chi_BR( Suu_mass, chi_mass, y_uu, y_x),WBWB_had_BR , np.around(Suu_to_chi_chi_to_VLQs_xs*WBWB_had_BR,2)   ) )

		return Suu_to_chi_chi_to_VLQs_xs*WBWB_had_BR
	elif decay == "HTHT":
		if not quiet: print( "mass point is %s, decay is %s, Suu production xs is %s, Suu -> chi chi BR is %s, hadronic BR is %s,  hadronic xs is %s"%(mass_point,decay,Suu_prod_xs[ "%s"%Suu_mass ], calculate_Suu_to_chi_chi_BR( Suu_mass, chi_mass, y_uu, y_x), HTHT_had_BR, np.around(Suu_to_chi_chi_to_VLQs_xs*HTHT_had_BR,2)   ) )

		return Suu_to_chi_chi_to_VLQs_xs*HTHT_had_BR
	elif decay == "ZTZT":
		if not quiet: print( "mass point is %s, decay is %s, Suu production xs is %s, Suu -> chi chi BR is %s, hadronic BR is %s,  hadronic xs is %s"%(mass_point,decay,Suu_prod_xs[ "%s"%Suu_mass ], calculate_Suu_to_chi_chi_BR( Suu_mass, chi_mass, y_uu, y_x), ZTZT_had_BR, np.around(Suu_to_chi_chi_to_VLQs_xs*ZTZT_had_BR,2)   ) )

		return Suu_to_chi_chi_to_VLQs_xs*ZTZT_had_BR
	elif decay == "WBHT":
		if not quiet: print( "mass point is %s, decay is %s, Suu production xs is %s, Suu -> chi chi BR is %s, hadronic BR is %s,  hadronic xs is %s"%(mass_point,decay,Suu_prod_xs[ "%s"%Suu_mass ], calculate_Suu_to_chi_chi_BR( Suu_mass, chi_mass, y_uu, y_x), WBHT_had_BR, np.around(Suu_to_chi_chi_to_VLQs_xs*WBHT_had_BR,2)  ) )

		return Suu_to_chi_chi_to_VLQs_xs*WBHT_had_BR
	elif decay == "WBZT":
		if not quiet: print( "mass point is %s, decay is %s, Suu production xs is %s, Suu -> chi chi BR is %s, hadronic BR is %s,  hadronic xs is %s"%(mass_point,decay,Suu_prod_xs[ "%s"%Suu_mass ], calculate_Suu_to_chi_chi_BR( Suu_mass, chi_mass, y_uu, y_x), WBZT_had_BR, np.around(Suu_to_chi_chi_to_VLQs_xs*WBZT_had_BR ,2) ) )

		return Suu_to_chi_chi_to_VLQs_xs*WBZT_had_BR
	elif decay == "HTZT":
		if not quiet: print( "mass point is %s, decay is %s, Suu production xs is %s, Suu -> chi chi BR is %s, hadronic BR is %s,  hadronic xs is %s"%(mass_point,decay,Suu_prod_xs[ "%s"%Suu_mass ], calculate_Suu_to_chi_chi_BR( Suu_mass, chi_mass, y_uu, y_x),HTZT_had_BR,  np.around(Suu_to_chi_chi_to_VLQs_xs*HTZT_had_BR,2)   ) )

		return Suu_to_chi_chi_to_VLQs_xs*HTZT_had_BR
	else:
		print("ERROR: decay %s did not match any of the accepted SuuToChiChi decays (WBWB,HTHT,ZTZT,WBHT,WBZT,HTZT)."%decay)
		return None


def return_Suu_to_chi_chi_full_xs(Suu_mass, chi_mass, y_uu = 2.0, y_x = 2.0, WB_BR=0.5, ZT_BR=0.25, HT_BR=0.25, quiet = False):
	## input signal mass point in the format Suu$Mass_chi$Mass
	## can be imported into other modules when needed


	# the production xs of Suu depends on y_uu^2, all values below are reference values for y_uu = 2
	Suu_prod_xs = { "4000": 32.0  * (pow(1.0*y_uu,2) / pow(0.2,2) ),   #fb
					"5000": 6.95   * (pow(1.0*y_uu,2) / pow(0.2,2) ),   #fb
					"6000": 1.37   * (pow(1.0*y_uu,2) / pow(0.2,2) ),   #fb  
					"7000": 0.231     * (pow(1.0*y_uu,2) / pow(0.2,2) ),   #fb
					"8000": 0.0303    * (pow(1.0*y_uu,2) / pow(0.2,2) )    #fb
	}

	Suu_to_chi_chi_to_VLQs_xs = Suu_prod_xs[ "%s"%Suu_mass ] * calculate_Suu_to_chi_chi_BR( Suu_mass, chi_mass, y_uu, y_x) 

	return Suu_to_chi_chi_to_VLQs_xs





def calculate_Suu_to_chi_chi_BR(Suu_mass, chi_mass, y_uu = 2.0, y_x = 2.0):
	# y_x = sqrt(pow(y_x_R,2) = pow(y_x_L,2)), eq 2.3 from bogdan's paper http://www.arxiv.org/pdf/1810.09429
	
	 # Suu to up quark pair partial width
	tau_SuuToUU = pow(y_uu , 2)* Suu_mass / (32*math.pi)   

	# Suu to VLQ pair partial width  (eq 2.8 of Bogdan's paper)

	tau_SuuToChiChi = ( pow(y_x,2) * Suu_mass * ( 1 - 2*pow( (1.0*chi_mass)/(1.0*Suu_mass),2) ) * pow(  1 - 4*pow( (1.0*chi_mass)/(1.0*Suu_mass),2)  ,0.5) )  / ( 32*math.pi )

	#print("The branching fraction for M_Suu = %s / M_chi = %s is "%(Suu_mass,chi_mass), ( tau_SuuToChiChi / (tau_SuuToChiChi + tau_SuuToUU) ) ) 
	return (tau_SuuToChiChi / (tau_SuuToChiChi + tau_SuuToUU))



""" ## FROM BOGDAN's EMAIL

Leading order S_uu production cross section at the 13 TeV LHC for y_uu = 0.2:  sigma_LO(pp -> S_uu)

{M_S [TeV]  ,  sigma_LO(p p -> S_uu) [fb] ,  Kfactor} =
{ 4 ,  32.0 ,  1.28 } 
{ 5 ,  6.95 ,   1.25 }
{ 6 ,  1.37 ,   1.22 }
{ 7 ,  0.231 ,  1.19 }
{ 8 ,  0.0303 ,  1.17 }

Branching fraction   B(S_uu -> chi chi) =  1/( 1+  (yuu/ychi)^2  (1-2r)^{-1} (1-4r)^{-1/2}  ),
where  r =  (mchi/M_S)^2   (see second Eq 4.2 of 1912.13155). 

The NLO cross section for  p p -> S_uu -> chi chi with S_uu on-shell is: 
sigma(p p -> S_uu -> chi chi) = sigma_LO(p p -> S_uu)  Kfactor  B(S_uu -> chi chi) 

For the default values 
yuu -> 0.2  (coupling of Suu to u u)
ychi -> 0.3  (coupling of Suu to chi chi)
I get the following results:

{M_S [TeV]  ,  m_chi [TeV]  ,  
B(S_uu -> chi chi) , sigma(p p -> S_uu -> chi chi)  [fb] } =
{ 4 , 1 ,     0.630 ,  25.8}
{ 4 , 1.5 ,  0.517 ,  21. 2}
{ 5 ,  1 ,    0.655 ,  5.69}
{ 5 ,  1.5 , 0.596 ,  5.18}
{ 5 ,  2 ,    0.479 ,  4.16}
{ 6 ,  1,     0.667 , 1.11}
{ 6 ,  1.5 , 0.630 , 1.05}
{ 6 ,  2 ,    0.566 , 0.946}
{ 6 ,  2.5 , 0.448 , 0.749}
{ 7 ,  1 ,    0.674 , 0.185}
{ 7 ,  1.5 , 0.649 , 0.178}
{ 7 ,  2 ,    0.607 , 0.167}
{ 7 ,  2.5 , 0.540 , 0.148}
{ 7 ,  3 ,    0.423 , 0.116}
{ 8 ,  1 ,    0.679 , 0.0241}
{ 8 ,  1.5 , 0.660 , 0.0234}
{ 8 ,  2 ,    0.630 , 0.0223}
{ 8 ,  2.5 , 0.586 , 0.0208}
{ 8 ,  3 ,    0.517 , 0.0183}

"""




