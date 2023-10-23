import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DCFMonteCarlo:
    def __init__(self, **kwargs):
        params = [
                    "rev_gwth", "rev_gwth_stddev", 
                    "cogs_base_rate", "infl_rate", "infl_rate_stddev", 
                    "sga_base_rate", "sga_stddev", 
                    "dep_amort_base_rate", "dep_amort_stddev", 
                    "int_exp", "int_exp_stddev", 
                    "tax_rate",
                    "curr_assets_base_rate", "curr_assets_gwth", "curr_assets_stddev",
                    "curr_liab_base_rate", "curr_liab_gwth", "curr_liab_stddev",
                    "capex_base_rate", "capex_gwth", "capex_gwth_stddev", 
                    "perp_gwth", "wacc", 
                    "num_sim", "num_f_years", 
                    "init_rev"
                ]
        for param in params:
            setattr(self, param, kwargs.get(param))

    def compute_net_income(self, rev, cogs, sga_rates, dep_amort_rates, int_exp_rates, tax_rates):
        return rev + cogs + sga_rates + dep_amort_rates + int_exp_rates + tax_rates
        
    def compute_fcf(self, net_income, dep_amort, deferred_taxes, tax_shield, int_exp, capex, var_nwc):
        return net_income - dep_amort - deferred_taxes - int_exp + tax_shield + capex + var_nwc
    
    def fcf_disc_coeff(self):
        return [1/(1 + self.wacc)**n for n in range(1, int(self.num_f_years) + 1)]

    def disc_fcf(self, fcf):
        return sum((fcf.transpose() * self.fcf_disc_coeff()).transpose())

    def term_val_npv(self, fcf):
        return (fcf.iloc[-1] * ((1 + self.perp_gwth) / (self.wacc - self.perp_gwth))) * self.fcf_disc_coeff()[-1]
    
    def revenues(self):
        rev_growth_rate = np.random.normal(self.rev_gwth, self.rev_gwth_stddev, int(self.num_f_years))
        revenues = [self.init_rev * (1 + rev_growth_rate[0])]
        for n in range(1, int(self.num_f_years)):
            revenues.append(revenues[n-1] * (1 + rev_growth_rate[n]))
        return revenues

    def cogs(self):
        inflation_rates = np.random.normal(self.infl_rate, self.infl_rate_stddev, int(self.num_f_years))
        cogs = [self.cogs_base_rate * self.init_rev * inflation_rates[0]]
        for n in range(1, int(self.num_f_years)):
            cogs.append(cogs[n-1] * (1 + inflation_rates[n]))
        return cogs
    
    def sga_rates(self):
        return np.random.normal(self.sga_base_rate, self.sga_stddev, int(self.num_f_years))
    
    def dep_amort_rates(self):
        return np.random.normal(self.dep_amort_base_rate, self.dep_amort_stddev, int(self.num_f_years))
    
    def int_exp_rates(self):
        return np.random.normal(self.int_exp, self.int_exp_stddev, int(self.num_f_years))

    def tax_rates(self):
        return np.ones(int(self.num_f_years)) * self.tax_rate

    def var_nwc(self):
        init_ca = self.curr_assets_base_rate * self.init_rev
        init_cl = self.curr_liab_base_rate * self.init_rev
        current_assets_rates = np.random.normal(self.curr_assets_gwth, self.curr_assets_stddev, int(self.num_f_years))
        current_liab_rates = np.random.normal(self.curr_liab_gwth, self.curr_liab_stddev, int(self.num_f_years))
        current_assets = [init_ca]
        current_liab = [init_cl]
        var_nwc = [0]
        for n in range(1, int(self.num_f_years)):
            current_assets.append(current_assets[n-1] * (1 + current_assets_rates[n]))
            current_liab.append(current_liab[n-1] * (1 + current_liab_rates[n]))
            var_nwc.append((current_assets[n-1] - current_liab[n-1]) - (current_assets[n] - current_liab[n]))
        return var_nwc
    
    def capex(self):
        capex_gwth_rates = np.random.normal(self.capex_gwth, self.capex_gwth_stddev, int(self.num_f_years))
        capex = [self.capex_base_rate * self.init_rev * (1 + capex_gwth_rates[0])]
        for n in range(1, int(self.num_f_years)):
            capex.append(capex[n-1] * (1 + capex_gwth_rates[n]))
        return capex
        

    def estimate(self):
        simulated_values = []
        pb_mod = round(int(self.num_sim)/50)
        pb_pct = 0.0

        for n in range(0, int(self.num_sim)):
            pb = '|'*int((int(self.num_sim)*pb_pct)/pb_mod)
            print(f"\rSimulations completed: {n+1}")
            print(f'> {int(pb_pct*100)}% {pb}', end='\033[1A')

            income_items = pd.DataFrame()
            fcf = pd.DataFrame()
            income_items['revenues'] = self.revenues()
            income_items['cogs'] = self.cogs()
            income_items['sga'] = self.sga_rates() * income_items['revenues']
            income_items['ebitda'] = income_items['revenues'] - income_items['cogs'] - income_items['sga']
            income_items['dep_amort'] = self.dep_amort_rates() * income_items['revenues']
            income_items['ebit'] = income_items['ebitda'] - income_items['dep_amort']
            income_items['int_exp'] = self.int_exp_rates() * income_items['revenues']
            income_items['tax_exp'] = self.tax_rates() * (income_items['ebit'] - income_items['int_exp'])
            income_items['net_income'] = income_items['ebit'] - income_items['int_exp'] - income_items['tax_exp'] 
            
            fcf['net_income'] = income_items['net_income']
            fcf['dep_amort'] = income_items['dep_amort']
            fcf['int_exp'] = income_items['int_exp']
            fcf['capex'] = self.capex()
            fcf['capex'].multiply(-1)
            fcf['var_nwc'] = self.var_nwc()
            fcf['fcf'] = fcf.sum(axis=1)
            
            disc_fcf = self.disc_fcf(fcf['fcf']) + self.term_val_npv(fcf['fcf'])
            simulated_values.append(disc_fcf)

            pb_pct = pb_pct + 0.02 if n%pb_mod == 0 else pb_pct
            pb_pct = 1.0 if n+2 == int(self.num_sim) else pb_pct

        print(f"\n\n\nDCF Mean: {np.round(np.mean(simulated_values), 2)}")
        print(f"DCF Standard deviation: {np.round(np.std(simulated_values), 2)}\n")
        
        fig, ax = plt.subplots(figsize=(12,6))

        ax.hist(simulated_values, bins=50)
        ax.set_title("Distribution of DCF values")
        plt.figtext(0.25,0.01, f"Estimated DCF value: {np.mean(simulated_values)}", ha='center', fontsize=10, color='red')
        # plt.show()

params = {}
with open('params.txt') as file:
    for line in file:
        (key, val) = line.split(":")
        params[key] = float(val.rstrip('\n'))

dcf = DCFMonteCarlo(**params)
dcf.estimate()
 
            


            






