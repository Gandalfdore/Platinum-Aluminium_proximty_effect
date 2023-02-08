# -*- coding: utf-8 -*-
"""
Created on 

@author: Smelly_Giraffe
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import scipy.stats as ss


##########################

dataIV=np.loadtxt(fname = "G:\My Drive\Codes\Pt_Al_project\Pt-Al_21mK.txt", skiprows=3)  ### -200 - +200 uV span and -150 - +150 nA
dataTEMP=np.loadtxt(fname = "G:\My Drive\Codes\Pt_Al_project\Pt-Al cold 1 upward 21mK_temp.txt", skiprows=3)


#########################

def linear_func (x_data, a_1, a_0):
    """Takes x values, and linear eqn coefficents to output y = (x_data * a_1) + a_0"""
    sth = (x_data * a_1) + a_0
    return sth

####################


################## Looop for the resistance values caluclation and import

length_of_array = np.shape(dataIV)[0]  #how many different measurment have we done

data_range_low = 0#100
data_range_high = 401#301

resistances = np.empty([4,(length_of_array - 1)])
data_x = dataIV[:1, data_range_low : data_range_high ][0].reshape((-1, 1))

for i in range(1, np.shape(dataIV)[0]):
    
    data_y = dataIV[i:i+1, data_range_low : data_range_high ][0]
    
    regression = LinearRegression().fit(data_x, data_y)   ### taking the I-V curves and gving a function that fits them
    
    coeff = regression.coef_    ### this is the slope, aka the resistance. the a_1
    resistances[0,i-1] = coeff
    
    intercept = regression.intercept_  ### the a_0
    resistances[1,i-1] = intercept
    
    score=regression.score(data_x, data_y)  ### goodness of the fit
    resistances[2,i-1] = score
    
    # print("Progress:", int(100*(i+1)/length_of_array),"%")  ### progress ba, if you need
    
###############################
    


############ Looop for the temperature values import

temp_values = np.empty(length_of_array - 1)
    
for i in range(1, np.shape(dataTEMP)[0]):
    temp_values [i-1] = (dataTEMP[i,0] + dataTEMP[i,-1])/2

########################

######################## Add all the values together in one dataframe

# temp_values.sort()
resistances[3,:] = temp_values
resistances = resistances.transpose()   
df = pd.DataFrame(resistances, columns = ['Resistance','Voltage Offshift (at 150 nA)','regression score','Temperature'])


################### Generate histogram

diapason = df.Resistance.max() - df.Resistance.min()
bins = 10

df["Resistance_strati"] = pd.cut(df["Resistance"],bins, labels = False)
# df["Resistance_strati"].value_counts()

df["Resistance_strati"] = df.Resistance.min() + df["Resistance_strati"] * (diapason/bins)

# df = df[0:30]
# df = df[31:51]

df["Resistance_strati"].hist()

# df.hist()

median = df["Resistance"].median()
mode = df["Resistance"].mode()[0]
std_dev = df["Resistance"].std()
std_dev_percent = 100*df["Resistance"].std()/df["Resistance"].median()
std_dev_measur = 1 - df['regression score'].mean()

print("\n---------------------------------------------------------------------")
print("-----------------------Statistical parameters------------------------")
print("---------------------------------------------------------------------")
print("Median of the resistance:", median)
print("Mode of the resistance:", mode)
print("Deviation of the resistance:", std_dev)
print("Relative deviation of the resistance:", std_dev_percent, "%")
print("Relative deviation of the measurment:", std_dev_measur, "%")
print("---------------------------------------------------------------------\n")

##########################  Chi sqare  test #############

# df["Resistance_median"] = df["Resistance"].median()
chi, p = ss.chisquare( f_obs = df["Resistance"])

print ("Chi square = {0}   and   p = {1}".format(chi,p))
print("---------------------------------------------------------------------")

################################

###------------------Plotter-----------------------

fig1, (ax1) = plt.subplots(1, 1, figsize = (5, 3.5), dpi = 300)
fig1, (ax2) = plt.subplots(1, 1, figsize = (5, 3.5), dpi = 300)
# ax1.hist (a,bins)
ax1.plot(data_x,data_y,'.', color='#d63c49',linewidth=1.5)
ax1.plot(data_x,linear_func(data_x, coeff, intercept),'-', color='blue',linewidth=1.5)

# ax2.plot(df['Temperature'],df['Resistance'],'-', color='blue',linewidth=1.5)
# ax2.plot(df['Temperature'],df['Resistance'],'.', color='orange',linewidth=1.5)
ax2.errorbar(df['Temperature'],df['Resistance'],yerr = (1 - df['regression score'])*df['Resistance'],fmt = '.',linewidth=1.5, label='Resistances')


ax1.set_ylabel('$Voltage$ $[V]$', fontsize=12)
ax1.set_xlabel('$Current$ $[A]$', fontsize=12)

ax2.set_xlabel('$Temperature$ $[K]$', fontsize=12)
ax2.set_ylabel('$Resistance$ $[Ohm]$', fontsize=12)
# ax1.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
# ax1.tick_params(direction='in', which='major', length=6)
# ax1.tick_params(direction='in', which='minor', length=3)

# ax1.set_xlim([1,100])
# ax1.set_ylim([-0.7,3])
# ax1.set_yscale('log')
# plt.tight_layout()

# ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
# ax1.yaxis.set_major_locator(plt.MaxNLocator(6))
# fig1.savefig('Junction resistance 8,9,10.jpg')
plt.legend()
plt.show()


################# Contingency table ########

### H0 hyphotesis - there is no relation between the data points with temperature smaller or bigger then 0.37K and deviation from the median value

threshold = 0.33

df_00 = df[(df['Temperature'] <= threshold) & (df['Resistance'] > median)]
df_01 = df[(df['Temperature'] > threshold) & (df['Resistance'] > median)]
df_10 = df[(df['Temperature'] <= threshold) & (df['Resistance'] <= median)]
df_11 = df[(df['Temperature'] > threshold) & (df['Resistance'] <= median)]

contingency_table = [[len(df_00),len(df_01)],[len(df_10),len(df_11)]]
stat, p, dof, expected = ss.chi2_contingency(contingency_table)


print("\n---------------------------------------------------------------------")
print("----------------------- Pearson's Chi^2 test------------------------")
print("---------------------------------------------------------------------")

alpha = 0.05
print("contingency_table value is " + str(contingency_table))
print("expected value is " + str(expected))
print("dof value is " + str(dof))

print("\nstat value is " + str(stat))
print("p value is " + str(p))

if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (H0 holds true)')
    
print("---------------------------------------------------------------------")

############Chi2 sweep #######################

# def chisquare_for_TvsR (df, threshold = float):
#     """This func computes the chi^2 pearson test to see if thre is is no relation between the data points with temperature smaller or bigger then 0.37K and deviation from the median value
#     INPUTS: The aformentioned df, a treshold for the test 
#     OUTPUTS: The chi^2 parameters """
    
    
#     df_00 = df[(df['Temperature'] <= threshold) & (df['Resistance'] > median)]
#     df_01 = df[(df['Temperature'] > threshold) & (df['Resistance'] > median)]
#     df_10 = df[(df['Temperature'] <= threshold) & (df['Resistance'] <= median)]
#     df_11 = df[(df['Temperature'] > threshold) & (df['Resistance'] <= median)]
    
#     contingency_table = [[len(df_00),len(df_01)],[len(df_10),len(df_11)]]
    
#     stat, p, dof, expected = ss.chi2_contingency(contingency_table)
    
#     return stat, p, dof, expected

# thresholds = np.linspace (0.1,1.2,20)
# p_values = np.zeros(20)
# for i in range(0,len(thresholds)):
#     p_values[i] = chisquare_for_TvsR (df, thresholds[i])[1]
    
    

# fig2, (ax1) = plt.subplots(1, 1, figsize = (5, 3.5), dpi = 300)
# ax1.plot(thresholds, p_values,'-', color='#d63c49',linewidth=1.5)

# ax1.set_xlabel('$Temperature$ $threshold$ $[K]$', fontsize=12)
# ax1.set_ylabel('$p$', fontsize=12)

############Chi2 sweep #######################
