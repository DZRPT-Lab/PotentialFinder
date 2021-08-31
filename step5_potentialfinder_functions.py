#### modified for paper
import os
import pandas as pd
import numpy as np
from scipy.stats import linregress, linregress
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
import sys
import pathlib
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler
from dateutil.parser import parse

type_1_count, type_2_count, type_3_count, type_4_count = 0, 0, 0, 0

def find_exponents(df,
                   fractionToAnalyze=1,
                   outputPath='outputs',
                   outputTable=True,
                   outputPlots=True,
                   outputTablename='Testtable',
                   logToScreen=True,
                   columnFillThreshold=0.5,
                   exp_b_threshold=0.01,
                   exp_r_s_threshold=0.95,
                   logistic_r_s_threshold=0.95,
                   k_threshold=0.2,
                   maxrows=5000,
                   debug=False):
    '''find and plot plot exponential data in a pandas dataframe
    :param dataframe df: The dataframe containing date-like and metric columns
    :param str fractionToAnalyze: The last xx percent (0.0-1.0) of data (indexed by the found date columns)
    :param str outputPath: The subfolder to write output table and plots to (created if not existing)
    :param bln outputTable: Write table to outputPath/table (True|False)
    :param bln outputPlots: Plot outputs to outputPath/plots (True|False)
    :param str timescaling: (sec|min|hour|day|mon|year) scaling the exponential factors to the given time unit
    :param str outputTablename: The table name used in the table, and the output plots
    :param bln logToScreen: Print out messages (True|False)
    :param dbl columnFillThreshold: filter out columns with more than xx percent (0.0-1.0) of the data missing
    :param dbl exp_threshold: only show and plot results with an exponent higher than x
    :param dbl exp_r_s_threshold: only show and plot results with an r squared lower than x
    :param int maxrows: specify the maximum number of rows. If df.rowsize > maxrows a random sample of rows is used
    :param bln debug: raise errors instead of continuing if True
    :return: A Pandas dataframe with a list of all date and metric column combinations with their exponential factor and R squared
    :rtype: dataframe
    '''
	
    global type_1_count, type_2_count, type_3_count, type_4_count

    def is_date(string, fuzzy=False):
        """
        Return whether the string can be interpreted as a date.

        :param string: str, string to check for date
        :param fuzzy: bool, ignore unknown tokens in string if True
        """
        try:
            float(string)
            return False
        except:
            pass

        try:
            parse(string, fuzzy=fuzzy)
            return True

        except:
            return False

    ######################## Functions: #########################
    def is_nan(x):
        return (x is np.nan or x != x)

    def print(text):
        if logToScreen:
            print(text)

    def normalize(lst):
        minval=min(lst)
        maxval=max(lst)
        return [(float(i)-minval)/(maxval-minval) for i in lst]

    def powerfit(xs, ys):
        S_x2_y = 0.0
        S_y_lny = 0.0
        S_x_y = 0.0
        S_x_y_lny = 0.0
        S_y = 0.0
        for (x,y) in zip(xs, ys):
            S_x2_y += x * x * y
            S_y_lny += y * np.log(y)
            S_x_y += x * y
            S_x_y_lny += x * y * np.log(y)
            S_y += y
        #end
        a = (S_x2_y * S_y_lny - S_x_y * S_x_y_lny) / (S_y * S_x2_y - S_x_y * S_x_y)
        b = (S_y * S_x_y_lny - S_x_y * S_y_lny) / (S_y * S_x2_y - S_x_y * S_x_y)
        return np.exp(a), b

    def linear_fit(x, y):
        res = linregress(x, y)
        return res

    def sum_lin_minus_exp(xs, y_lin,  y_exp):
        sum = 0
        for i, x in enumerate(xs):
            distance = abs((y_lin.intercept + y_lin.slope*x) - y_exp[i])
            # Normalizing across the y-axis
            distance = distance / (max(y_exp) - min(y_exp))
            sum += distance

        # normalize by number of points
        dist = sum/len(xs)
        return dist

    def find_gap_in_data(ys):
        miny = min(ys)
        maxy = max (ys)
        interval = (maxy - miny) / (50 - 1)
        bucket_count = [0 for i in range(50)]

        for y in ys:
            position = int(y/interval)
            bucket_count[position] += 1

        o_count = 0
        for elem in bucket_count:
            if elem == 0:
                o_count += 1
        dist = o_count / 50
        return dist

    def logisticfit(x,y):
        popt, pcov = curve_fit(sigmoid,xdata= x,ydata= y)
        return popt

    def sigmoid(x, x0, k):
        y = 1 / (1 + np.exp(-k*(x-x0)))
        return y


    def normalize_values(values):
        values=np.array(np.float64(values))
        values=np.array(values).reshape(-1,1)
        scaler = MinMaxScaler()
        scaler.fit(values)
        scaled_values=scaler.transform(values)
        return_values=scaled_values.squeeze()
        return return_values, scaler

    def denormalize_values(values, scaler):
        return scaler.inverse_transform(values.reshape(-1,1)).squeeze()


    def cleanfilename(filename,cleanlist):
        text=filename
        for cleanword in cleanlist:
            text = text.replace(cleanword, '-')
        text=text.replace('-----','-')
        text=text.replace('----','-')
        text=text.replace('---','-')
        text=text.replace('--','-')
        text=text.replace('-','-')
        if text!=filename:
            print("Warning: Provided filename '"+filename+"' is invalid. Set to '"+text+"'")
        return text

    def checkAndGetBool(val):
        if isinstance(val, str):
            return val.lower() in ['true', '1', 't', 'y', 'yes']
        if isinstance(val, (int, float)):
            if val>=1:
                return 1
            else:
                return 0


    ############################ Input handling: #############################
    data=df

    ##### Input handling: logToScreen ###
    logToScreen=checkAndGetBool(logToScreen)
    #### Input handling: maxrows ##############
    if maxrows<=0:
        rownum=df.shape[0]
        maxrows=min(rownum,5000)
        print("Warning: maxrows cannot be <1. Set to " + str(rownum))

    ##### Input handling: df ###################
    if df is None:
        print("Warning: Dataframe is empty. Returning None")
        return None
    if df.shape[0]==0:
        print("Warning: Dataframe is empty. Returning None")
        return None
    if df.shape[0]>maxrows:
        data=data.sample(maxrows)

    ##### Input handling: fractionToAnalyze ###
    ##### ensure that the fraction is between 0 and 1:
    if fractionToAnalyze>1:
        print("Warning: Fraction to analyze was specified > 1 ! Set to 1" )
        fractionToAnalyze=1
    if fractionToAnalyze<=0:
        print("Warning: Fraction to analyze was specified <= 0 ! Set to 0.1" )
        fractionToAnalyze=0.1

    ##### Input handling: outputPath ###
    outputPath=outputPath.replace("//","/")
    outputPath=outputPath.replace("\\","/")
    outputPath=outputPath.replace("\\\\","/")
    if outputPath[:1]==".":
        if outputPath[0:2]!="./":
            outputPath="./"+outputPath[1:]
    elif outputPath[0:1]=="/":
            outputPath="."+outputPath
    else:
        outputPath="./"+outputPath


    ####ensure that output folders exist:
    cwd = pathlib.Path.cwd() #used to handle paths in different environments
    if outputPlots:
         #ensure output folder is existing:
        if not (cwd/outputPath).is_dir():
            pathlib.Path("./"+outputPath+"/").mkdir(parents=True, exist_ok=True)
        if not (cwd/outputPath/"plots").is_dir():
            pathlib.Path("./"+outputPath+"/plots").mkdir(parents=True, exist_ok=True)
    if outputTable:
       #ensure output folder is existing:
        if not (cwd/outputPath).is_dir():
                pathlib.Path("./"+outputPath+"/").mkdir(parents=True, exist_ok=True)
        if not (cwd/outputPath/"table").is_dir():
                pathlib.Path("./"+outputPath+"/table").mkdir(parents=True, exist_ok=True)


    ##### Input handling: outputTable ###
    outputTable=checkAndGetBool(outputTable)

    ##### Input handling: outputPlots ###
    outputPlots=checkAndGetBool(outputPlots)

    ##### Input handling: outputTablename ###
    if outputTablename is None:
        print("Warning: No valid outputTableName provided. As it is used for file naming, 'testtable' is used.")
        outputTablename="testtable"

    ##### Input handling: columnFillThreshold ###
    if columnFillThreshold>1:
        print("Warning: columnFillThreshold was specified > 1 ! Set to 1" )
        columnFillThreshold=1
    if columnFillThreshold<=0:
        print("Warning: columnFillThreshold was specified <= 0 ! Set to 0.1" )
        columnFillThreshold=0.1

    ############################ Code: #############################

    #### specify the columns of the returntable:
    resulttablecols =[
        'tablename','datecol','valuecol','analyzed_fraction',
        'delta_y','exp_B','exp_A','exp_r_squared',
        'logistic_k','logistic_r_s',
        'distance_exp_lin','n_gap_columns'
    ]

    print ('######## Starting ' + outputTablename + ' ###### fraction '+str(fractionToAnalyze)+'########')
    print ('##########################################')

    #### specify which column types are regarded as numeric (and get analyzed):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    #### specify which chars to remove in the filenames:
    cleanlist=['(',')',' ',':','[',']','#','/']


    #### initialize an empty data frame with the columns for returning the results:
    df_result = pd.DataFrame(columns=resulttablecols)

    #### Loop through all datetimecols:
    for datetimecol in data.columns:
        df_temp=data.copy()


        ### identify date columns and append as converted column (datetime) and converted2 (numeric - for correlation)

        if any(df_temp[datetimecol].astype(str).str.len()<4):
            print("####### The datetime column "+ datetimecol+": has less than 4 chars --> not used as datetime column.")
            continue
        try:
            df_temp['date_converted']=pd.to_datetime(df_temp[datetimecol])
            #df_temp['date_days_since_first_date']=mdates.date2num(df_temp['date_converted'])
            dates_converted=mdates.date2num(df_temp['date_converted'])
            df_temp['date_days_since_first_date']=dates_converted-min(dates_converted)
        except:
            if debug:
                print("####### Datetime column "+ datetimecol+": conversion error!") #raise
                continue
            else:
                print("####### Datetime column "+ datetimecol+": conversion error!")
                continue
        if any(df_temp['date_converted'] < '1700-01-01'):  #1700 because of pandas nanoseconds limitation
            print("####### The datetime column "+ datetimecol+" had values before 1700-01-01 --> not used as date")
            continue
        if datetimecol!='date_converted':
            print("####### The datetime column: "+ datetimecol + " is used as datetimecol!")


        ### filter columns where only 50% of the rows have values:
        df_temp= df_temp.loc[:, df_temp.isnull().mean() < columnFillThreshold]
		
        if 'date_days_since_first_date' not in df_temp.columns:
            continue

        ### Only use the last fraction of the data, indexed / determined by the time column at hand:
        maxval=max(df_temp['date_days_since_first_date']) #use converted time columns to allow for easy multiplication
        minval=min(df_temp['date_days_since_first_date'])
        timepoint_to_cut=(maxval-minval)*(1-fractionToAnalyze)+minval
        df_temp=df_temp[(df_temp['date_days_since_first_date'] > timepoint_to_cut)]

        ###loop through all numeric cols:
        for numericcol in data.select_dtypes(include=numerics).columns:
            if numericcol==datetimecol or numericcol not in df_temp.columns:
                type_4_count += 1
                continue ###skip this column if it is the date col

            # Remove Type 4 errors
            numerical_data = list(df_temp[numericcol])
            if numerical_data and is_date(str(numerical_data[0])):
                type_4_count += 1
                continue

            #### Start computing the values:
            print('##### Computing numeric col: '+ numericcol)
            print('##### Date col: '+datetimecol)

            fitted_exponential=False
            fitted_logistic=False
            fitted_powerlaw=False

            try:  #trying exponential fit
                df_temp=df_temp
                #### zero values will break the algorithm with log calculations.
                #### Therefore, we move all data points slightly above 0.
                #### 'slightly" is defined as min_max_diff/1000000
                if df_temp[numericcol].shape[0]==0:continue #exit if this numeric column has no values
                min_max_diff=max(df_temp[numericcol])-min(df_temp[numericcol])
                min0=False
                if min(df_temp[numericcol])==0:
                       min0=True
                shiftamount=min_max_diff/1000000

                cols_to_keep=['date_converted','date_days_since_first_date',datetimecol,numericcol]
                df_temp_2=df_temp[cols_to_keep].copy()# this df is used for cleaning and afterwards splitted
                #remove all rows with nan or inf:
                df_temp_2=df_temp_2.replace([np.inf, -np.inf], np.nan)
                df_temp_2=df_temp_2.dropna(subset=[numericcol])
                df_temp_2=df_temp_2.sort_values(['date_converted'],ascending=True)

                #get minimum val, and move all points if negative:
                minval=min(df_temp_2[numericcol])

                df_temp_3=df_temp_2
                if minval<=0:
                    df_temp_3[numericcol]=df_temp_3[numericcol]+shiftamount
                    df_temp_3[numericcol]=df_temp_3[numericcol]-minval
                    print('! moved by '+str(minval) +' to correct for negative values')


                x=df_temp_3['date_days_since_first_date']
                print(x)
                y=df_temp_3[numericcol]

                delta_y=max(y)-min(y)


                datetime=df_temp_3['date_converted']#datetime
                if len(y)<100: #minimum 100 cases
                    continue
                #### fit a regression line and get the slope (= exponential factor):
                exp_a,exp_b=powerfit(x,y)
                y_exp=exp_a*np.exp(exp_b*x)

                n_gap_columns = find_gap_in_data(y)
                if n_gap_columns > 0.4:
                    type_1_count += 1
                    continue

                # fit a linear regression
                y_lin = linear_fit(x, y)

                # calculate the distance between linear model and exponential model
                distance_exp_lin = None
                # distance_exp_lin = sum_lin_minus_exp(x, y_lin, y_exp)
                # if distance_exp_lin < 1:
                #     type_3_count += 1
                #     continue

                #compute r²:
                exp_residuals = y-y_exp
                ss_res = np.sum(exp_residuals**2)
                ss_tot = np.sum((y-np.mean(y))**2)
                exp_r_s = 1 - (ss_res / ss_tot)

                if (abs(exp_r_s)<exp_r_s_threshold or abs(exp_r_s)>1 or abs(exp_b)<exp_b_threshold or is_nan(exp_r_s) or is_nan(exp_b)):
                    fitted_exponential=False
                    exp_r_s=np.nan
                    exp_b=np.nan
                    exp_a=np.nan
                    continue
                else:
                    fitted_exponential=True

                # check the slope, if it is negative, we remove it because even
                # if it is deemed exponential it is not of our interest
                if exp_b < 0:
                    type_2_count += 1
                    fitted_exponential = False
            except:
                if debug:
                    raise
                else:
                    print("Unexpected error exp fit:" + str(sys.exc_info()[0]))


            try: #trying logistic fit
                k=np.nan
                x0=np.nan
                logistic_r_s=np.nan


                #### fit a logistic curve:

                x_norm,scaler_x=normalize_values(x)
                y_norm,scaler_y=normalize_values(df_temp_3[numericcol])
                popt=logisticfit(x_norm,y_norm)

                ##get r²:
                y_logistic= sigmoid(x_norm, *popt)
                y_logistic2=denormalize_values(y_logistic,scaler_y)
                residuals = y-y_logistic2
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y-np.mean(y))**2)
                logistic_r_s = 1 - (ss_res / ss_tot)
                x0=mdates.num2date(denormalize_values(popt[0],scaler_x))
                k=popt[1]

                if (abs(logistic_r_s)<logistic_r_s_threshold  or abs(logistic_r_s)>1  or abs(k)<k_threshold or is_nan(logistic_r_s) or is_nan(k)):
                    fitted_exponential=False
                    logistic_r_s=np.nan
                    k=np.nan
                    x0=np.nan
                    continue
                else:
                    fitted_exponential=True

            except:
                if debug:
                    raise
                else:
                    print("Unexpected error exp fit:" + str(sys.exc_info()[0]))
            try:
                if exp_r_s > 0:
                    #append to output table:
                    df_result=df_result.append(pd.Series([
                        outputTablename,datetimecol,numericcol,fractionToAnalyze,
                        delta_y,exp_b,exp_a,exp_r_s,k,logistic_r_s,
                        distance_exp_lin, n_gap_columns],index=resulttablecols),ignore_index=True)

            except Exception as e:
                print(str(e))

            if outputTable:
                df_result.drop(df_result[df_result['valuecol'] == df_result['datecol']].index, inplace=True)
                df_result.to_csv(str(pathlib.Path(outputPath+'/table/'+cleanfilename(outputTablename+"-"
                            +str(int(round(fractionToAnalyze*100,0)))+"perc.csv",cleanlist))))

            ################################### Plotting #########################################

            try: #plotting

                if outputPlots and (fitted_exponential==True or fitted_logistic==True or fitted_powerlaw==True) and exp_r_s > 0:



                    ########### Plotparameters:
                    figsize_plot=(6,12) #inches
                    fontsize_plot=10
                    linestyle='-'#'--'
                    linewidth_plot=2
                    ############################ Plot 1: Logarithmized - Dev ############################
                    fig, (ax1,ax2) = plt.subplots(2,1,figsize=figsize_plot)
                    fig.subplots_adjust(bottom=0.3,left=0.2,hspace = 0.8)
                    ax1.plot(datetime, y,'.', alpha=.3,color='#000000', markersize=5) #plotting the values
                    label_start_y=-0.5
                    if fitted_exponential:
                        expa=round(exp_a,5)
                        expb=round(exp_b,5)
                        ax1.plot(datetime,y_exp,linestyle, linewidth=linewidth_plot,color='#00A287') #plotting the line
                        #plotting additional information:
                        ax1.annotate(f'${expa}^{{{expb}x}}$',xy=(0.1,label_start_y-0.16),xycoords='axes fraction',
                        fontsize=fontsize_plot,color='#00A287' )

                        ax1.annotate("r²="+str(round(exp_r_s,3)), xy=(0.1,label_start_y-0.08),xycoords='axes fraction',
                        fontsize=fontsize_plot,color='#00A287')
                        ax1.annotate("Exponential Fit:", xy=(0.1,label_start_y),xycoords='axes fraction',
                            fontsize=fontsize_plot+2, color='#00A287')

                    if fitted_logistic:
                        # logistic regression plot:


                        ax1.plot(datetime, y_logistic2, color='#C87000', linewidth=2)
                        #plotting additional information:


                        ax1.annotate("k="+str(round(k,3)), xy=(0.6,label_start_y-0.24),xycoords='axes fraction',
                        fontsize=fontsize_plot, color='#C87000')

                        ax1.annotate("x₀="+str(x0), xy=(0.6,label_start_y-0.16),xycoords='axes fraction',
                        fontsize=fontsize_plot, color='#C87000')
                        ax1.annotate("r²="+str(round(r_squared_sigmoid,3)), xy=(0.6,label_start_y-0.08),xycoords='axes fraction',
                        fontsize=fontsize_plot, color='#C87000')
                        ax1.annotate("Logistic Fit:", xy=(0.6,label_start_y),xycoords='axes fraction',
                        fontsize=fontsize_plot+2, color='#C87000')

                    ax1.set_title("Last "+str(100*fractionToAnalyze)+"%", fontsize=19)
                    ax1.set_yscale('log')
                    if min0: #only set min y axis if 0 values are existing:
                        x1,x2,y1,y2 = plt.axis()
                        ax1.axis((x1,x2,shiftamount/10,y2))
                        #plt.annotate("0 (no log)", xy=(-0.005,shiftamount),xycoords=('axes fraction','data'),fontsize=fontsize_plot,color='red',ha='right')
                    ax1.tick_params('x', labelrotation=90)


                ############################ Plot 2: Regular Plot - Dev ############################
                    #fig, ax = plt.subplots(figsize=figsize_plot)
                    #fig.subplots_adjust(bottom=0.3,left=0.2)

                    ax2.plot(datetime, y,'.', alpha=.3,color='#000000', markersize=5) #plotting the values

                    if fitted_exponential:

                        ax2.plot(datetime,y_exp,linestyle, linewidth=linewidth_plot,color='#00A287') #plotting the line



                    if fitted_logistic:
                        # logistic regression plot:
                        y_logistic= sigmoid(x_norm, *popt)
                        y_logistic2=denormalize_values(y_logistic,scaler_y)
                        ax2.plot(datetime, y_logistic2, color='#C87000', linewidth=2)


                    if min0: #only set min y axis if 0 values are existing:
                        x1,x2,y1,y2 = plt.axis()
                        ax2.axis((x1,x2,shiftamount/10,y2))
                        #plt.annotate("0", xy=(-0.005,shiftamount),xycoords=('axes fraction','data'),fontsize=fontsize_plot,color='red',ha='right')

                    ax2.tick_params('x', labelrotation=90)


                    plt.savefig(str(pathlib.Path(outputPath+'/plots/'+cleanfilename(outputTablename+'-'+datetimecol+
                                '-'+numericcol,cleanlist)+'-'+str(int(round(fractionToAnalyze*100,0)))+'perc-nonlog.png')))
                    plt.close('all')

            except:
                if debug:
                    raise
                else:
                    print("Unexpected error:" + str(sys.exc_info()[0]))
            #plt.close('all')

#    print("Type-2 error count: {}, Type-4 error count: {} ".format(type_2_count, type_4_count))
    return(df_result)
