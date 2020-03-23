import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

def drawdown(return_series:pd.Series):
    """
    Input : Return Series
    Input Type : pd.Series
    Output :  Padas Dataframe
                
            Starting wealth value = 1000;    
            
            wealth_values | previous_peaks | drawback
                          |                |
                          |                |
                          |                |
                          |                |                      
    
    """
    wealth_values = 100*(1+return_series).cumprod()
    peak_values = wealth_values.cummax()
    drawdown = (wealth_values-peak_values)/peak_values
    
    return pd.DataFrame({"wealth_values": wealth_values,
                        "previous_peaks":peak_values,
                        "drawdown":drawdown})

def get_ffme_returns():
    """
    Load the Fama French Dataset for the returns of the Top and Botton Deciles by Market Cap
    """
    
    me_m = pd.read_csv("data_set/Portfolios_Formed_on_ME_monthly_EW.csv", na_values = 99.99,
                      index_col = 0, header = 0)
    rets = me_m[["Lo 10", "Hi 10"]]
    rets = rets/100
    rets.columns = ["small_cap", "large_cap"]
    rets.index = pd.to_datetime(rets.index, format = "%Y%m").to_period("M")
    
    return rets



def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund Index
    """
    
    hfi = pd.read_csv("data_set/edhec-hedgefundindices.csv", na_values = 99.99,
                      index_col = 0, header = 0, parse_dates = True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period("M")
    
    return hfi


def get_ind_returns():
    
    ind = pd.read_csv("data_set/ind30_m_vw_rets.csv", header = 0, index_col = 0 , parse_dates = True)/100
    ind.index = pd.to_datetime(ind.index, format = "%Y%m").to_period("M")
    ind.columns = ind.columns.str.strip()
    
    return ind

def get_ind_size():
    
    ind = pd.read_csv("data_set/ind30_m_size.csv", header = 0, index_col = 0 , parse_dates = True)
    ind.index = pd.to_datetime(ind.index, format = "%Y%m").to_period("M")
    ind.columns = ind.columns.str.strip()
    
    return ind

def get_ind_nfirms():
    
    ind = pd.read_csv("data_set/ind30_m_nfirms.csv", header = 0, index_col = 0 , parse_dates = True)
    ind.index = pd.to_datetime(ind.index, format = "%Y%m").to_period("M")
    ind.columns = ind.columns.str.strip()
    
    return ind

def get_total_ind_returns():
    ind_returns = get_ind_returns()
    ind_nfirms = get_ind_nfirms()
    ind_avg_size = get_ind_size()
    
    ind_market_cap = ind_nfirms * ind_avg_size
    ind_total_market_cap = ind_market_cap.sum(axis =1)
    
    ind_weights = ind_market_cap.divide(ind_total_market_cap, axis = "rows")
    
    ind_total_returns = (ind_returns*ind_weights).sum(axis = 1)

    return pd.DataFrame(ind_total_returns, columns = ["Returns"]);

def semideviation(r):
    """
    Returns Semideviation of return series, i.e deviation in the negative side
    """
    is_negative = r<0  
    return r[is_negative].std(ddof=0)


def skewness(r):
    
    """
    Return Kurtosis of the series
    Alternative to scipy.skew()
    """
    
    demeaned = (r - r.mean())
    sigma  = r.std(ddof = 0)
    exp = (demeaned**3).mean()
    
    return exp/sigma**3


def kurtosis(r):
    """
    Return the Kurtosis
    Alternative to scipy.kurtosis()
    """
    
    demeaned = (r-r.mean())
    sigma = r.std(ddof =0)
    exp = (demeaned**4).mean()
    
    return exp/sigma**4


def is_normal(r, level = 0.01):
    """
    Applies the Jarque Bera test to determine if a series is normal or not
    By default Test is applied at 1% of cofidence
    Returs True if the hypothesis of normally is accepted with 1% confidence, else False
    """
    
    result = scipy.stats.jarque_bera(r)
    
    return result[1] >= level 

def var_historic(r, level = 5):
    """
    Returns Historic Value at risk at a specified level
    i.e Returns the number such that "level" percent of the returns
    fall below that number, and (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.apply(var_historic, args = (level,))
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be Series of DataFrame")

        
from scipy.stats import norm

def var_gaussian(r, level):
    """
    Returns the parametric gaussian VaR of a Series of DataFrame
    """
    #Compute Z score assuming it is gaussian
    z = norm.ppf(level/100)
    return -(r.mean() + z*r.std(ddof=0))


def annualized_rets(r:pd.Series, periods_per_year):
    """
    Inputs: r:pd.Series or pd.DataFrame
            periods_per_year = frequency of return in a year
                If data has daily return value then periods_per_year = 365 or 252
                If data has monthly return value then periods_per_year = 12            
    Output: Returns annualized return
    """
    
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods) - 1;

def annualized_vol(r:pd.Series, periods_per_year):
    """
    Inputs: r:pd.Series or pd.DataFrame
            periods_per_year = frequency of return in a year
                If data has daily return value then periods_per_year = 365 or 252
                If data has monthly return value then periods_per_year = 12            
    Output: Returns annualized Volatility
    """
    
    return r.std()*(periods_per_year**0.5)

def sharpe_ratio(r:pd.Series, risk_free_rate, periods_per_year):
    """
    Input = risk_free_rate -> Annual risk free rate should be provided
            periods_per_year = frequency of return in a year
                If data has daily return value then periods_per_year = 365 or 252
                If data has monthly return value then periods_per_year = 12                                
    Output: Returns Sharpe Ratio = (Annualized Return - Risk Free Rate)/Volatility
    """
    #Converting annual risk_free_rate to per period
    risk_free_rate_period = (1+risk_free_rate)**(1/periods_per_year) - 1
    excess_rate = r - risk_free_rate_period 
    annualized_return = annualized_rets(excess_rate, periods_per_year)
    annualizez_volatility = annualized_vol(r, periods_per_year)
    
    return annualized_return/annualizez_volatility


def portfolio_return(weights, returns):
    """
    Weights -> Returns
    """
    return np.dot(weights.T, returns)
    
    
def portfolio_volatility(weights, cov_mat):
    """
    Weights -> Volatility
    """
    
    return (np.dot(weights.T, cov_mat).dot(weights))**0.5


def plot_ef2(n_points, expected_returns, cov_mat):
    
    """
    Plot 2 Asset Effecient Frontier
    """
    
    if expected_returns.shape[0] != 2:
        raise ValueError("plot_ef2 can only plot 2 asset frontier")
    weights = [np.array([w,1-w]) for w in np.linspace(start = 0, stop = 1, num = n_points, endpoint= True)]
    portfolio_rets = [portfolio_return(weights=w, returns= expected_returns) for w in weights]
    portfolio_vol = [portfolio_volatility(weights=w, cov_mat = cov_mat) for w in weights]
    
    ef = pd.DataFrame({"Portfolio Returns": portfolio_rets, "Portfolio Volatility": portfolio_vol})
    
    return ef.plot.scatter( x = "Portfolio Volatility", y = "Portfolio Returns", style = "-", figsize = (12, 6))
    
    
from scipy.optimize import minimize

def minimize_vol(target_return, expected_returns, cov_mat):
    """
    target -> Weights
    """
    
    n = expected_returns.shape[0]
    
    initial_guess = np.repeat(1/n,n)
    
    bounds = ((0.0, 1.0),)*n    
    return_is_target = {
        "type":"eq",
        "args":(expected_returns,),
        "fun": lambda weights, expected_returns: target_return - portfolio_return(weights=weights, returns=expected_returns)
    }
    weights_sum_to_1 = {
        "type":"eq",
        "fun": lambda weights: np.sum(weights) - 1
    }
    
    results = minimize(portfolio_volatility, initial_guess, 
                       args = (cov_mat,), method = "SLSQP",
                       options = {"disp":False},
                       constraints = (return_is_target, weights_sum_to_1),
                       bounds = bounds
                      )
    return results.x    
    


def optimal_weights(n_points, expected_returns, cov_mat):
    
    min_ret = expected_returns.min()
    max_ret = expected_returns.max()  
    
    target_returns = np.linspace(start = min_ret, stop = max_ret, num = n_points)
    
    weights = [minimize_vol(target_return= t, expected_returns= expected_returns, cov_mat= cov_mat) for t in target_returns] 
     
    return weights


def plot_ef(n_points, expected_returns, cov_mat, risk_free_rate = 0.1, show_cml = False, show_ew = False, show_gmv = False):
    
    """
    Plot N Asset Effecient Frontier
    """
    
    columns = expected_returns.index
    
    weights = optimal_weights(n_points = n_points, expected_returns = expected_returns, cov_mat = cov_mat)
    portfolio_rets = [portfolio_return(weights=w, returns= expected_returns) for w in weights]
    portfolio_vol = [portfolio_volatility(weights=w, cov_mat = cov_mat) for w in weights]
    
    data = {cols:[weights[row][index] for row in range(n_points)] for index, cols in enumerate(columns) }
    data.update({"Portfolio Returns": portfolio_rets, "Portfolio Volatility": portfolio_vol})
    
    ef = pd.DataFrame(data)
    
    fig = plt.figure(figsize = (12,6))
    ax = fig.add_subplot(111)
    
    ef.plot.scatter(x = "Portfolio Volatility", y = "Portfolio Returns", color = "red", marker = "o", label = "Effecient Frontier", ax = ax)
    
    if show_ew:
        n = expected_returns.shape[0]
        weights_ew = np.repeat(1/n,n)
        rets_ew = portfolio_return(weights = weights_ew, returns = expected_returns)
        vol_ew = portfolio_volatility(weights = weights_ew, cov_mat = cov_mat)
        
        ax.plot(vol_ew, rets_ew, color = "goldenrod", marker = "o", markersize = 12, label = "Equivalent Weight Portfolio")
        
    if show_gmv:
        n = expected_returns.shape[0]
        er = np.repeat(1, n)
        weights_gmv = msr(risk_free_rate = 0.1, expected_returns = er, cov_mat = cov_mat)
        rets_gmv = portfolio_return(weights = weights_gmv, returns = expected_returns)
        vol_gmv = portfolio_volatility(weights = weights_gmv, cov_mat = cov_mat)
        
        ax.plot(vol_gmv, rets_gmv, color = "darkblue", marker = "x", markersize = 12, label = "Global Minimum Volatality")
                
    if show_cml:
        
        ax.set_xlim(left = 0)
        weights_msr = msr(risk_free_rate = risk_free_rate, expected_returns = expected_returns, cov_mat = cov_mat)
        rets_msr = portfolio_return(weights = weights_msr, returns = expected_returns)
        vol_msr = portfolio_volatility(weights = weights_msr, cov_mat = cov_mat)
        
        x_cml = [0, vol_msr]
        y_cml = [risk_free_rate, rets_msr]
        
        ax.plot(x_cml, y_cml, linestyle = "dashed", color = "green", marker = "o", markersize = 10, label = "Capital Market Line")
     
    ax.legend()   
        
    return ef, ax

def msr(risk_free_rate, expected_returns, cov_mat):
    """
    weights for max sharpe ratio
    risk_free_rate, COV, Expected return -> Weights
    """
    
    n = expected_returns.shape[0]
    
    initial_guess = np.repeat(1/n,n)
    
    bounds = ((0.0, 1.0),)*n    

    weights_sum_to_1 = {
        "type":"eq",
        "fun": lambda weights: np.sum(weights) - 1
    }
    
    def neg_sharpe_ratio(weights, risk_free_rate, expected_returns, cov_mat ):
        """
        Returns the negative of the sharpe ratio
        """
        
        r = portfolio_return(weights=weights, returns=expected_returns)
        volatility = portfolio_volatility(weights=weights, cov_mat = cov_mat)
        
        return -(r - risk_free_rate)/volatility
        
        
    
    results = minimize(neg_sharpe_ratio, initial_guess, 
                       args = (risk_free_rate, expected_returns, cov_mat,), method = "SLSQP",
                       options = {"disp":False},
                       constraints = ( weights_sum_to_1),
                       bounds = bounds
                      )
    return results.x 

def panel_to_pandas(series:pd.core.panel.Panel, index_name = None):
    is_first_instance = True;
    r = series.copy()
    for item in r.items:
        if is_first_instance:
            df1 = r[item]
            if index_name:
                df1[index_name] = item
            is_first_instance = False
        else:
            df = r[item]
            if index_name:
                df[index_name] = item
            df1 = df1.append(df)
    
    if index_name:
        df1.reset_index(inplace = True)
        df1 = df1.set_index([index_name, "index"])
    
    return df1;


def get_summary_stats(rets, risk_free_rate ,periods_per_year):
    
    guassian_var5 = rets.apply(lambda r: var_gaussian(r, level = 5))
    max_drawdown = rets.apply(lambda r: drawdown(r).drawdown.min())
    semi_deviation_ = rets.apply(lambda r: semideviation(r))
    skewness_ = rets.apply(lambda r: skewness(r))
    kutosis_ = rets.apply(lambda r: kurtosis(r))
    annualized_rets_ = rets.apply(lambda r: annualized_rets(r, periods_per_year = periods_per_year))
    annualized_vol_ = rets.apply(lambda r: annualized_vol(r, periods_per_year = periods_per_year))
    sharpe_ratio_ = rets.apply(lambda r: sharpe_ratio(r, risk_free_rate = risk_free_rate, periods_per_year = periods_per_year))
    
    return pd.DataFrame({"Skewness":skewness_,
                 "Kurtosis":kutosis_,
                 "Annualized Returns":annualized_rets_,
                 "Annualized Volatility":annualized_vol_,
                 "Sharpe Ratio":sharpe_ratio_,
                 "Maximum Drawdown":max_drawdown,
                 "VAR at 5%":guassian_var5});
                              
                              
def gbm(n_years = 10, n_scenarios =1000, period_per_year = 12, mu = 0.03, sigma = 0.10):
    dt = 1/period_per_year
    n_rows = n_years*period_per_year
    return np.random.normal(loc = mu*dt, scale = sigma*(dt)**0.5, size=(n_rows, n_scenarios))


def cppi(risky_returns, floor = 0.8, riskfree_annual_rate = 0.03, m = 3, 
         dynamic_floor = False, max_drawdown = 0.4, verbose = False):
    
    dates = risky_returns.index
    n_steps = len(dates)

    start = 100
    asset_worth, prev_peak = start, start
    risk_free_returns = pd.DataFrame().reindex_like(risky_returns)
    risk_free_returns[:] = riskfree_annual_rate/12

    asset_worth_history = pd.DataFrame().reindex_like(risky_returns)
    risk_weight_history = pd.DataFrame().reindex_like(risky_returns)
    cushion_history = pd.DataFrame().reindex_like(risky_returns)
    floor_value_history = pd.DataFrame().reindex_like(risky_returns)

    for step in range(n_steps):
        
        if dynamic_floor:
            prev_peak = np.maximum(asset_worth, prev_peak)
            floor_value = prev_peak*floor
        else: 
            floor_value = start*floor
        cushion = (asset_worth - floor_value)/asset_worth

        risky_weight = m*cushion
        risky_weight = np.minimum(1, risky_weight)
        risky_weight = np.maximum(0, risky_weight)
        risk_free_weight = 1 - risky_weight

        risky_alloc = risky_weight*asset_worth
        risk_free_alloc = risk_free_weight*asset_worth

        if verbose:
            display("Asset Worth:\n", asset_worth)
            display("Cusion  = ", cushion)
            display("Risky weight  = ", risky_weight)


        asset_worth = risky_alloc*(1+risky_returns.iloc[step]) + risk_free_alloc*(1+risk_free_returns.iloc[step])

        asset_worth_history.iloc[step] = asset_worth
        risk_weight_history.iloc[step] = risky_weight
        cushion_history.iloc[step] = cushion
        floor_value_history.iloc[step] = floor_value
    
    asset_worth_history.iloc[0] = start
    risky_returns_worth = start*(1+risky_returns).cumprod()
        
    return {"Asset Value": asset_worth_history,
           "Floor Value": floor_value_history,
           "Cushion" : cushion_history,
           "Risk Weights": risk_weight_history,
           "No CPPI Asset Value": risky_returns_worth}


def discount(t, r):
    return (1+r)**(-t)

def pv(liabilities, r):
    dates = liabilities.index.values
    discount_ = discount(dates, r)
    return np.dot(discount_, liabilities).sum()

def funding_ratio(asset, liabilities, r):
    return asset/pv(liabilities, r)







    
    
    