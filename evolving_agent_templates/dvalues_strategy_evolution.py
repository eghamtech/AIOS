#start_of_genes_definitions
#key=dvalue_delta;  type=random_float;  from=0.0001;  to=0.1;  step=0.00001
#key=dvalue_scale;  type=random_float;  from=1;  to=3;  step=0.01
#key=tp_mult;  type=random_float;  from=1;  to=3;  step=0.01
#key=max_pos;  type=random_int;  from=1;  to=20;  step=1
#key=daily_max_pos;  type=random_int;  from=1;  to=20;  step=1
#key=days_max_pos;  type=random_int;  from=1;  to=5;  step=1
#key=sign_changes_window;  type=random_int;  from=1;  to=100;  step=1
#key=sign_changes_max;  type=random_int;  from=-100;  to=100;  step=1
#key=sign_changes_min;  type=random_int;  from=-100;  to=100;  step=1
#key=spread;  type=random_from_set;  set=0.25
#key=multiplier;  type=random_from_set;  set=5
#key=qty;  type=random_from_set;  set=1
#key=price_decimals;  type=random_from_set;  set=2
#key=init_balance;  type=random_from_set;  set=100000
#key=criteria;  type=random_from_set;  set='total_pnl'
#end_of_genes_definitions

cDATE = 0
cOPEN = 1
cHIGH = 2
cLOW = 3
cCLOSE = 4
cIS_LAST_ROW = 5
cDVALUE = 6
cDVALUE_SIGN = 7
cDVALUE_LAST = 8

class cls_ev_agent_{id}:
    import pandas as pd
    import datetime as dt
    import math
    import numpy
    
    import warnings
    warnings.filterwarnings("ignore")
    
    result_id = {id}
    target_definition = "{field_to_predict}"
    
    p_dvalue_delta = {dvalue_delta}
    p_spread = {spread}
    p_multiplier = {multiplier}
    p_max_pos = {max_pos}
    p_daily_max_pos = {daily_max_pos}
    p_days_max_pos = {days_max_pos}
    p_sign_changes_window = {sign_changes_window}
    p_sign_changes_max = {sign_changes_max}
    p_sign_changes_min = {sign_changes_min}
    p_qty = {qty}
    p_tp_mult = {tp_mult}
    p_dvalue_scale = {dvalue_scale}
    p_price_decimals = {price_decimals}
    p_init_balance = {init_balance}
    p_criteria = {criteria} # one of: 'total_pnl', 'profit_factor', 'sharpe_ratio', 'drawdown'
    
    def __init__(self):
        return
    
    def apply(self, df_add):
        # this method is called by AIOS when additional data is supplied and needs to be predicted on
        # df_add shouldn't contain columns with text values - only numeric
        return
    
    def run(self, mode):
        # this is main method called by AIOS with supplied DNA Genes to process data
        #global df
        #if 'df' not in globals():
        #    df = self.pd.read_csv(workdir + trainfile)
        
        
        self.load_data()
        self.fill_dvalues()
        self.main_loop()
        self.get_figures()
        self.get_fitness()
        
        print ("fitness=" + str(self.fitness))
    
    def load_data(self):
        print ("loading data...")
        self.df = self.pd.read_csv(workdir + trainfile, usecols=["date", "open", "high", "low", "close", "is_last_row"])
        print ("nrows=", len(self.df))
        self.df["dvalue"] = float('nan')
        self.df["dvalue_sign"] = float('nan')
        self.df["dvalue_last"] = float('nan')
        self.asim = self.df.values
        print ("data loaded ok...")
        
    def fill_dvalues(self):
        print ("start fill dvalues...")
        row = self.asim[len(self.asim)-1]

        row[cDVALUE] = row[cCLOSE]
        last_dvalue = row[cCLOSE]
        is_new_datastructure_started = False

        cnt = 0

        for i in range(len(self.asim)-2, -1, -1):
            row = self.asim[i]
            if is_new_datastructure_started:
                last_dvalue = row[cOPEN]
                is_new_datastructure_started = False


            dvalue_up = last_dvalue*(1+self.p_dvalue_delta)
            dvalue_down = last_dvalue*(1-self.p_dvalue_delta)

            if row[cHIGH]>=dvalue_up:
                row[cDVALUE] = dvalue_up
                row[cDVALUE_SIGN] = 1
                row[cDVALUE_LAST] = dvalue_up
                last_dvalue = dvalue_up
            elif row[cLOW]<=dvalue_down:
                row[cDVALUE] = dvalue_down
                row[cDVALUE_SIGN] = -1
                row[cDVALUE_LAST] = dvalue_down
                last_dvalue = dvalue_down
            else:
                #row[cDVALUE] = float('nan')
                #row[cDVALUE_SIGN] = float('nan')
                row[cDVALUE_LAST] = last_dvalue


            if row[cIS_LAST_ROW]!=0:
                is_new_datastructure_started = True
            cnt+=1
            if cnt>=100000:
                print(i)
                cnt = 0
                
    def main_loop(self):
        aPos = []
        aSign = []
        realized_pnl = 0
        positions_closed = 0

        self.dayProfit = {}
        self.dayCount = {}
        self.dayPositionsCount = {}
        self.results = []
        self.max_drawdown = 0

        block_size = 100000
        total_processed = 0
        
        cnt = 0

        print ("start main loop...")
        
        for i in range(len(self.asim)-1, -1, -1):
            row = self.asim[i]

            if row[cOPEN] < row[cLOW]:
                row[cOPEN] = row[cLOW]
            if row[cCLOSE] < row[cLOW]:
                row[cCLOSE] = row[cLOW]
            if row[cLOW]<0.9*row[cHIGH]:
                #print(row)
                row[cOPEN] = row[cHIGH]
                row[cLOW] = row[cHIGH]
                row[cCLOSE] = row[cHIGH]

            bid = row[cOPEN]
            ask = bid + self.p_spread

            max_grid_price = 0
            min_grid_price = 1e6

            nearest_below = 0
            nearest_above = 1e6

            count_below = 0
            count_above = 0

            unreal_pnl = 0

            todays_trades_count = 0
            today = row[cDATE][0:10]

            if len(aPos)>0:
                unreal_pnl_several = 0
                if len(aPos)>1:
                    for pos in aPos:
                        unreal_pnl_several += pos['qty'] * self.p_multiplier * (bid-pos['open_price'])

                count_trades = len(aPos)

                for j in range(len(aPos)-1, -1, -1):
                    pos = aPos[j]

                    condition_close = (i>0 and row[cIS_LAST_ROW]!=0) or row[cHIGH]>=pos['tp_price']
                    if condition_close:
                        close_price = pos['tp_price']
                        if row[cIS_LAST_ROW]!=0:
                            close_price = bid

                        realized_pnl += pos['qty'] * self.p_multiplier * (close_price-pos['open_price'])
                        positions_closed+=1
                        del aPos[j]
                    else:
                        if pos['open_price']>max_grid_price:
                            max_grid_price = pos['open_price']
                        if pos['open_price']<min_grid_price:
                            min_grid_price = pos['open_price']

                        if pos['open_price']>=ask:
                            count_above+=1
                            if pos['open_price']<nearest_above:
                                nearest_above = pos['open_price']
                        else:
                            count_below+=1
                            if pos['open_price']>nearest_below:
                                nearest_below = pos['open_price']

                        unreal_pnl += pos['qty'] * self.p_multiplier * (bid-pos['open_price'])

                        d1 = self.dt.datetime.strptime(str(pos['date'])[0:10], "%Y-%m-%d")
                        d2 = self.dt.datetime.strptime(today, "%Y-%m-%d")
                        date_diff = (d2-d1).days
                        if date_diff<self.p_days_max_pos:
                            todays_trades_count+=1


            if not self.math.isnan(row[cDVALUE_SIGN]):
                aSign.append(row[cDVALUE_SIGN])

            #get sum of dvalue_sign over last sign_changes_window Bars
            latest_sum_dsign = sum(aSign[-self.p_sign_changes_window:])

            dtx = today
            d1 = self.dt.datetime.strptime(dtx, "%Y-%m-%d")
            if (d1.weekday()==6):
                #move sunday's amounts to prev friday
                dtx = str(d1 - self.dt.timedelta(days=2))[0:10]


            if dtx not in self.dayProfit:
                self.dayProfit[dtx] = 0
            if dtx not in self.dayCount:
                self.dayCount[dtx] = 0
            if dtx not in self.dayPositionsCount:
                self.dayPositionsCount[dtx] = 0

            if row[cIS_LAST_ROW]==0 and latest_sum_dsign<=self.p_sign_changes_max and latest_sum_dsign>=self.p_sign_changes_min:
                #check for new positions conditions
                if len(aPos)<self.p_max_pos:
                    if len(aPos)==0:
                        aPos.append({
                            'date': row[cDATE],
                            'qty': self.p_qty,
                            'open_price': ask,
                            'tp_price': bid + round(self.p_dvalue_delta * bid * self.p_tp_mult, self.p_price_decimals)
                        })
                        self.dayPositionsCount[dtx]+=1
                    else:
                        delta = round(bid * self.p_dvalue_delta, self.p_price_decimals)
                        curr_delta = self.p_dvalue_delta * self.math.pow(self.p_dvalue_scale, count_above - 1)
                        net_delta = round(bid * curr_delta, self.p_price_decimals)
                        tp_price = round(bid * (1 + curr_delta * self.p_tp_mult), self.p_price_decimals)

                        curr_delta2 = self.p_dvalue_delta * self.math.pow(self.p_dvalue_scale, count_above)
                        net_delta2 = round(bid * curr_delta2, self.p_price_decimals)

                        allow_open = False

                        if row[cLOW]+self.p_spread<=min_grid_price-net_delta:
                            allow_open = True
                        elif row[cHIGH]+self.p_spread>=max_grid_price+delta:
                            allow_open = True
                        elif ask<=nearest_above-net_delta and ask>=nearest_below+net_delta2:
                            allow_open = True

                        if (allow_open and todays_trades_count<self.p_daily_max_pos):
                            aPos.append({
                                'date': row[cDATE],
                                'qty': self.p_qty,
                                'open_price': ask,
                                'tp_price': tp_price
                            })
                            self.dayPositionsCount[dtx]+=1

            self.results.append({
                'realized_pnl': realized_pnl,
                'unreal_pnl': unreal_pnl,
                'total_pnl': realized_pnl + unreal_pnl,
                'positions_closed': positions_closed,
                'positions_open': len(aPos),
                'latest_sum_dsign': latest_sum_dsign
            })
            total_processed+=1

            if (-unreal_pnl>self.max_drawdown):
                self.max_drawdown = -unreal_pnl


            n_profit_bar = 0
            if len(self.results)==1:
                n_profit_bar = self.results[0]['total_pnl']
            else:
                n_profit_bar = self.results[len(self.results)-1]['total_pnl']-self.results[len(self.results)-2]['total_pnl']

            self.dayProfit[dtx] += n_profit_bar
            self.dayCount[dtx] += 1

            cnt+=1
            if cnt>=block_size:
                print(i)
                cnt = 0

        print("Done")

    def get_figures(self):
        aProfit = []
        for key in self.dayProfit:
            aProfit.append(self.dayProfit[key]/max(self.p_init_balance, 1)*100)


        days_has_pos = 0
        days_total = 0

        for key in self.dayPositionsCount:
            if self.dayPositionsCount[key]>0:
                days_has_pos+=1
            days_total+=1

        regularity = days_has_pos / days_total

        avg = sum(aProfit)/max(1, len(aProfit))
        sharpe_ratio = 0
        if len(aProfit)>=2:
            sharpe_ratio = avg*252 / (self.numpy.std(aProfit)*self.math.sqrt(252))

        if len(self.results)==0:
            self.res = {'realized_pnl': 0, 'unreal_pnl': 0, 'total_pnl': 0, 'positions_closed': 0, 'positions_open': 0, 'latest_sum_dsign': 0}
        else:
            self.res = dict(self.results[len(self.results) - 1])

        profit_factor = self.res['total_pnl'] / max(0.01, self.max_drawdown)
        expected_payoff = self.res['total_pnl'] / max(1, self.res['positions_closed'] + self.res['positions_open'])

        self.res['sharpe_ratio'] = sharpe_ratio
        self.res['max_drawdown'] = self.max_drawdown
        self.res['profit_factor'] = profit_factor
        self.res['expected_payoff'] = expected_payoff
        self.res['regularity'] = regularity
        
        print(self.res)
        
    def get_fitness(self):
        self.fitness = 0
        if self.p_criteria=="total_pnl":
            self.fitness = -self.res['total_pnl']
        elif self.p_criteria=="profit_factor":
            self.fitness = -self.res['profit_factor'] * self.res['regularity']
        elif self.p_criteria=="sharpe_ratio":
            self.fitness = -self.res['sharpe_ratio'] * self.res['regularity']
        elif self.p_criteria=="drawdown":
            if self.res['max_drawdown']<=0:
                self.fitness = self.p_init_balance
            else:
                self.fitness = self.res['max_drawdown']/(self.res['realized_pnl'] or 0.001)/(self.res['regularity'] or 0.001)
        
ev_agent_{id} = cls_ev_agent_{id}()
