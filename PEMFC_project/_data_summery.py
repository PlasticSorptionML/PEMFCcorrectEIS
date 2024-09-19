import pandas as pd
import numpy as np

def target_summery(y) :
    count = y.value_counts()
    df_count = pd.DataFrame(index=count.index,columns=['counts','proportion'])
    df_count['counts'] = count.values
    df_count['proportion'] = np.divide(count.values,np.sum(count.values))
    return df_count
