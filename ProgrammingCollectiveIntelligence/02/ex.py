import cf
import os

filepath=r'C:\Users\Administrator\Desktop\draft\data'+os.sep
d1=pd.read_table(filepath+'u.data',names=['userid','itemid','score','scoretime'])
d2=pd.read_table(filepath+'u.item',sep='|',header=None).iloc[:,0:3].rename(columns={0:'itemid',1:'movie',2:'release_date'})
d3=pd.read_table(filepath+'u.user',sep='|',names=['userid','age','gender','occupation','zipcode'])

data=pd.merge(d1,d2,on='itemid').loc[:,['userid','movie','score']]
prefs=data[~data[['userid','movie']].duplicated()].set_index(['userid','movie']).unstack()
prefs.columns=prefs.columns.droplevel()
