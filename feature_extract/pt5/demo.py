import glob
import os
import pickle
import re

# 1147
esm2 = pickle.load(open('/home/415/hz_project/MMSMAPlus-master/data/pt5/cafa3/T100900009800.pkl', 'rb'))['pt5']


print(esm2.T[:,:esm2.shape[0]-1].shape)

# string_test ='ABCD'
#
# string_new = " ".join(list(re.sub(r"[UZOB]", "X", string_test)))
#
# print(len(string_new))

# on_save = '/home/415/hz_project/MMSMAPlus-master/data/pt5/cafa3'
# pt_files = glob.glob(os.path.join(on_save, '*.pkl'))
# pt_id = []
# for file_path in pt_files:
#     sample_id = os.path.basename(file_path).split('.')[0]
#     pt_id.append(sample_id)
#
# if str('T100900009800') in pt_id:
#     print(True)


# test = 'AA'
#
# if len(test) > 5:
#     test = test[:5]
#
# print(test)