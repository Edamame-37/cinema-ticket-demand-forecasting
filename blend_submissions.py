import pandas as pd

# Load submission dari Exp07 (Tree Models - RMSE sangat baik: 469, MASE: 0.44)
sub_exp07 = pd.read_csv('experiments/Exp07_RatioTargeting_FeaturePruning/submission.csv')

# Load submission dari Exp08 (Deep Learning - RMSE buruk: 1152, MASE Elite: 0.42)
sub_exp08 = pd.read_csv('experiments/Exp08_DeepLearningEmbeddings/submission.csv')

# Lakukan Blending (50% Tree, 50% Deep Learning)
# Ini akan menarik turun RMSE yang meledak dari DL, sambil tetap menjaga ketajaman MASE-nya.
sub_blend = sub_exp07.copy()
sub_blend['total_ticket'] = (sub_exp07['total_ticket'] * 0.5) + (sub_exp08['total_ticket'] * 0.5)

# Pembulatan ke bilangan bulat (tiket tidak bisa desimal)
sub_blend['total_ticket'] = sub_blend['total_ticket'].round().astype(int)

# Simpan ke dalam folder baru
import os
os.makedirs('experiments/Exp09_HybridBlending', exist_ok=True)
sub_blend.to_csv('experiments/Exp09_HybridBlending/submission.csv', index=False)

print("Kombinasi Submission (Exp07 + Exp08) berhasil dibuat di: experiments/Exp09_HybridBlending/submission.csv")
