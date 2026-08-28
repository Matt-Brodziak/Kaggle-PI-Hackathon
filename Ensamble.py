import pandas as pd
from scipy.stats import rankdata
from LR_model import test, test_probsLR
from RF_model import test_probsRF


WEIGHT = 0.6
ID_COL = "anonymised_id"

n_samples = len(test_probsLR)
rank_lr = rankdata(test_probsLR) / n_samples
rank_rf = rankdata(test_probsRF) / n_samples

blend_rank = WEIGHT * rank_lr + (1 - WEIGHT) * rank_rf

submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    "employed_prob": blend_rank,
})

submission.to_csv("Submissions/Ensamble.csv", index=False)
print("\nSaved Ensamble.csv")