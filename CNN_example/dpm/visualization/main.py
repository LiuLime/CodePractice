import sys

sys.path.append("..")

from utils import common, draw
from collate_metric import *


# logger = log.logger()


class Profile:
    def __init__(self):
        """dataset evaluation"""
        pass

    def system_profile(self):
        # source system-target system heatmap, triangle heatmap
        dataset_pivot = data_pool.pivot_table(columns="x_dataset_id",
                                              index="y_dataset_id",
                                              values="x_inchikey.id",
                                              aggfunc="count",
                                              fill_value=0)
        draw.draw_heatmap(dataset_pivot,
                          define_cmap=True,
                          save_title="./output_ML/system_heatmap.png",
                          fig_title="source-target system")

        # system compounds classyfire
        compound_class = data_pool["x_superclass"].value_counts().reset_index()
        draw.draw_barplot(compound_class, "count", "x_superclass", "./output_ML/compound_count.png")

        # pairs rt range,hue = superclass
        draw.draw_scatter(df=data_pool, x="x_rt", y="y_rt",
                          hue="x_superclass",
                          save_path_title="output_ML/system_pairs_range")


# %%


# class _Metrics:
#     def __init__(self, metric_dict: dict):
#         self.MSE = metric_dict["MSE_eval"]
#         self.RMSE = metric_dict["RMSE_eval"]
#         self.MAE = metric_dict["MAE_eval"]
#         self.MedAE = metric_dict["MedAE_eval"]
#         self.R2 = metric_dict["R2_eval"]
#         self.personr = metric_dict["personr_eval"]

# def __getitem__(self, metric: str):
#     metric += "_eval"
#     return self.metric_dict[metric]
# @property
# def AE(self):
#     return "absolute errors"


# class _Meta:
#     def __init__(self, metric_dict):
#         self.kfold = metric_dict["kfold"]
#         self.subgroup = metric_dict["sub_group"]
#         self.bestparms = None


# class Stasis:
#     def __init__(self, metric_dict):
#         self.metrics = _Metrics(metric_dict)
#         self.meta = _Meta(metric_dict)


# %%
# subset_mean_metric =metric_df_mean.groupby(by=["sub_group"]).apply(lambda x: stat.mean(x)).reset_index()
# print(subset_mean_metric.head())
"""select best_parms"""

# metric_df = []
# source_rt = eval_systems["x_rt"]
# for m, metric_list in metric_data.items():
#     for epoch in metric_list:
#         c = calculate(source_rt, epoch["eval_target"], epoch["eval_preds"])
#         epoch["MRE_eval"] = c.MRE()
#     metric_df.extend(metric_list)
# metric_df = pd.DataFrame(metric_df)
# output_ML = metric_df.groupby(['method'], as_index=False).agg({
#                                                             "MSE_eval": metric_mean,
#
#                                                             "RMSE_eval": metric_mean,
#
#                                                             "MAE_eval": metric_mean,
#
#                                                             "MedAE_eval": metric_mean,
#
#                                                             "R2_eval": metric_mean,
#
#                                                             "personr_eval": metric_mean
#                                                             })
#
# common.save_csv(output_ML, "./output_ML/performance.csv")

# %% analyze bad examples by RE


# metric_path = "../model_ML/output_ML/01_240802_train5000_10epoch_evalfull/metric_small_10epoch.json"
# metric_data = common.read_json(metric_path)
# metric_df = []
# source_rt = eval_systems["x_rt"]

# RE_df = []

# draw performance figure
# for m, metric_list in metric_data.items():
#     for idx, epoch in enumerate(metric_list):
#         temp_dict = {}
#         c = calculate(source_rt, epoch["eval_target"], epoch["eval_preds"])
#         RE = c.RE()
#         temp_dict["method"] = m
#         temp_dict["epoch"] = idx
#         temp_dict["RE<=1"] = RE[RE <= 1].shape[0]
#         temp_dict["RE>1<=2"] = RE[(RE > 1) & (RE <= 2)].shape[0]
#         temp_dict["RE>2"] = RE[RE > 2].shape[0]
#         # temp_dict["total"] = len(source_rt)
#         RE_df.append(temp_dict)
# RE_df = pd.DataFrame(RE_df)
# RE_df_epoch0 = RE_df[RE_df["epoch"] == 0]
# draw.draw_stack_barplot(df=RE_df_epoch0[["RE<=1", "RE>1<=2", "RE>2"]],
#                         x=RE_df_epoch0["method"],
#                         labels=["RE<=1", "RE>1<=2", "RE>2"],
#                         stack_num=3,
#                         save_title="./output_ML/stack_barplot_RE.png")


# %%
if __name__ == "__main__":
    config_ml = common.read_yaml("../config.yaml")
    random_split_path = config_ml["path"]["random_split_dataset"]
    data_pool_path = "../../collate_data/train.csv"
    data_pool = common.read_csv(data_pool_path)
    train_descriptor = common.read_csv("../../collate_data/train_descriptor.csv")
    # metric_path = "../model_ML/output_ML/RF_metric.json"
    # metric_data = common.read_json(metric_path)
    model_list = config_ml["train"]["method"]

    _, subgroup_mean_metric_df = batch_subgroup_mean_metric(model_list)

    # 保存为了缩短读取时间
    data = subgroup_mean_metric_df[subgroup_mean_metric_df[("algorithm", "Unnamed: 14_level_1")] == "GB"]
    draw.draw_line_plot(data, x=("sub_group", "Unnamed: 1_level_1"), y=("MedAE_eval", "mean"),
                        # hue=("algorithm","Unnamed: 14_level_1"),
                        column_clip_on=("MedAE_eval", "mean"),
                        clip_on_ymin=-0.2,
                        clip_on_ymax=2)
    # generator = load_features_labels()
    # for idx, (g, f, l) in enumerate(generator):
    #     print(idx)
    #     print(g, f.shape, l.shape)
    #     break
    # test = f"../model_ML/output_ML/RF_metric.json"
    # metric_data = common.read_json(test)
    # s = Statistic(metric_data)
    # preds_slice = s.pred_slice
    # for _, rows in preds_slice.iterrows():
    #     test = rows["y_test"][0]
    #     break
    # s.collate_preds()
