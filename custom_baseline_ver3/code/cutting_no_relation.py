import pandas as pd


def random_cutting_no_relation(csv_path, nth_max=0):  # nth_max = 몇 번째 class를 기준으로 자를지
    origin_df = pd.read_csv(csv_path)
    random_seeds = [42, 1202, 3]

    except_nr_df = origin_df[origin_df.label != "no_relation"]
    nr_df = origin_df[origin_df.label == "no_relation"]
    except_nr_df_count = except_nr_df["label"].value_counts()
    number_of_nr = except_nr_df_count[nth_max]

    for seed in random_seeds:
        nr_random_df = nr_df.sample(n=number_of_nr, random_state=seed)  # fraction of axis items to return.
        custom_df = pd.concat([except_nr_df, nr_random_df])
        custom_df = custom_df.sort_values("id")
        print(f"----------{seed} seed-----------")
        print(custom_df["label"].value_counts())
        print("--------------------------")
        custom_df.to_csv(f"./random_cutting_nr_seedof_{seed}.csv", index=False)


random_cutting_no_relation("./preprocess.csv", 0)

