import pandas as pd


joints = ["HeadPitch", "HeadYaw", "HipPitch", "HipRoll", "KneePitch",
  "LElbowRoll", "LElbowYaw", "LHand", "LShoulderPitch", "LShoulderRoll", "LWristYaw",
  "RElbowRoll", "RElbowYaw", "RHand", "RShoulderPitch", "RShoulderRoll", "RWristYaw"]

df_data = pd.read_csv("rebl-pepper/data_augmented.csv", usecols=["id"]+joints)
df_label = pd.read_csv("rebl-pepper/labels_augmented.csv", usecols=["id","valence","arousal"])

df_data_grouped = df_data.groupby("id")
for id, df_g in df_data_grouped:
    print(id)
    df_write = df_g[joints]
    filename_write = id + ".csv"
    df_write.to_csv("./train/"+filename_write, index=False, header=None)

with open("./train/column_names.txt", "w") as f:
    f.write(",".join(joints))

