#french
DIR = "/home/swj0419/joint_emb"

fr_ents = set([])
with open(f"{DIR}/reference/JAPE/data/dbp15k/fr_en/0_3/ent_ids_1", "r") as f:
    for line in f:
        ent = line.strip().split("/")[-1].lower()
        fr_ents.add(ent)

with open(f"{DIR}/data/wk3l_60k/entity_names/fr_60k_short_entity.txt", "r") as f:
    for line in f:
        ent = line.strip().replace(" ","_")
        fr_ents.add(ent)

with open(f"{DIR}/data/all_ents_stats/fr", "w") as f:
    for item in fr_ents:
        f.write("%s\n" % item.lower())

zh_ents = set([])
with open(f"{DIR}/reference/JAPE/data/dbp15k/zh_en/0_3/ent_ids_1", "r") as f:
    for line in f:
        ent = line.strip().split("/")[-1].lower()
        zh_ents.add(ent)

with open(f"{DIR}/data/all_ents_stats/zh", "w") as f:
    for item in zh_ents:
        f.write("%s\n" % item.lower())


ja_ents = set([])
with open(f"{DIR}/reference/JAPE/data/dbp15k/ja_en/0_3/ent_ids_1", "r") as f:
    for line in f:
        ent = line.strip().split("/")[-1].lower()
        ja_ents.add(ent)

with open(f"{DIR}/data/all_ents_stats/ja", "w") as f:
    for item in ja_ents:
        f.write("%s\n" % item.lower())


de_ents = set([])
with open(f"{DIR}/data/wk3l_60k/entity_names/de_60k_short_entity.txt", "r") as f:
    for line in f:
        ent = line.strip().replace(" ","_")
        de_ents.add(ent)

with open(f"{DIR}/data/all_ents_stats/de", "w") as f:
    for item in de_ents:
        f.write("%s\n" % item.lower())



en_ents = set([])
with open(f"{DIR}/reference/JAPE/data/dbp15k/fr_en/0_3/ent_ids_2", "r") as f:
    for line in f:
        ent = line.strip().split("/")[-1].lower()
        en_ents.add(ent)

with open(f"{DIR}/reference/JAPE/data/dbp15k/ja_en/0_3/ent_ids_2", "r") as f:
    for line in f:
        ent = line.strip().split("/")[-1].lower()
        en_ents.add(ent)

with open(f"{DIR}/reference/JAPE/data/dbp15k/zh_en/0_3/ent_ids_2", "r") as f:
    for line in f:
        ent = line.strip().split("/")[-1].lower()
        en_ents.add(ent)


with open(f"{DIR}/data/wk3l_60k/entity_names/en_60k_short_entity.txt", "r") as f:
    for line in f:
        ent = line.strip().replace(" ","_")
        en_ents.add(ent)


with open(f"{DIR}/data/all_ents_stats/en", "w") as f:
    for item in en_ents:
        f.write("%s\n" % item.lower())


