import os

write_align = False
DIR = "/home/swj0419/joint_emb"
lang1 = "zh"
lang2 = "en"
olddir = f"{DIR}/reference/JAPE/data/dbp15k/{lang1}_{lang2}/0_3"
newdir = f"{DIR}/reference/JAPE/data/dbp15k/{lang1}_{lang2}/0_3_pro"
if not os.path.exists(newdir):
    os.mkdir(newdir)

id = "1"
id2lang = {"1":lang1, "2":lang2}

id2newid_ent_list = []
id2newid_rel_list = []
for id, lang in id2lang.items():
    # read id2ents1
    id2ents = {}
    id2newid_ent = {}
    with open(f"{olddir}/ent_ids_{id}") as f:
        for i, line in enumerate(f):
            line = line.strip().split()
            id2ents[line[0]] = line[1]
            id2newid_ent[line[0]] = str(i)
    ents2id = {v:k for k,v in id2ents.items()}

    # read id2rels
    id2rels = {}
    id2newid_rel = {}
    with open(f"{olddir}/rel_ids_{id}") as f:
        for i, line in enumerate(f):
            line = line.strip().split()
            id2rels[line[0]] = line[1]
            id2newid_rel[line[0]] = str(i)
    rels2id = {v:k for k,v in id2rels.items()}

    # read triples
    triples = []
    with open(f"{olddir}/triples_{id}") as f:
        for line in f:
            triples.append(tuple(line.strip().split()))

    # write ent_ids
    dump_ent_ids = f"{newdir}/ent_ids_{id}"
    with open(dump_ent_ids, "w") as f:
        for oid, ent in id2ents.items():
            nid = id2newid_ent[oid]
            f.write(f"{nid}\t{ent}\n")

    # write rel_ids
    dump_rel_ids = f"{newdir}/rel_ids_{id}"
    with open(dump_rel_ids, "w") as f:
        for oid, ent in id2rels.items():
            nid = id2newid_rel[oid]
            f.write(f"{nid}\t{ent}\n")


    # write triples
    dump_triple = f"{newdir}/triples_{id}"
    with open(dump_triple, "w") as f:
        for t in triples:
             nhid = id2newid_ent[t[0]]
             nrid = id2newid_rel[t[1]]
             ntid = id2newid_ent[t[2]]
             f.write(f"{nhid}\t{nrid}\t{ntid}\n")
    id2newid_ent_list.append(id2newid_ent)
    id2newid_rel_list.append(id2newid_rel)

# read align
sup_ent_pairs = []
with open(f"{olddir}/sup_ent_ids") as f:
    for line in f:
        sup_ent_pairs.append(line.strip().split())

sup_rel_pairs = []
with open(f"{olddir}/sup_rel_ids") as f:
    for line in f:
        sup_rel_pairs.append(line.strip().split())

ref_ent_pairs = []
with open(f"{olddir}/ref_ent_ids") as f:
    for line in f:
        ref_ent_pairs.append(line.strip().split())


# write
dump_sup_ent_pairs = f"{newdir}/sup_ent_ids"
with open(dump_sup_ent_pairs, "w") as f:
    for t in sup_ent_pairs:
        nid_1 = id2newid_ent_list[0][t[0]]
        nid_2 = id2newid_ent_list[1][t[1]]
        f.write(f"{nid_1}\t{nid_2}\n")

dump_sup_rel_pairs = f"{newdir}/sup_rel_ids"
with open(dump_sup_rel_pairs, "w") as f:
    for t in sup_rel_pairs:
        nid_1 = id2newid_rel_list[0][t[0]]
        nid_2 = id2newid_rel_list[1][t[1]]
        f.write(f"{nid_1}\t{nid_2}\n")

dump_ref_ent_pairs = f"{newdir}/ref_ent_ids"
with open(dump_ref_ent_pairs, "w") as f:
    for t in ref_ent_pairs:
        nid_1 = id2newid_ent_list[0][t[0]]
        nid_2 = id2newid_ent_list[1][t[1]]
        f.write(f"{nid_1}\t{nid_2}\n")





