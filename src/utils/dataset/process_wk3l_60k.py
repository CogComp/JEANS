
def checkexists_update(en_ents2id, key):
    if key in en_ents2id:
        # print(key)
        pass
    else:
        en_ents2id[key] = len(en_ents2id)
    return en_ents2id

# read all ents
def read_all_ents(en_fr_align_train_file, en_fr_align_test_file, en_triples_file, lang):
    en_ents2id = {}
    # en_rels2id = {}
    with open(en_fr_align_test_file, "r", encoding = "utf-8") as f:
        for line in f:
            line = line.strip().split("@@@")
            if lang == "lang0":
                en_ents2id = checkexists_update(en_ents2id, line[0])
            else:
                en_ents2id = checkexists_update(en_ents2id, line[1])
    with open(en_fr_align_train_file, "r", encoding = "utf-8") as f:
        for line in f:
            line = line.strip().split("@@@")
            if lang == "lang0":
                en_ents2id = checkexists_update(en_ents2id, line[0])
            else:
                en_ents2id = checkexists_update(en_ents2id, line[1])
    with open(en_triples_file, "r", encoding = "utf-8") as f:
        for line in f:
            line = line.strip().split("@@@")
            en_ents2id = checkexists_update(en_ents2id, line[0])
            en_ents2id = checkexists_update(en_ents2id, line[2])
            # en_rels2id = checkexists_update(en_rels2id, line[0])
    en_id2ents = {v:k for k,v in en_ents2id.items()}
    # en_id2rels = {v:k for k,v in en_rels2id.items()}

    return en_ents2id, en_id2ents

def write_ent_ids(en_id2ents, ent_ids_file, id_url_ent_name_file):
    ent_ids_fout = open(ent_ids_file,"w")
    id_url_ent_name_fout = open(id_url_ent_name_file, "w")
    for id, ent in en_id2ents.items():
        ent =  ent.replace(" ","_")
        ent_db = "dbpedia/" + ent
        ent_ids_fout.write(f"{id}\t{ent_db}\n")
        id_url_ent_name_fout.write(f"{id}\t{ent_db}\t{ent_db}\t{ent}\n")
    ent_ids_fout.close()
    id_url_ent_name_fout.close()

def write_rel_ids(en_id2rels, rel_ids_file):
    with open(rel_ids_file, "w") as f:
        for id, ent in en_id2rels.items():
            ent = ent.replace(" ", "_")
            ent_db = "dbpedia/" + ent
            f.write(f"{id}\t{ent_db}\n")


# read train_align ents
def read_align(en_fr_align_train_file, en_ents2id, fr_ents2id):
    en_fr_align_pairs = set([])
    with open(en_fr_align_train_file, "r", encoding = "utf-8") as f:
        for line in f:
            line = line.strip().split("@@@")
            en_fr_align_pairs.add((en_ents2id[line[0]],fr_ents2id[line[1]]))
    return en_fr_align_pairs

def write_ent_pairs(en_fr_pairs, outfile):
    with open(outfile, "w") as f:
        for pair in en_fr_pairs:
            f.write(f"{pair[0]}\t{pair[1]}\n")

def read_all_rels(triples_file):
    rels2id = {}
    with open(triples_file, "r", encoding = "utf-8") as f:
        for line in f:
            line = line.strip().split("@@@")
            rels2id[line[1]] = len(rels2id)
    id2rels = {v:k for k,v in rels2id.items()}
    return rels2id, id2rels

def write_triples(triples_file, triples_out, en_ents2id, en_rels2id):
    with open(triples_out, "w") as fout:
        with open(triples_file, "r", encoding = "utf-8") as f:
            for line in f:
                line = line.strip().split("@@@")
                id1, id2, id3 = en_ents2id[line[0]], en_rels2id[line[1]], en_ents2id[line[2]]
                fout.write(f"{id1}\t{id2}\t{id3}\n")



if __name__ == "__main__":
    out_dir = "/home/swj0419/joint_emb/data/wk3l_60k/fr_en"
    en_ent_file = "/home/swj0419/joint_emb/data/wk3l_60k/entity_names/en_60k_short_entity.txt"
    en_fr_align_file= "/home/swj0419/joint_emb/data/wk3l_60k/alignment/en_fr_60k.csv"
    fr_ent_file = "/home/swj0419/joint_emb/data/wk3l_60k/entity_names/fr_60k_short_entity.txt"
    en_fr_align_train_file = "/home/swj0419/joint_emb/data/wk3l_60k/alignment/en_fr_60k_train25.csv"
    en_fr_align_test_file = "/home/swj0419/joint_emb/data/wk3l_60k/alignment/en_fr_60k_test75.csv"
    en_triples_file = "/home/swj0419/joint_emb/data/wk3l_60k/structure/en_60k.csv"
    fr_triples_file = "/home/swj0419/joint_emb/data/wk3l_60k/structure/fr_60k.csv"

    en_ents2id, en_id2ents = read_all_ents(en_fr_align_train_file, en_fr_align_test_file, en_triples_file, "lang0")
    fr_ents2id, fr_id2ents = read_all_ents(en_fr_align_train_file, en_fr_align_test_file, fr_triples_file, "lang1")
    en_rels2id, en_id2rels = read_all_rels(en_triples_file)
    fr_rels2id, fr_id2rels = read_all_rels(fr_triples_file)

    en_fr_align_pairs_train = read_align(en_fr_align_train_file, en_ents2id, fr_ents2id)
    en_fr_align_pairs_test = read_align(en_fr_align_test_file, en_ents2id, fr_ents2id)
    write_ent_pairs(en_fr_align_pairs_train, f"{out_dir}/sup_ent_ids")
    write_ent_pairs(en_fr_align_pairs_test, f"{out_dir}/ref_ent_ids")
    write_triples(en_triples_file, f"{out_dir}/triples_1", en_ents2id, en_rels2id)
    write_triples(fr_triples_file, f"{out_dir}/triples_2", fr_ents2id, fr_rels2id)
    write_ent_ids(en_id2ents, f"{out_dir}/ent_ids_1", f"{out_dir}/id_url_ent_name_1")
    write_ent_ids(fr_id2ents, f"{out_dir}/ent_ids_2", f"{out_dir}/id_url_ent_name_2")
    write_rel_ids(en_id2rels, f"{out_dir}/rel_ids_1")
    write_rel_ids(fr_id2rels, f"{out_dir}/rel_ids_2")





