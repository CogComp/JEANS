import pickle
import os
import time
from Trie import Trie, TrieNode
import os
cwd = os.getcwd()
print(cwd)

# build dbpedia entity name
lang1 = "fr"
lang2 = "en"
id = "2"
id2lang = {"1":lang1, "2":lang2}
dir = f"../reference/JAPE/data/dbp15k/{lang1}_{lang2}/0_3_pro"
dump_path = f"{dir}/id_url_ent_name_{id}"
overwrite = False

if overwrite is True:
    fout = open(dump_path, "w")
    for line in open(os.path.join(dir,f"ent_ids_{id}"), "r", encoding="utf-8"):
        line = line.strip().split("\t")
        eid, url = line[0], line[1]
        ent =  line[1].split("/")[-1]
        ent_name = ent.split("(")[0].replace("_"," ").strip()
        ent = "dbpedia/" + ent
        fout.write(f"{eid}\t{url}\t{ent}\t{ent_name}\n")
    fout.close()

# dump_path
trie = Trie()
all_ents = set([])
for line in open(dump_path,"r",encoding="utf-8"):
    # lower
    line = line.rstrip('\n').split('\t')
    trie.insert(line[3], line[2])
    all_ents.add(line[2])
print("Built trie.")

out_file = f"../data/wiki_db/{lang1}_{lang2}/{id2lang[id]}.txt"
corpora_file = f"../data/wiki/{id2lang[id]}.txt"
num = 0
found = 0
found_entity = set([])
with open(out_file, "w") as f:
    t0 = time.time()
    for line in open(corpora_file, "r", encoding="utf-8"):
        hit4line = 0
        num += 1
        # lower
        # line = line.lower()
        line = line.rstrip('\n').split(' ')
        index = 0
        w_line = ""
        l = len(line)
        while index < l:
            cur = index
            cur_node = trie.root
            rst = None
            while True:
                if cur < l:
                    next = cur_node.nodes.get(line[cur])
                    if next != None:
                        cur_node = next
                        cur += 1
                    else:
                        rst = cur_node.sub
                        break
                else:
                    rst = cur_node.sub
                    break
            if rst == None:
                w_line += line[index] + " "
                index += 1
            else:
                hit4line = 1
                w_line += rst + " "
                # print(w_line)
                index = cur
                found += 1
                found_entity.add(rst)
        if hit4line == 1:
            f.write(w_line.rstrip(' ') + '\n')
        if num % 10000 == 0:
            print("Scanned",num,"lines.",(time.time() - t0) ,"seconds.", "#found: ", found)

print("Scanned",num,"lines.",(time.time() - t0) ,"seconds.", "#found: ", found)
with open(f"../data/wiki_db/{id2lang[id]}_ent.txt", "w") as f:
    f.write("\n".join(found_entity))

