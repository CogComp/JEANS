import pickle
import os
import time
from Trie import Trie, TrieNode
import os
from collections import OrderedDict

cwd = os.getcwd()
print(cwd)

# build dbpedia entity name
dir = "../data/all_ents_stats"
id = "5"
id2lang = {"1":"fr", "2":"en", "3": "ja", "4": "zh", "5":"de"}
lang = id2lang[id]
dump_path = f"{dir}/{lang}"
overwrite = False

all_ents = set([])
# dump_path
trie = Trie()
for line in open(dump_path,"r",encoding="utf-8"):
    # lower
    line = line.rstrip('\n').split('\t')
    surface = line[0].replace("_"," ")
    label = "dbpedia/"+line[0]
    all_ents.add(label)
    trie.insert(surface, label)
print("Built trie.")


out_file = f"../data/wiki_db/{id2lang[id]}.txt"
corpora_file = f"../data/wiki/{id2lang[id]}.txt"
num = 0
found = 0
found_entity = set([])
entity_hit_count = {k:0 for k in all_ents}
with open(out_file, "w") as f:
    t0 = time.time()
    for line in open(corpora_file, "r", encoding="utf-8"):
        num += 1
        # lower
        line = line.lower()
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
                w_line += rst + " "
                # print(w_line)
                index = cur
                found += 1
                found_entity.add(rst)
                entity_hit_count[rst] += 1
        f.write(w_line.rstrip(' ') + '\n')
        if num % 10000 == 0:
            print("Scanned",num,"lines.",(time.time() - t0) ,"seconds.", "#found: ", found)

print("Scanned",num,"lines.",(time.time() - t0) ,"seconds.", "#found: ", found, "# total: ", len(all_ents), "hit: ", len(found_entity)/len(all_ents))
with open(f"../data/wiki_db/{id2lang[id]}_ent.txt", "w") as f:
    f.write("\n".join(found_entity))

sorted(entity_hit_count.items(), key=lambda x: x[1], reverse=True)

# write stats to file
with open(f"../data/wiki_db/{id2lang[id]}_stats.txt","w") as f:
    f.write(f"#total match: {found}, # unique entities: {len(all_ents)}, # found entities:{len(found_entity)}, hit: {found/len(all_ents)*100}%, average occurrence: {len(found_entity)/len(all_ents)}\n")
    for key in sorted(entity_hit_count, key = entity_hit_count.get, reverse=True):
        v = entity_hit_count[key]
        f.write(f"{key}: {v}\n")







