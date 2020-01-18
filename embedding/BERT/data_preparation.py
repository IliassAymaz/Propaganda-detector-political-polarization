import os.path
import glob
import pandas as pd

path_prefix = "./"
file_list = glob.glob(os.path.join(path_prefix + "data/datasets/train-articles", "*.txt"))
articles_content, articles_id = ([], [])
for filename in file_list:
    with open(filename, "r", encoding="utf-8") as f:
        articles_content.append(f.readlines())
        articles_id.append(os.path.basename(filename).split(".")[0])
articles = dict(zip(articles_id, articles_content))

a_id, sentences, spans, start_offset, end_offset =  ([], [], [], [], [])
for article_id in articles_id:
    article_content = articles[article_id]
    sentences.extend(article_content)

    train_label_filename = article_id + ".task1-SI.labels"
    try:
        data = pd.read_csv(path_prefix + "data/datasets/train-labels-task1-span-identification/" + train_label_filename, header = None, sep = "\t")
        data = data.sort_values(data.columns[1], ascending = True)
        
        content_index = 0
        for one_sentence in article_content:
#            print(content_index)
            add = 0
            a_id.append(article_id)
            for _, row in data.iterrows():
                if row[1] > content_index + len(one_sentence):
                    break
                elif row[1] >= content_index and row[2] < content_index + len(one_sentence):
#                    print(row[1]-content_index)
#                    print(one_sentence[row[1]-content_index:row[2]-content_index])
                    start_offset.append(row[1]-content_index)
                    end_offset.append(row[2]-content_index)
                    spans.append(one_sentence[row[1]-content_index:row[2]-content_index])
                    add = 1
                    break
                else:
                    continue
            
            content_index = content_index + len(one_sentence)
            if add == 0:
                spans.append(None)
                start_offset.append(-1)
                end_offset.append(-1)
    except:
        a_id.extend([article_id] * len(article_content))
        spans.extend([""] * len(article_content))
        start_offset.extend([-1] * len(article_content))
        end_offset.extend([-1] * len(article_content))
        continue

ss_table = pd.DataFrame(list(zip(a_id, sentences, spans, start_offset, end_offset)), columns =["Article", "Sentence", "Span", "Start", "End"])
ss_table = ss_table[ss_table["Sentence"] != "\n"]
ss_table = ss_table[ss_table["Sentence"] != "\ufeff\n"]

ss_table.to_csv("train_table.csv", index = False, encoding = "utf-8")
