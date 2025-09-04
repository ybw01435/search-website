
from flask import Flask, render_template, request
import jieba
import os
from collections import Counter, defaultdict
import numpy as np
import pickle
from itertools import combinations

class Posting(object):
    special_docid = -1
    def __init__(self, docid, tf=0):
        self.docid = docid
        self.tf = tf
    def __repr__(self):
        return "<docid: %d, tf: %d>" % (self.docid, self.tf)

def load_stopwords(file_path): #加载停用词表
    stopwords=set()
    if os.path.exists(file_path):
        with open(file_path,'r',encoding='utf-8')as f:
            for line in f:
                stopword=line.strip()
                if stopword:
                    stopwords.add(stopword)
    return  stopwords
stopwords=load_stopwords("C:/Users/y/OneDrive/Desktop/停用词.txt")

#添加自定义词典
jieba.load_userdict("C:\\Users\\y\\OneDrive\\Desktop\\自定义词典.txt")
#倒排索引构建样例，找出所有正文信息
collections=[file for file in os.listdir('C:\\Users\\y\\crawled_htmls') if os.path.splitext(file)[1]=='.txt']
#print(collections)

#依次打开每个保存正文信息的文本文件，分词，构造term_docid_pairs，计算文本长度
log_func=np.vectorize(lambda x:1.0+np.log10(x) if x>0 else 0.0)
def read_file_term_docid_pairs_and_doc_length(file_path_1,file_path_2):
    if os.path.exists(file_path_1) and os.path.exists(file_path_2):
        with open(file_path_1, 'rb') as fin:
            term_docid_pairs = pickle.load(fin)
        with open(file_path_2, 'rb') as fin:
            doc_length = pickle.load(fin)
    else:
        term_docid_pairs=[]
        doc_length=[]
        for docid,filename in enumerate(collections):
            with open(os.path.join("C:\\Users\\y\\crawled_htmls",filename),encoding='utf-8')as f:
        
                terms=[term for term in jieba.cut_for_search(f.read()) if len(term.strip())>1 and term.strip() not in stopwords]
                #构造term_docid_pairs
                for term in terms:
                    term_docid_pairs.append((term,docid))
                #统计tf
                term_counts=np.array(list(Counter(terms).values()))
                log_tf=log_func(term_counts)
                doc_length.append(np.sqrt(np.sum(log_tf**2)))
        with open(file_path_1, 'wb') as fout:
            pickle.dump(term_docid_pairs, fout)
        with open(file_path_2,'wb')as fout:
            pickle.dump(doc_length,fout)

    return (term_docid_pairs,doc_length)
file_path_1='./term_docid_pairs_2_5.pkl'
file_path_2='./doc_length_2_5.pkl'
term_docid_pairs,doc_length=read_file_term_docid_pairs_and_doc_length(file_path_1,file_path_2)
# print(term_docid_pairs)
#print(doc_length)

for docid,filename in enumerate(collections):
    with open(os.path.join("C:\\Users\\y\\crawled_htmls",filename),encoding='utf-8')as f:
        collections[docid]=f.readline().strip()

term_docid_pairs=sorted(term_docid_pairs)
#构造倒排索引  

def read_file_inverted_index(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fin:
            inverted_index = pickle.load(fin)
        
    else:
        inverted_index=defaultdict(lambda:[Posting(Posting.special_docid,0)])

        for term,docid in term_docid_pairs:
            postings_list=inverted_index[term]
            if docid!=postings_list[-1].docid:  #postings_list的值发生改变，字典的值也会发生改变
                postings_list.append(Posting(docid,1))
            else:
                postings_list[-1].tf+=1
        inverted_index=dict(inverted_index)
        with open(file_path, 'wb') as fout:
                pickle.dump(inverted_index, fout)
    return inverted_index
file_path='./inverted_index_2_5.pkl'
inverted_index=read_file_inverted_index(file_path) 

#计算idf
total_docs=len(collections)
idf={}
for term in inverted_index:
    doc_count=len(inverted_index[term])-1
    idf[term]=np.log10((total_docs + 1) / (doc_count + 1))

#辅助函数，安全获取postings_list（处理查询词不存在的情况）
def get_postings_list(inverted_index,query_term):
    try:
        return inverted_index[query_term][1:]
    except KeyError:
        return []
#获取包含特定术语的文档ID集合
def get_docids_for_term(inverted_index,term):
    postings=get_postings_list(inverted_index,term)
    return set(posting.docid for posting in postings)

# 计算余弦相似度
def calculate_cosine_similarity(query_terms,docid,inverted_index,doc_length,idf):
    score = 0.0
    for term, count in query_terms.items():
        postings = get_postings_list(inverted_index, term)
        tf = next((p.tf for p in postings if p.docid == docid), 0)
        if tf == 0:
            continue
        w_tq = log_func(count) * idf.get(term, 0)
        w_td = log_func(tf) * idf.get(term, 0)
        score += w_td * w_tq
    
    if doc_length[docid] == 0:
        return 0.0
    return score / doc_length[docid]

#改进后的多级布尔and查询，包含每一级查询的所有可能结果
def advanced_query(query, inverted_index=inverted_index, collections=collections, idf=idf, doc_length=doc_length, k=20):
    query_terms = Counter(term for term in jieba.cut_for_search(query) if len(term.strip()) > 1 and term.strip() not in stopwords)
    if not query_terms:
        return []

    # 按idf从高到低排序查询词（如5,4,3,2,1）
    sorted_terms = sorted(query_terms.keys(), key=lambda x: -idf.get(x, 0))
    n = len(sorted_terms)
    query_terms_counter = dict(query_terms)
    
    results = []
    used_docids = set()

    # 逐级查询：第1级缺失0个词（全包含），第2级缺失1个词，...，第n级缺失n-1个词
    for missing_count in range(0, n):
        # 当前级别需要保留的词数 = 总词数 - 缺失词数
        keep_count = n - missing_count
        if keep_count <= 0:
            break

        # 生成所有“保留keep_count个词”的组合
        for current_terms in combinations(sorted_terms, keep_count):
            #对当前组合执行布尔and查询
            docid_sets = [get_docids_for_term(inverted_index, term) for term in current_terms]
            if any(not s for s in docid_sets):
                continue 
            # 计算集合的交集（布尔AND）
            common_docs = set.intersection(*docid_sets)
            # 排除已使用过的文档
            common_docs = common_docs - used_docids
            if not common_docs:
                continue

            # 计算相似度并排序
            scored_docs = []
            for docid in common_docs:
                score = calculate_cosine_similarity(query_terms_counter, docid,
                                                  inverted_index, doc_length, idf)
                scored_docs.append((docid, score))
            scored_docs.sort(key=lambda x: -x[1])

            #添加结果，直到满足k值
            for docid, score in scored_docs:
                if len(results) >= k:
                    break
                results.append((docid, score))
                used_docids.add(docid)
        
            if len(results) >= k:
                break

        if len(results) >= k:
                break
    return [collections[docid] for docid, score in results[:k]]

def evaluate(query, k=20):
    return advanced_query(query, k=k)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods = ['GET'])
def query():
    key = request.args.get('key')

    results=evaluate(key)

    return render_template('results.html', key=key, results=results)

app.run(host='0.0.0.0', port=12345, debug=True)
