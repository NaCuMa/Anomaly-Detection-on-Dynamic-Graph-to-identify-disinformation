import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def preprocess(data_name, ext_emb):
  u_list, i_list, ts_list, label_list = [], [], [], []
  e_feat_l = []
  n_feat_l = []
  idx_list = []

  #-------------------------------------------------------------------------------------
  if ext_emb !=0:
    with open(data_name) as f:
      s = next(f)
      for idx, line in enumerate(f):
        e = line.strip().split(',')

        if int(float(e[0])) == -1:
          pass
        else:
          e_feat = int(e[4])
          e_feat_l.append(e_feat)
  
    e_feat_l_ext = []
    dict_e_feat = {}
    e_idx = set(e_feat_l)
    for i in range(ext_emb):
      with open('./data/raw_data/'+data_name[7:-4]+f'_e_feat_{i}.csv') as f:
        s = next(f)
        for idx, line in enumerate(f):
          e = line.strip().split('"')
          if int(e[0][:-1]) in e_idx:
            dict_e_feat[int(e[0][:-1])] = len(dict_e_feat)
            e = e[1][1:-1].split(',')
            e_feat_l_ext.append(np.array([float(x) for x in e]))
          if len(dict_e_feat)==len(e_idx):
            break
      if len(dict_e_feat)==len(e_idx):
        print("Les éléments nécessaires à la constitution du jeu de données ont tous été lus.")
        break
      print("Le fichier ", i+1, "/", ext_emb, "a été lu.")
    print(len(np.unique(e_feat_l)))
    print(len(dict_e_feat.keys()))
    #pd.DataFrame(dict_e_feat.keys()).to_csv('./data/inter.csv')
  #-------------------------------------------------------------------------------------

  with open(data_name) as f:
    s = next(f)
    for idx, line in enumerate(f):
      e = line.strip().split(',')

      if int(float(e[0])) == -1:
        n_feat = np.array([float(x) for x in e[1:]])
        n_feat_l.append(n_feat)
      else:
        u = int(float(e[0]))
        i = int(float(e[1]))

        ts = float(e[2])
        label = float(e[3])  # int(e[3])
        if ext_emb == 0:
          e_feat = np.array([float(x) for x in e[4:]])
        
        u_list.append(u)
        i_list.append(i)
        ts_list.append(ts)
        label_list.append(label)

        if ext_emb == 0:
          idx_list.append(idx-len(n_feat_l))
          e_feat_l.append(e_feat)
        else:
          idx_list.append(dict_e_feat[e_feat_l[idx-len(n_feat_l)]])
  
  if ext_emb != 0:
    e_feat_l = e_feat_l_ext

  return pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list}), np.array(e_feat_l), np.array(n_feat_l)


def reindex(df, bipartite=True):
  new_df = df.copy()
  if bipartite:
    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    upper_u = df.u.max() + 1
    new_i = df.i + upper_u

    new_df.i = new_i
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
  else:
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

  return new_df


def run(data_name, bipartite=False, ext_emb=0):
  Path(f"data/{data_name}").mkdir(parents=True, exist_ok=True)
  #PATH = './data/{}/{}.csv'.format(data_name)
  #OUT_DF = './data/{}/ml_{}.csv'.format(data_name)
  #OUT_FEAT = './data/{}/ml_{}.npy'.format(data_name)
  #OUT_NODE_FEAT = './data/{}/ml_{}_node.npy'.format(data_name)
  PATH = f'./data/{data_name}.csv'
  OUT_DF = f'./data/{data_name}/ml_{data_name}.csv'
  OUT_FEAT = f'./data/{data_name}/ml_{data_name}.npy'
  OUT_NODE_FEAT = f'./data/{data_name}/ml_{data_name}_node.npy'

  df, e_feat, n_feat = preprocess(PATH, ext_emb)
  
  #df, e_feat = preprocess(PATH)
  new_df = reindex(df, bipartite)
  e_empty = np.zeros(e_feat.shape[1])[np.newaxis, :]
  e_feat = np.vstack([e_empty, e_feat])

  if len(n_feat) == 0:
    max_idx = max(new_df.u.max(), new_df.i.max())
    n_feat = np.zeros((max_idx + 1, 172))
  else:
    n_empty = np.zeros(n_feat.shape[1])[np.newaxis, :]
    n_feat = np.vstack([n_empty, n_feat])
  
  new_df.to_csv(OUT_DF)
  np.save(OUT_FEAT, e_feat)
  np.save(OUT_NODE_FEAT, n_feat)

parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')
parser.add_argument('--ext_emb', type=int, help='Number of the extra files containing the embeddings',default=0)

args = parser.parse_args()

run(args.data, bipartite=args.bipartite, ext_emb = args.ext_emb)