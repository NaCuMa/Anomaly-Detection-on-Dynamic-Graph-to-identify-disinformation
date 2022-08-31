import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from utils.utils import RandEdgeSampler


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
  # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
  # negatives for validation / test set)
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_ap, val_auc = [], []
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    for k in range(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      sources_batch = data.sources[s_idx:e_idx]
      destinations_batch = data.destinations[s_idx:e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

      size = len(sources_batch)
      _, negative_samples = negative_edge_sampler.sample(size)

      pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                            negative_samples, timestamps_batch,
                                                            edge_idxs_batch, n_neighbors)

      pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
      true_label = np.concatenate([np.ones(size), np.zeros(size)])

      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))

  return np.mean(val_ap), np.mean(val_auc)


def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
  pred_prob = np.zeros(len(data.sources))
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance / batch_size)

  with torch.no_grad():
    decoder.eval()
    tgn.eval()
    for k in range(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)

      sources_batch = data.sources[s_idx: e_idx]
      destinations_batch = data.destinations[s_idx: e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = edge_idxs[s_idx: e_idx]

      source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                   destinations_batch,
                                                                                   destinations_batch,
                                                                                   timestamps_batch,
                                                                                   edge_idxs_batch,
                                                                                   n_neighbors)
      pred_prob_batch = decoder(source_embedding).sigmoid()
      pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

  auc_roc = roc_auc_score(data.labels, pred_prob)
  return auc_roc

def eval_embedding(model, data, n_neighbors, time_step, batch_size=200, anomaly = False, neg = False, grads = False):
  rand_sampler = RandEdgeSampler(data.sources, data.destinations, seed=0)
  #with torch.no_grad():
    #model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
  TEST_BATCH_SIZE = batch_size
  num_test_instance = len(data.sources)
  num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

  embeddings = []
  memory= []
  if anomaly:
    anomaly_edges = []
  if neg:
    neg_edges = []
    neg_edges_prob = []
  if grads:
    grads_edges = []

  nodes = np.arange(model.n_nodes)
  step = 0

  embeddings.append((step, 0, model.embedding_module.compute_embedding(memory=model.memory.get_memory(list(range(model.n_nodes))),
                                            source_nodes=nodes,
                                            timestamps=np.full((1,nodes.shape[0]), max(model.memory.get_last_update(list(range(model.n_nodes)))))[0],
                                            n_layers=model.n_layers,
                                            n_neighbors=model.n_neighbors).detach().numpy()))
  memory.append((step, 0, model.memory.get_memory(list(range(model.n_nodes))).detach().numpy()))
  step += 1


    #----------------------- Reliability module -------------------------------

  criterion = torch.nn.BCELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
  pos_label = torch.ones(1, dtype=torch.float)
  layers=[x for x in model.parameters()]
  for l in range(2,14):
    layers[l].requires_grad = False

    #------------------------------------------------------

  for k in range(num_test_batch):
    s_idx = k * TEST_BATCH_SIZE
    e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
    sources_batch = data.sources[s_idx:e_idx]
    destinations_batch = data.destinations[s_idx:e_idx]
    timestamps_batch = data.timestamps[s_idx:e_idx]
    edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

    """model.compute_temporal_embeddings_exp(sources_batch, destinations_batch,
                                                            timestamps_batch, edge_idxs_batch, n_neighbors)"""
    size = len(sources_batch)
    _, negatives_batch = rand_sampler.sample(size)
    pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                            negatives_batch, timestamps_batch,
                                                            edge_idxs_batch, n_neighbors)
    if anomaly:
      anomaly_edges_scores = 1 - pos_prob.detach().numpy()
      anomaly_edges.extend(anomaly_edges_scores.tolist())
      
    if neg:
      neg_edges.extend(negatives_batch.tolist())
      neg_edges_prob.extend(neg_prob.detach().numpy().tolist())
      
      #------------------------------------------------------
    if grads:
      for i in range(e_idx-s_idx):
        optimizer.zero_grad()
        #print(pos_prob.squeeze()[i])
        #print(pos_label[0])
        loss = criterion(pos_prob.squeeze()[i], pos_label[0])
        loss.backward(retain_graph=True)
        gradients = 0
        gradients += np.linalg.norm(layers[0].grad.view(-1).detach().numpy())
        gradients += np.linalg.norm(layers[1].grad.view(-1).detach().numpy())
        for l in range(10,24):
          if layers[l].grad != None:
            gradients += np.linalg.norm(layers[l].grad.view(-1).detach().numpy())
        #gradients = torch.cat(gradients)
        grads_edges.append(gradients)
        print(i)
      model.memory.detach_memory()
      print(k, '/', num_test_batch)
      #------------------------------------------------------
        
    if step*time_step <= data.timestamps[s_idx]:# and data.timestamps[s_idx] < (step+1)*time_step:
        
      time = max(model.memory.get_last_update(list(range(model.n_nodes))))
      embeddings.append((step, time, model.embedding_module.compute_embedding(memory=model.memory.get_memory(list(range(model.n_nodes))),
                                            source_nodes=nodes, timestamps=np.full(nodes.shape[0], time),
                                            n_layers=model.n_layers, n_neighbors=model.n_neighbors).detach().numpy()))
      memory.append((step, time, model.memory.get_memory(list(range(model.n_nodes))).detach().numpy()))
      print("Etape ", step," sur ", int(data.timestamps[-1]/time_step))
      step += 1
    
  last_memory, _ = model.get_updated_memory(list(range(model.n_nodes)), model.memory.messages)

  embeddings.append((step, data.timestamps[-1], model.embedding_module.compute_embedding(memory=last_memory, source_nodes=nodes,
                                            timestamps=np.full(nodes.shape[0], data.timestamps[-1]),
                                            n_layers=model.n_layers, n_neighbors=model.n_neighbors).detach().numpy()))
  memory.append((step, data.timestamps[-1], last_memory.detach().numpy()))

  output = [embeddings, memory]
  if anomaly:
    output.append(anomaly_edges)
  if neg:
    output.append(neg_edges)
    output.append(neg_edges_prob)
  if grads:
    output.append(grads_edges)
  return output

def eval_prob(model, node, batch_size=200):
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = model.n_nodes-1
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    part = []
    nodes = np.arange(num_test_instance+1)
    #time = max(model.memory.get_last_update(nodes).detach().numpy())

    for k in range(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE + 1
      e_idx = min(num_test_instance + 1, s_idx + TEST_BATCH_SIZE)
      batch = nodes[s_idx:e_idx]
      part.extend(model.compute_edge_affinity_wum(np.full(len(batch), node), batch).detach().numpy().tolist())

    return np.array(part).reshape(1,num_test_instance)[0]