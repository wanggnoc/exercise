def cid_from_other_source():
    """
    some drug can not be found in pychem, so I try to find some cid manually.
    the small_molecule.csv is downloaded from http://lincs.hms.harvard.edu/db/sm/
    """
    f = open(folder + "small_molecule.csv", 'r')
    reader = csv.reader(f)
    reader.next()
    cid_dict = {}
    for item in reader:
        name = item[1]
        cid = item[4]
        if not name in cid_dict: 
            cid_dict[name] = str(cid)
    unknow_drug = open(folder + "unknow_drug_by_pychem.csv").readline().split(",")
    drug_cid_dict = {k:v for k,v in cid_dict.iteritems() if k in unknow_drug and not is_not_float([v])}
    return drug_cid_dict

def save_cell_mut_matrix():
    f = open("PANCANCER_Genetic_feature.csv")
    reader = csv.reader(f)
    next(reader)
    features = {}
    cell_dict = {}
    mut_dict = {}
    matrix_list = []
    for item in reader:
        cell_id = item[1]
        mut = item[5]
        is_mutated = int(item[6])
        if mut in mut_dict:
            col = mut_dict[mut]
        else:
            col = len(mut_dict)
            mut_dict[mut] = col
        if cell_id in cell_dict:
            row = cell_dict[cell_id]
        else:
            row = len(cell_dict)
            cell_dict[cell_id] = row
        if is_mutated == 1:
            matrix_list.append((row, col))
    cell_feature = np.zeros((len(cell_dict), len(mut_dict)))
    for item in matrix_list:
        cell_feature[item[0], item[1]] = 1
    return cell_dict, cell_feature

##############
def save_mix_drug_cell_matrix():
    f = open("PANCANCER_IC.csv")
    reader = csv.reader(f)
    next(reader)
    cell_dict, cell_feature = save_cell_mut_matrix()
    drug_dict, drug_smile, smile_graph = load_drug_smile()
    temp_data = []
    bExist = np.zeros((len(drug_dict), len(cell_dict)))
    for item in reader:
        drug = item[0]
        cell = item[3]
        ic50 = item[8]
        ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
        temp_data.append((drug, cell, ic50))
    xd = []
    xc = []
    y = []
    lst_drug = []
    lst_cell = []
    random.shuffle(temp_data)
    for data in temp_data:
        drug, cell, ic50 = data
        if drug in drug_dict and cell in cell_dict:
            xd.append(drug_smile[drug_dict[drug]])
            xc.append(cell_feature[cell_dict[cell]])
            y.append(ic50)
            bExist[drug_dict[drug], cell_dict[cell]] = 1
            lst_drug.append(drug)
            lst_cell.append(cell)        
    with open('drug_dict', 'wb') as fp:
        pickle.dump(drug_dict, fp)
    xd, xc, y = np.asarray(xd), np.asarray(xc), np.asarray(y)
    size = int(xd.shape[0] * 0.8)
    size1 = int(xd.shape[0] * 0.9)
    with open('list_drug_mix_test', 'wb') as fp:
        pickle.dump(lst_drug[size1:], fp)        
    with open('list_cell_mix_test', 'wb') as fp:
        pickle.dump(lst_cell[size1:], fp)
    xd_train = xd[:size]
    xd_val = xd[size:size1]
    xd_test = xd[size1:]
    xc_train = xc[:size]
    xc_val = xc[size:size1]
    xc_test = xc[size1:]
    y_train = y[:size]
    y_val = y[size:size1]
    y_test = y[size1:]
    dataset = 'GDSC'
    print('preparing ', dataset + '_train.pt in pytorch format!')
    train_data = TestbedDataset(root='data', dataset=dataset+'_train_mix', xd=xd_train, xt=xc_train, y=y_train, smile_graph=smile_graph)
    val_data = TestbedDataset(root='data', dataset=dataset+'_val_mix', xd=xd_val, xt=xc_val, y=y_val, smile_graph=smile_graph)
    test_data = TestbedDataset(root='data', dataset=dataset+'_test_mix', xd=xd_test, xt=xc_test, y=y_test, smile_graph=smile_graph)

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return c_size, features, edge_index

def load_drug_smile():
    reader = csv.reader(open("drug_smiles.csv"))
    next(reader, None)
    drug_dict = {}
    drug_smile = []
    for item in reader:
        name = item[0]
        smile = item[2]
        if name in drug_dict:
            pos = drug_dict[name]
        else:
            pos = len(drug_dict)
            drug_dict[name] = pos
        drug_smile.append(smile)
    smile_graph = {}
    for smile in drug_smile:
        g = smile_to_graph(smile)
        smile_graph[smile] = g
    return drug_dict, drug_smile, smile_graph


for item in reader:
    drug = item[0]
    cell = item[3]
    ic50 = item[8]
    ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
    temp_data.append((drug, cell, ic50))

xd = []
xc = []
y = []
lst_drug = []
lst_cell = []
random.shuffle(temp_data)
for data in temp_data:
    drug, cell, ic50 = data
    if drug in drug_dict and cell in cell_dict:
        xd.append(drug_smile[drug_dict[drug]])
        xc.append(cell_feature[cell_dict[cell]])
        y.append(ic50)
        bExist[drug_dict[drug], cell_dict[cell]] = 1
        lst_drug.append(drug)
        lst_cell.append(cell)
    
with open('drug_dict', 'wb') as fp:
    pickle.dump(drug_dict, fp)

xd, xc, y = np.asarray(xd), np.asarray(xc), np.asarray(y)

size = int(xd.shape[0] * 0.8)
size1 = int(xd.shape[0] * 0.9)

with open('list_drug_mix_test', 'wb') as fp:
    pickle.dump(lst_drug[size1:], fp)
    
with open('list_cell_mix_test', 'wb') as fp:
    pickle.dump(lst_cell[size1:], fp)

xd_train = xd[:size]
xd_val = xd[size:size1]
xd_test = xd[size1:]

xc_train = xc[:size]
xc_val = xc[size:size1]
xc_test = xc[size1:]

y_train = y[:size]
y_val = y[size:size1]
y_test = y[size1:]

dataset = 'GDSC'
print('preparing ', dataset + '_train.pt in pytorch format!')

train_data = TestbedDataset(root='data', dataset=dataset+'_train_mix', xd=xd_train, xt=xc_train, y=y_train, smile_graph=smile_graph)
val_data = TestbedDataset(root='data', dataset=dataset+'_val_mix', xd=xd_val, xt=xc_val, y=y_val, smile_graph=smile_graph)
test_data = TestbedDataset(root='data', dataset=dataset+'_test_mix', xd=xd_test, xt=xc_test, y=y_test, smile_graph=smile_graph)

from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis', 
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None,smile_graph=None,saliency_map=False):
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.saliency_map = saliency_map
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, y,smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        pass
    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']
    def download(self):
        pass
    def _download(self):
        pass
    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    def process(self, xd, xt, y, smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            c_size, features, edge_index = smile_graph[smiles]
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]))            
            if self.saliency_map == True:
                GCNData.target = torch.tensor([target], dtype=torch.float, requires_grad=True)
            else:
                GCNData.target = torch.FloatTensor([target])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            data_list.append(GCNData)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    def getXD(self):
        return self.xd

xd_train = xd[:size]
xd_val = xd[size:size1]
xd_test = xd[size1:]

xc_train = xc[:size]
xc_val = xc[size:size1]
xc_test = xc[size1:]

y_train = y[:size]
y_val = y[size:size1]
y_test = y[size1:]

python training.py --model 0 --train_batch 1024 --val_batch 1024
 --test_batch 1024 --lr 0.0001 --num_epoch 300 --log_interval 20 --cuda_name "cuda:0"


def train(model, device, train_loader, optimizer, epoch, log_interval):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    loss_fn = nn.MSELoss()
    avg_loss = []
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())
        if batch_idx % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    return sum(avg_loss)/len(avg_loss)

for epoch in range(num_epoch):
    train_loss = train(model, device, train_loader, optimizer, epoch+1, log_interval)
    G,P = predicting(model, device, val_loader)
    ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P)] 
    G_test,P_test = predicting(model, device, test_loader)
    ret_test = [rmse(G_test,P_test),mse(G_test,P_test),pearson(G_test,P_test),spearman(G_test,P_test)]
    train_losses.append(train_loss)
    val_losses.append(ret[1])
    val_pearsons.append(ret[2])
    if ret[1]<best_mse:
        torch.save(model.state_dict(), model_file_name)
        with open(result_file_name,'w') as f:
            f.write(','.join(map(str,ret_test)))
        best_epoch = epoch+1
        best_mse = ret[1]
        best_pearson = ret[2]
        print(' rmse improved at epoch ', best_epoch, '; best_mse:', best_mse,model_st,dataset)
    else:
        print(' no improvement since epoch ', best_epoch, '; best_mse, best pearson:', best_mse, best_pearson, model_st, dataset)
draw_loss(train_losses, val_losses, loss_fig_name)
draw_pearson(val_pearsons, pearson_fig_name)

def calculate_value_individual_drug(modeling, num_mut, cuda_name, processed_data_file, model_file):
    dataset = "GDSC"
    with open ('mut_dict', 'rb') as fp:
        mut_dict = pickle.load(fp)
        mut_arr = np.asarray([k for k, v in mut_dict.items()]) 
    if (not os.path.isfile(processed_data_file)):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        test_data = TestbedDataset(root='data', dataset=dataset+'_bortezomib')
        test_loader = DataLoader(test_data)
        model_st = modeling.__name__
        print('\npredicting for ', dataset, ' using ', model_st)
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)
        lstY = []
        lstM = []
        lstV = []
        if os.path.isfile(model_file):
            model.load_state_dict(torch.load(model_file))
            model.eval()
            for data in test_loader:
                data = data.to(device)
                output, _ = model(data)
                data.target.retain_grad()
                output.backward()
                lstY.append(data.y.cpu().numpy()[0])
                grad = data.target.grad
                values, indexes = grad.topk(num_mut)
                lstV.append(values)
                lstM.append(mut_arr[np.squeeze(np.asarray(indexes.cpu().numpy()))])
        else:
            print('model is not available!')
        listCell = []
        with open ('cell_blind_sal', 'rb') as fp:
            listCell = pickle.load(fp)
        lstTopY = [lstY[k] for k in np.asarray(lstY).argsort()[:num_mut]]
        lstTopM = [lstM[k] for k in np.asarray(lstY).argsort()[:num_mut]]
        lstV = [lstV[k] for k in np.asarray(lstY).argsort()[:num_mut]]
        listCell = [listCell[k] for k in np.asarray(lstY).argsort()[:num_mut]]
        print(lstTopM)
        print(lstV)

python saliency_map.py --model 0 --num_feature 10 --processed_data_file "data/processed/GDSC_bortezomib.pt" 
--model_file "model_GINConvNet_GDSC.model" --cuda_name "cuda:0"