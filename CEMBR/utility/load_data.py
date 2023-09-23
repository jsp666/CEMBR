import numpy as np
import random as rd
import scipy.sparse as sp
from time import time

class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        pv_file = path +'/pv.txt'
        cart_file = path +'/cart.txt'

        # get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {} # 负样本池

        self.exist_users = []   # 训练集中的uid

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))    # 训练集中有多少item
                    self.n_users = max(self.n_users, uid)   # 训练集中有多少user
                    self.n_train += len(items)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1

        self.print_statistics()
        # 下列生成的矩阵不是节点和节点的邻接矩阵
        # dok_matrix采用字典来记录矩阵中不为0的元素
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)  # 行为buy的user-item邻接矩阵
        self.R_pv = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)   # 行为pv的user-item邻接矩阵
        self.R_cart = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32) # 行为cart的user-item邻接矩阵

        self.train_items, self.test_set = {}, {}
        self.pv_set, self.cart_set = {},{}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]  # 训练集结构{用户id:物品集}

                    for i in train_items:
                        self.R[uid, i] = 1.

                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items
        with open(pv_file) as f_pv:
            for l in f_pv.readlines():
                if len(l) == 0: break
                l = l.strip('\n')
                items = [int(i) for i in l.split(' ')]
                uid, pv_items = items[0], items[1:]

                for i in pv_items:
                    self.R_pv[uid, i] = 1.
                self.pv_set[uid] = pv_items

        with open(cart_file) as f_cart:
            for l in f_cart.readlines():
                if len(l) == 0: break
                l = l.strip('\n')
                items = [int(i) for i in l.split(' ')]
                uid, cart_items = items[0], items[1:]

                for i in cart_items:
                    self.R_cart[uid, i] = 1.
                self.cart_set[uid] = cart_items


    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')

            adj_mat_pv = sp.load_npz(self.path + '/s_adj_mat_pv.npz')
            norm_adj_mat_pv = sp.load_npz(self.path + '/s_norm_adj_mat_pv.npz')
            mean_adj_mat_pv = sp.load_npz(self.path + '/s_mean_adj_mat_pv.npz')

            adj_mat_cart = sp.load_npz(self.path + '/s_adj_mat_cart.npz')
            norm_adj_mat_cart = sp.load_npz(self.path + '/s_norm_adj_mat_cart.npz')
            mean_adj_mat_cart = sp.load_npz(self.path + '/s_mean_adj_mat_cart.npz')
            
            # 邻接矩阵的形状(29693, 29693),用户数21716+物品数7977
            print('already load adjacency matrix, use {:.4f}s'.format(time() - t1))
        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat(self.R)
            adj_mat_pv, norm_adj_mat_pv, mean_adj_mat_pv = self.create_adj_mat(self.R_pv)
            adj_mat_cart, norm_adj_mat_cart, mean_adj_mat_cart = self.create_adj_mat(self.R_cart)
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)

            sp.save_npz(self.path + '/s_adj_mat_pv.npz', adj_mat_pv)
            sp.save_npz(self.path + '/s_norm_adj_mat_pv.npz', norm_adj_mat_pv)
            sp.save_npz(self.path + '/s_mean_adj_mat_pv.npz', mean_adj_mat_pv)

            sp.save_npz(self.path + '/s_adj_mat_cart.npz', adj_mat_cart)
            sp.save_npz(self.path + '/s_norm_adj_mat_cart.npz', norm_adj_mat_cart)
            sp.save_npz(self.path + '/s_mean_adj_mat_cart.npz', mean_adj_mat_cart)
            
        try:
            pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
            pre_adj_mat_pv = sp.load_npz(self.path + '/s_pre_adj_mat_pv.npz')
            pre_adj_mat_cart = sp.load_npz(self.path + '/s_pre_adj_mat_cart.npz')
        except Exception:
            # buy behavior邻接矩阵标准化
            rowsum = np.array(adj_mat.sum(axis = 1))   # 在无向图中,邻接矩阵可以转换成度矩阵
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            print('generate pre adjacency matrix.')
            pre_adj_mat = norm_adj.tocsr()
            sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)
            # pv behavior邻接矩阵标准化
            rowsum = np.array(adj_mat_pv.sum(axis = 1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat_pv)
            norm_adj = norm_adj.dot(d_mat_inv)
            print('generate pre view adjacency matrix.')
            pre_adj_mat_pv = norm_adj.tocsr()
            sp.save_npz(self.path + '/s_pre_adj_mat_pv.npz', norm_adj)
            # add to cart behavior邻接矩阵标准化
            rowsum = np.array(adj_mat_cart.sum(axis = 1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat_cart)
            norm_adj = norm_adj.dot(d_mat_inv)
            print('generate pre cart adjacency matrix.')
            pre_adj_mat_cart = norm_adj.tocsr()
            sp.save_npz(self.path + '/s_pre_adj_mat_cart.npz', norm_adj)

        return pre_adj_mat, pre_adj_mat_pv, pre_adj_mat_cart

    # 创建每个行为的邻接矩阵
    def create_adj_mat(self, which_R):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = which_R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix, use {:.4f}s'.format(time() - t1))

        t2 = time()
        
        # 标准化邻接矩阵
        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(axis = 1))

            d_inv = np.power(rowsum, -1).flatten()  # 为什么是-1次方
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))    # 加自环
        mean_adj_mat = normalized_adj_single(adj_mat)   # 不加自环

        print('already normalize adjacency matrix, use {:.4f}s'.format(time() - t2))
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items
    
    def sample_test(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.test_set.keys(), self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.test_set[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in (self.test_set[u]+self.train_items[u]) and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items
    
        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items
    
    def get_num_users_items(self):
        return self.n_users, self.n_items

    # 稀疏度计算公式(n_train+n_test)/(n_users*n_items)
    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))

    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state

    def create_sparsity_split(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train)
        n_rates = 0

        split_state = []
        temp0=[]
        temp1=[]
        temp2=[]
        temp3=[]
        temp4=[]

        #print user_n_iid

        for idx, n_iids in enumerate(sorted(user_n_iid)):
            if n_iids <9:
                temp0+=user_n_iid[n_iids]
            elif n_iids <13:
                temp1+=user_n_iid[n_iids]
            elif n_iids <17:
                temp2+=user_n_iid[n_iids]
            elif n_iids <20:
                temp3+=user_n_iid[n_iids]
            else:
                temp4+=user_n_iid[n_iids]
            
        split_uids.append(temp0)
        split_uids.append(temp1)
        split_uids.append(temp2)
        split_uids.append(temp3)
        split_uids.append(temp4)
        split_state.append("#users=[%d]"%(len(temp0)))
        split_state.append("#users=[%d]"%(len(temp1)))
        split_state.append("#users=[%d]"%(len(temp2)))
        split_state.append("#users=[%d]"%(len(temp3)))
        split_state.append("#users=[%d]"%(len(temp4)))


        return split_uids, split_state

    def create_sparsity_split2(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)
        return split_uids, split_state