import tensorflow.compat.v1 as tf
import sys
import copy

from utility.batch_test import *

class Cross_Attention(object):
    def __init__(self, in_feats, out_feats, num_relations, num_heads):
        super(Cross_Attention, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._num_relations = num_relations
        self._num_heads = num_heads

    def build(self, node_features, relations_crossing_attention_weight):
        """
        :param node_features: list, [(num_nodes, n_heads * hidden_dim)]
        :param relations_crossing_attention_weight: Parameter the shape is (n_heads, hidden_dim)
        :return: output_features: Tensor
        """
        # (num_relations, num_nodes, n_heads, hidden_dim)
        node_features = tf.reshape(tf.stack(node_features, 0), [self._num_relations, -1, self._num_heads, self._out_feats])
        # shape -> (num_relations, num_nodes, n_heads, 1),  (node_relations_num, num_nodes, n_heads, hidden_dim) * (n_heads, hidden_dim)
        node_relation_attention = tf.reduce_sum(
            node_features * relations_crossing_attention_weight, -1, keepdims=True)
        node_relation_attention = tf.nn.softmax(
            tf.nn.leaky_relu(node_relation_attention), dim=0)
        # shape -> (num_nodes, n_heads, hidden_dim),  (num_relations, num_nodes, n_heads, hidden_dim) * (num_relations, num_nodes, n_heads, 1)
        node_features = tf.reduce_sum(node_features * node_relation_attention, 0)
        # shape -> (num_nodes, n_heads * hidden_dim)
        node_features = tf.reshape(node_features, [-1, self._num_heads * self._out_feats])

        return node_features


class CEMBR(object):
    def __init__(self, max_item_view, max_item_cart, max_item_buy, data_config):
        # argument settings
        self.model_type = 'CEMBR'
        self.adj_type = args.adj_type   # pre
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_fold = 100
        self.wid = eval(args.wid)    # 0.1 for beibei, 0.01 for taobao
        self.buy_adj = data_config['buy_adj']   # 购买行为的邻接矩阵
        self.pv_adj = data_config['pv_adj']     # 查看行为的邻接矩阵
        self.cart_adj = data_config['cart_adj'] # 添加到购物车行为的邻接矩阵
        # self.n_nonzero_elems = self.norm_adj.count_nonzero()
        self.lr = args.lr   # 学习率
        self.emb_dim = args.embed_size  # 隐特征
        self.batch_size = args.batch_size   # 批量处理数据的大小
        self.weight_size = eval(args.layer_size)    # 权重shape
        self.n_layers = len(self.weight_size)   # GNN层数为4
        # self.regs = eval(args.regs) # 正则化,在后面未调用
        self.decay = args.decay # 10 for beibei, 1e-1 for taobao
        self.verbose = args.verbose # 日志显示
        self.max_item_view = max_item_view
        self.max_item_cart = max_item_cart
        self.max_item_buy = max_item_buy
        self.coefficient = eval(args.coefficient) # 0.0/6, 5.0/6, 1.0/6 for beibei and 1.0/6, 4.0/6, 1.0/6 for taobao

        self.n_heads = 4
        self.hidden_dim = self.emb_dim // self.n_heads

        tf.disable_eager_execution()
        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        # for train
        # placeholder definition
        self.input_u = tf.placeholder(tf.int32, [None, 1], name="input_uid")    # shape = (None,1)
        self.label_view = tf.placeholder(tf.int32, [None, self.max_item_view], name="label_view")   # shape = (None,max_item_view)
        self.label_cart = tf.placeholder(tf.int32, [None, self.max_item_cart], name="label_cart")   # shape = (None,max_item_cart)
        self.label_buy = tf.placeholder(tf.int32, [None, self.max_item_buy], name="label_buy")  # shape = (None,max_item_buy)
        self.labels = [self.label_view, self.label_cart, self.label_buy]
        self.n_relations = len(self.labels)    # 3种行为

        # dropout
        self.node_dropout_flag = args.node_dropout_flag # 是否使用node_dropout
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

        # 1.for test
        # 2.负采样训练
        self.users = tf.placeholder(tf.int32, shape=(None,))    # shape = (None,)
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))    # 正样本 shape = (None,)
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))    # 负样本

        self.cross_attention = Cross_Attention(in_feats=self.n_heads * self.hidden_dim,
                                               out_feats=self.hidden_dim,
                                               num_relations=self.n_relations,
                                               num_heads=self.n_heads)

        self.initializer = tf.keras.initializers.glorot_uniform()
        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights = self._init_weights()

        """
        *********************************************************
        Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
        """
        self.ua_embeddings, self.ia_embeddings, self.r_embeddings = self.create_gcn_embed()
        
        # self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        # self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        # self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)

        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """
        # for test
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)

        self.dot = tf.einsum('ac,bc->abc', self.u_g_embeddings, self.pos_i_g_embeddings)
        self.batch_ratings = tf.einsum('ajk,lk->aj', self.dot, self.r_embeddings[-1])

        self.mf_loss, self.emb_loss = self.create_non_sampling_loss()
        self.loss = self.mf_loss + self.emb_loss
        # self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.u_g_embeddings,
                                                                        #   self.pos_i_g_embeddings,
                                                                        #   self.neg_i_g_embeddings)
        # self.loss = self.mf_loss + self.emb_loss + self.reg_loss
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    # 1
    def _init_weights(self):
        all_weights = dict()
        self.W = tf.Variable(self.initializer(shape=[self.n_relations, 1, self.n_users + self.n_items], dtype=tf.float32), name='W')
        self.relations_crossing_attention_weight = tf.Variable(self.initializer(shape=[self.n_heads, self.hidden_dim], dtype=tf.float32), name='relations_crossing_attention_weight')
        # 初始化embedding
        # initializer = tf.contrib.layers.xavier_initializer()
        # 用户隐特征(n_user, 64)
        all_weights['user_embedding'] = tf.Variable(self.initializer([self.n_users, self.emb_dim]),
                                                    name='user_embedding')
        # 项目隐特征(n_item, 64)
        all_weights['item_embedding'] = tf.Variable(self.initializer([self.n_items, self.emb_dim]),
                                                    name='item_embedding')
        # 关系隐特征(n_relation, 64)
        all_weights['relation_embedding'] = tf.Variable(self.initializer([self.n_relations, self.emb_dim]),
                                                    name='relation_embedding')
        print('using xavier initialization')

        self.weight_size_list = [self.emb_dim] + self.weight_size

        for k in range(self.n_layers):
            all_weights['W_gc_%d' % k] = tf.Variable(
                self.initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_gc_%d' % k)
            all_weights['W_rel_%d' % k] = tf.Variable(
                self.initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_rel_%d' % k)
            all_weights['B_rel_%d' % k] = tf.Variable(
                self.initializer([self.n_relations, self.weight_size_list[k + 1]]), name='B_rel_%d' % k)

        return all_weights

    # 功能同2.1
    def _split_A_hat(self, A):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            temp = self._convert_sp_mat_to_sp_tensor(A[start:end])
            # temp = self._update_node_attention(input, temp, self.n_users + self.n_items)
            A_fold_hat.append(temp)
        return A_fold_hat

    # 修改gcn_embedding
    def create_gcn_embed(self):
        A_fold_hat_pv = self._convert_sp_mat_to_sp_tensor(self.pv_adj)
        A_fold_hat_cart = self._convert_sp_mat_to_sp_tensor(self.cart_adj)
        A_fold_hat_buy = self._convert_sp_mat_to_sp_tensor(self.buy_adj)
        A_fold_hat_pv = self._dropout_sparse(A_fold_hat_pv, 1 - self.node_dropout[0])
        A_fold_hat_cart = self._dropout_sparse(A_fold_hat_cart, 1 - self.node_dropout[0])
        A_fold_hat_buy = self._dropout_sparse(A_fold_hat_buy, 1 - self.node_dropout[0])
        
        A_fold_hat = [A_fold_hat_pv, A_fold_hat_cart, A_fold_hat_buy]

        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        r_embeddings = self.weights['relation_embedding']
        
        all_embeddings = [embeddings]   # (29693,64)
        all_r = [r_embeddings]
        # GCN核心代码,建模为统一的异质图
        for k in range(self.n_layers):
            embeds = []
            for i in range(self.n_relations):
                r = tf.nn.embedding_lookup(r_embeddings, i)
                r = tf.reshape(r, [-1, self.emb_dim])
                temp_embed = tf.sparse_tensor_dense_matmul(A_fold_hat[i], embeddings)
                
                # embeddings.shape = (29693,64) x r(1,64) x (64,64)
                embeddings = temp_embed * tf.keras.activations.sigmoid(tf.transpose(self.W[i]) * r)

                embeds.append(embeddings)
            
            # 自动学习行为的权重
            embeddings = self.cross_attention.build(embeds, self.relations_crossing_attention_weight)
            embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[0])

            # 更新关系类型的向量表示
            # 试试非线性激活函数?
            r_embeddings = tf.matmul(r_embeddings, self.weights['W_rel_%d' % k]) + self.weights['B_rel_%d' % k]
            r_embeddings = tf.nn.dropout(r_embeddings, 1 - self.node_dropout[0])
            all_embeddings += [embeddings]
            all_r.append(r_embeddings)
        
        # 不同层的重要性?
        # MLP?
        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1)

        all_r = tf.stack(all_r, 1)
        all_r = tf.reduce_mean(all_r, axis=1, keepdims=True)

        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        # item0-7976, token是对item7977的表示
        token_embedding = tf.zeros([1, self.emb_dim], name='token_embedding')
        i_g_embeddings = tf.concat([i_g_embeddings, token_embedding], axis = 0) # 7977+1?为什么在插入一行
        
        return u_g_embeddings, i_g_embeddings, all_r

    # 3 非采样损失函数
    def create_non_sampling_loss(self):
        loss = []
        uid = tf.nn.embedding_lookup(self.ua_embeddings, self.input_u)
        uid = tf.reshape(uid, [-1, self.emb_dim])
        temp = tf.einsum('ab,ac->bc', self.ia_embeddings, self.ia_embeddings) * tf.einsum('ab,ac->bc', uid, uid)
        for i, label in enumerate(self.labels):
            iid = tf.nn.embedding_lookup(self.ia_embeddings, label)
            # self.pos_num_cart = tf.cast(tf.not_equal(self.label_cart, self.n_items), 'float32')
            # self.pos_cart = tf.einsum('ab,abc->abc', self.pos_num_cart, self.pos_cart)
            ui_score = tf.einsum('ac,abc->abc', uid, iid)
            ui_score = tf.einsum('ajk,lk->aj', ui_score, self.r_embeddings[i])

            temp_loss = self.wid[i] * tf.reduce_sum(temp * tf.matmul(self.r_embeddings[i], self.r_embeddings[i], transpose_a=True))
            temp_loss += tf.reduce_sum((1.0 - self.wid[i]) * tf.square(ui_score) - 2.0 * ui_score)
            
            loss.append(self.coefficient[i] * temp_loss)
        mf_loss = sum(loss)

        # regularizer = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(self.weights['item_embedding'])
        regularizer = tf.nn.l2_loss(uid) + tf.nn.l2_loss(self.ia_embeddings)

        emb_loss = self.decay * regularizer # 正则化

        return mf_loss, emb_loss

    # 2.1.2
    def _dropout_sparse(self, X, keep_prob):
        """
            Dropout for sparse tensors.
        """
        noise_shape = [X.values.shape[0]]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape) # 从均匀分布中输出随机值
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)  # 强制转换成布尔型
        pre_out = tf.sparse.retain(X, dropout_mask) # 在一个SparseTensor中保留指定的非空值.

        return pre_out * tf.div(1., keep_prob)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()    # 将邻接矩阵的行索引和列索引组成矩阵,并转置
        return tf.SparseTensor(indices, coo.data, coo.shape)    # 稀疏矩阵

def get_labels(temp_set, k = 0.9999):
    max_item = 0

    item_lenth = []
    for i in temp_set:
        item_lenth.append(len(temp_set[i]))
        if len(temp_set[i]) > max_item:
            max_item = len(temp_set[i])
    item_lenth.sort()

    max_item = item_lenth[int(len(item_lenth) * k) - 1] # 最多项目数

    # 将每个用户对应的项目数对齐
    for i in temp_set:
        if len(temp_set[i]) > max_item:
            temp_set[i] = temp_set[i][0:max_item]   # 截取0-max_item
            # 随机采样
            # temp_set[i] = random.sample(temp_set[i], max_item)
        while len(temp_set[i]) < max_item:
            temp_set[i].append(n_items)             # 不够max_item,填充n_items
    return max_item, temp_set

# 将用户和购买、查看、添加到购物车的项目分别保存
def get_train_instances1(view_label, cart_label, buy_label):
    user_train, view_item, cart_item, buy_item = [], [], [], []
    # buy为训练集
    for i in buy_label.keys():
        user_train.append(i)    # [user1, user2, user3, ...]
        buy_item.append(buy_label[i])   # [item1, item2, item3, ...]
        # if not view_label.has_key(i):
        if i not in view_label: # 用户没有view的行为
            view_item.append([n_items] * max_item_view)
        else:
            view_item.append(view_label[i])

        # if not cart_label.has_key(i):
        if i not in cart_label: # 用户没有add to cart的行为
            cart_item.append([n_items] * max_item_cart)
        else:
            cart_item.append(cart_label[i])

    user_train = np.array(user_train)
    view_item = np.array(view_item)
    cart_item = np.array(cart_item)
    buy_item = np.array(buy_item)
    user_train = user_train[:, np.newaxis]
    return user_train, view_item, cart_item, buy_item

if __name__ == '__main__':
    tf.set_random_seed(2020)
    np.random.seed(2020)
    print(args)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    pre_adj, pre_adj_pv, pre_adj_cart = data_generator.get_adj_mat()

    config['buy_adj'] = pre_adj
    config['pv_adj'] = pre_adj_pv
    config['cart_adj'] = pre_adj_cart
    print('use the pre adjcency matrix')

    n_users, n_items = data_generator.n_users, data_generator.n_items
    
    train_items = copy.deepcopy(data_generator.train_items)
    pv_set = copy.deepcopy(data_generator.pv_set)
    cart_set = copy.deepcopy(data_generator.cart_set)

    # 用户行为_标签{用户id:[项目id]},将不同的item数对齐,对齐时使用了一个不存在的item7977
    # 创新点4,基于GCN的邻居采样策略
    max_item_buy, buy_label = get_labels(train_items)   # 目标行为-购买
    max_item_view, view_label = get_labels(pv_set)      # 辅助行为-查看
    max_item_cart, cart_label = get_labels(cart_set)    # 辅助行为-添加到购物车

    t0 = time()

    # build model
    model = CEMBR(max_item_view, max_item_cart, max_item_buy, data_config=config)

    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    cur_best_pre_0 = 0.
    print('without pretraining.')

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []

    stopping_step = 0
    should_stop = False
    # 该实验没有使用负样本训练模型
    user_train1, view_item1, cart_item1, buy_item1 = get_train_instances1(view_label, cart_label, buy_label)

    for epoch in range(args.epoch):
        # 打乱排列顺序
        shuffle_indices = np.random.permutation(np.arange(len(user_train1)))
        user_train1 = user_train1[shuffle_indices]
        view_item1 = view_item1[shuffle_indices]
        cart_item1 = cart_item1[shuffle_indices]
        buy_item1 = buy_item1[shuffle_indices]

        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.

        n_batch = len(user_train1) // args.batch_size
        # 换成dataloader,可以避免占用内存过大?
        for idx in range(n_batch):
            start_index = idx * args.batch_size
            end_index = min((idx + 1) * args.batch_size, len(user_train1))  # 不足batch_size

            u_batch = user_train1[start_index:end_index]
            v_batch = view_item1[start_index:end_index]
            c_batch = cart_item1[start_index:end_index]
            b_batch = buy_item1[start_index:end_index]

            _, batch_loss, batch_mf_loss, batch_emb_loss = sess.run(
                [model.opt, model.loss, model.mf_loss, model.emb_loss],
                feed_dict={model.input_u: u_batch,
                           model.label_buy: b_batch,
                           model.label_view: v_batch,
                           model.label_cart: c_batch,
                           model.node_dropout: eval(args.node_dropout),
                           model.mess_dropout: eval(args.mess_dropout)})
            
            loss += batch_loss / n_batch    # 为什么除以n_batch?
            mf_loss += batch_mf_loss / n_batch
            emb_loss += batch_emb_loss / n_batch
            
            # print('idx={}, batch_loss={}, batch_mf_loss={}, batch_emb_loss={}'.format(idx, batch_loss, batch_mf_loss, batch_emb_loss))
            # print('idx={}, loss={}, mf_loss={}, emb_loss={}'.format(idx, loss, mf_loss, emb_loss))

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss)
                print(perf_str)
            continue
        
        # users_to_test = list(data_generator.train_items.keys())
        # ret = test(sess, model, users_to_test, drop_flag=True, train_set_flag=1)
        # perf_str = 'Epoch %d: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
        #            'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
        #            (epoch, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
        #             ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
        #             ret['ndcg'][0], ret['ndcg'][-1])
        # print(perf_str)
        
        t2 = time()
        # test_set{uid:[item]}
        users_to_test = list(data_generator.test_set.keys())    # 测试集
        ret = test(sess, model, users_to_test, drop_flag=True)  # Beibei
        # ret = test(sess, model, users_to_test, batch_test_flag=True, drop_flag=True)	# Taobao

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch {} [{:.1f}s + {:.1f}s]: train==[{:.5f}={:.5f} + {:.5f}],' \
                       'recall={}, precision={}, hit={}, ndcg={}'.format(
                        epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss,
                        ret['recall'], ret['precision'], ret['hit_ratio'], ret['ndcg'])
            print(perf_str)

            """
                Get the performance w.r.t. different sparsity levels.
            """
            if 0:
                users_to_test_list, split_state = data_generator.get_sparsity_split()

                for i, users_to_test in enumerate(users_to_test_list):
                    ret = test(sess, model, users_to_test, drop_flag=True)

                    final_perf = "recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                         ('\t'.join(['%.5f' % r for r in ret['recall']]),
                          '\t'.join(['%.5f' % r for r in ret['precision']]),
                          '\t'.join(['%.5f' % r for r in ret['hit_ratio']]),
                          '\t'.join(['%.5f' % r for r in ret['ndcg']]))
                    print(final_perf)
        
        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            model.save_weights(args.weights_path + args.model_type)
            print('save the weights in path:', args.weights_path + args.model_type)

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)