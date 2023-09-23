import utility.metrics as metrics
from utility.parser_Beibei import parse_args
# from utility.parser_Taobao import parse_args
# from utility.parser_ml-1m import parse_args
# from utility.parser_yelp import parse_args
from utility.load_data import *
import multiprocessing
import heapq

cores = multiprocessing.cpu_count() // 2

args = parse_args()
Ks = eval(args.Ks)

data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
if args.dataset == 'Beibei':
    BATCH_SIZE = args.batch_size // 2
else:
    BATCH_SIZE = args.batch_size
# 参数：测试集item, test_items = all_items - train_items, 用户u的预测评分
# part 1.1
def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    # 前K_max预测分数最高的item
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

# full 1.2
def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

# full 1.1
def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

# part 1.2
def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}

# 参数x: [[rate_item1, rate_item2, rate_item3, ...], uid]
# 1
def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]   # [7977]
    # uid
    u = x[1]
    try:
        # user u's items in the training set [item1, item2, item3, ...]
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []
    # user u's items in the test set [item1, item2, ...]
    user_pos_test = data_generator.test_set[u]

    all_items = set(range(ITEM_NUM))    # items set

    test_items = list(all_items - set(training_items))  # 求差集,所有item-训练集item

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)

def test_one_user_train(x):
    # user u's ratings for user u
    rating = x[0]
    # uid
    u = x[1]

    training_items = []
    # user u's items in the train set
    user_pos_test = data_generator.train_items[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)

def test(sess, model, users_to_test, drop_flag=False, batch_test_flag=False, train_set_flag=0):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE * 15

    test_users = users_to_test  # uid
    n_test_users = len(test_users)  # 测试集中的用户数
    n_user_batchs = n_test_users // u_batch_size + 1    # 测试过程中的batch大小

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end] # 不足batch_size
        rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))    # 评分[0-1] shape = (batch, num_items)
        
        if batch_test_flag:
            n_item_batchs = ITEM_NUM // i_batch_size + 1
            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

                item_batch = range(i_start, i_end)

                if drop_flag == False:
                    i_rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                                model.pos_items: item_batch})
                else:
                    i_rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                                model.pos_items: item_batch,
                                                                model.node_dropout: [0.] * len(eval(args.layer_size)),
                                                                model.mess_dropout: [0.] * len(eval(args.layer_size))})
                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == ITEM_NUM

        else:
            item_batch = range(ITEM_NUM)    # 所有item(0-7976)都加入到模型中

            if drop_flag == False:
                rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                            model.pos_items: item_batch})
            else:
                # 返回预测分数
                rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                            model.pos_items: item_batch,
                                                            model.node_dropout: [0.] * len(eval(args.layer_size)),
                                                            model.mess_dropout: [0.] * len(eval(args.layer_size))})

        user_batch_rating_uid = zip(rate_batch, user_batch)
        if train_set_flag == 0:
            # 每个用户有5个指标recall precision ndcg hit_ratio,同时每个指标有top10 50 100三个数据
            batch_result = pool.map(test_one_user, user_batch_rating_uid)
            # batch_result = list(map(test_one_user, user_batch_rating_uid))
        else:
            batch_result = pool.map(test_one_user_train, user_batch_rating_uid)
            # batch_result = list(map(test_one_user, user_batch_rating_uid))
        
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision'] / n_test_users
            result['recall'] += re['recall'] / n_test_users
            result['ndcg'] += re['ndcg'] / n_test_users
            result['hit_ratio'] += re['hit_ratio'] / n_test_users
            result['auc'] += re['auc'] / n_test_users

    assert count == n_test_users
    pool.close()
    return result