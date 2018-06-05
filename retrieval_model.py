import tensorflow as tf
from tensorflow.contrib.layers.python.layers import fully_connected

def add_fc(inputs, outdim, train_phase, scope_in):
    fc =  fully_connected(inputs, outdim, activation_fn=None, scope=scope_in + '/fc')
    fc_bnorm = tf.layers.batch_normalization(fc, momentum=0.1, epsilon=1e-5,
                         training=train_phase, name=scope_in + '/bnorm')
    fc_relu = tf.nn.relu(fc_bnorm, name=scope_in + '/relu')
    fc_out = tf.layers.dropout(fc_relu, seed=0, training=train_phase, name=scope_in + '/dropout')
    return fc_out

def pdist(x1, x2):
    """
        x1: Tensor of shape (h1, w)
        x2: Tensor of shape (h2, w)
        Return pairwise distance for each row vector in x1, x2 as
        a Tensor of shape (h1, h2)
    """
    x1_square = tf.reshape(tf.reduce_sum(x1*x1, axis=1), [-1, 1])
    x2_square = tf.reshape(tf.reduce_sum(x2*x2, axis=1), [1, -1])
    return tf.sqrt(x1_square - 2 * tf.matmul(x1, tf.transpose(x2)) + x2_square + 1e-4)

def embedding_loss(vd_embeds, sent_embeds, vd_labels, args):
    """
        im_embeds: (b, 512) video embedding tensors
        sent_embeds: (sample_size * b, 512) sentence embedding tensors
            where the order of sentence corresponds to the order of videos and
            setnteces for the same video are next to each other
        im_labels: (sample_size * b, b) boolean tensor, where (i, j) entry is
            True if and only if sentence[i], video[j] is a positive pair
    """
    # compute embedding loss
    sent_im_ratio = args.sample_size
    num_img = args.batch_size
    num_sent = num_img * sent_im_ratio

    sent_im_dist = pdist(sent_embeds, vd_embeds)
    # video loss: sentence, positive video, and negative video
    pos_pair_dist = tf.reshape(tf.boolean_mask(sent_im_dist, vd_labels), [num_sent, 1])
    neg_pair_dist = tf.reshape(tf.boolean_mask(sent_im_dist, ~vd_labels), [num_sent, -1])
    im_loss = tf.clip_by_value(args.margin + pos_pair_dist - neg_pair_dist, 0, 1e6)
    im_loss = tf.reduce_mean(tf.nn.top_k(im_loss, k=args.num_neg_sample)[0])
    # sentence loss: video, positive sentence, and negative sentence
    neg_pair_dist = tf.reshape(tf.boolean_mask(tf.transpose(sent_im_dist), ~tf.transpose(vd_labels)), [num_img, -1])
    neg_pair_dist = tf.reshape(tf.tile(neg_pair_dist, [1, sent_im_ratio]), [num_sent, -1])
    sent_loss = tf.clip_by_value(args.margin + pos_pair_dist - neg_pair_dist, 0, 1e6)
    sent_loss = tf.reduce_mean(tf.nn.top_k(sent_loss, k=args.num_neg_sample)[0])
    # sentence only loss (neighborhood-preserving constraints)
    sent_sent_dist = pdist(sent_embeds, sent_embeds)
    sent_sent_mask = tf.reshape(tf.tile(tf.transpose(vd_labels), [1, sent_im_ratio]), [num_sent, num_sent])
    pos_pair_dist = tf.reshape(tf.boolean_mask(sent_sent_dist, sent_sent_mask), [-1, sent_im_ratio])
    pos_pair_dist = tf.reduce_max(pos_pair_dist, axis=1, keep_dims=True)
    neg_pair_dist = tf.reshape(tf.boolean_mask(sent_sent_dist, ~sent_sent_mask), [num_sent, -1])
    sent_only_loss = tf.clip_by_value(args.margin + pos_pair_dist - neg_pair_dist, 0, 1e6)
    sent_only_loss = tf.reduce_mean(tf.nn.top_k(sent_only_loss, k=args.num_neg_sample)[0])

    loss = im_loss * args.im_loss_factor + sent_loss + sent_only_loss * args.sent_only_loss_factor
    return loss


def recall_k(vd_embeds, sent_embeds, vd_labels, ks=None):
    """
        Compute recall at given ks.
    """
    sent_im_dist = pdist(sent_embeds, vd_embeds)
    def retrieval_recall(dist, labels, k):
        # Use negative distance to find the index of
        # the smallest k elements in each row.
        pred = tf.nn.top_k(-dist, k=k)[1]
        # Create a boolean mask for each column (k value) in pred,
        # s.t. mask[i][j] is 1 iff pred[i][k] = j.
        pred_k_mask = lambda topk_idx: tf.one_hot(topk_idx, labels.shape[1],
                            on_value=True, off_value=False, dtype=tf.bool)
        # Create a boolean mask for the predicted indicies
        # by taking logical or of boolean masks for each column,
        # s.t. mask[i][j] is 1 iff j is in pred[i].
        pred_mask = tf.reduce_any(tf.map_fn(
                pred_k_mask, tf.transpose(pred), dtype=tf.bool), axis=0)
        # Entry (i, j) is matched iff pred_mask[i][j] and labels[i][j] are 1.
        matched = tf.cast(tf.logical_and(pred_mask, labels), dtype=tf.float32)
        return tf.reduce_mean(tf.reduce_max(matched, axis=1))
    return tf.concat(
        [tf.map_fn(lambda k: retrieval_recall(tf.transpose(sent_im_dist), tf.transpose(vd_labels), k),
                   ks, dtype=tf.float32),
         tf.map_fn(lambda k: retrieval_recall(sent_im_dist, vd_labels, k),
                   ks, dtype=tf.float32)],
        axis=0)


def embedding_model(vd_feats, sent_feats, train_phase, vd_labels,
                    fc_dim = 2048, embed_dim = 512):
    """
        Build two-branch embedding networks.
        fc_dim: the output dimension of the first fc layer.
        embed_dim: the output dimension of the second fc layer, i.e.
                   embedding space dimension.
    """
    # video branch.
    dim_hidden = 1000
    dim_video = 4096
    n_video_lstm_step = 80
    batch_size = vd_feats.shape[0]
    lstm1 = tf.contrib.rnn.BasicLSTMCell(dim_hidden, state_is_tuple=False)
    lstm2 = tf.contrib.rnn.BasicLSTMCell(dim_hidden, state_is_tuple=False)
    encode_video_W = tf.Variable(tf.random_uniform([dim_video, dim_hidden], -0.1, 0.1), name='encode_video_W')
    encode_video_b = tf.Variable(tf.zeros([dim_hidden]), name='encode_video_b')
    video_flat = tf.reshape(vd_feats, [-1, dim_video])
    video_emb = tf.nn.xw_plus_b(video_flat, encode_video_W, encode_video_b)
    video_emb = tf.reshape(video_emb, [batch_size, n_video_lstm_step, dim_hidden])

    state1 = tf.zeros([batch_size, lstm1.state_size])
    state2 = tf.zeros([batch_size, lstm2.state_size])
    padding = tf.zeros([batch_size, dim_hidden])
    with tf.variable_scope(tf.get_variable_scope()) as scope:
        for i in range(0, n_video_lstm_step):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            with tf.variable_scope("LSTM1"):
                output1, state1 = lstm1(video_emb[:, i, :], state1)
            with tf.variable_scope("LSTM2"):
                output2, state2 = lstm2(tf.concat([padding, output1], 1), state2)

    im_fc1 = add_fc(tf.concat([state1, state2], 1), fc_dim, train_phase, 'vd_embed_1')
    im_fc2 = fully_connected(im_fc1, embed_dim, activation_fn=None,
                             scope = 'vd_embed_2')
    v_embed = tf.nn.l2_normalize(im_fc2, 1, epsilon=1e-10)
    # Text branch.
    sent_fc1 = add_fc(sent_feats, fc_dim, train_phase,'sent_embed_1')
    sent_fc2 = fully_connected(sent_fc1, embed_dim, activation_fn=None,
                               scope = 'sent_embed_2')
    s_embed = tf.nn.l2_normalize(sent_fc2, 1, epsilon=1e-10)
    return v_embed, s_embed


def setup_train_model(vd_feats, sent_feats, train_phase, vd_labels, args):
    # im_feats b x video_feature_dim
    # sent_feats 5b x sent_feature_dim
    # train_phase bool (Should be True.)
    # im_labels 5b x b
    v_embed, s_embed = embedding_model(vd_feats, sent_feats, train_phase, vd_labels)
    loss = embedding_loss(v_embed, s_embed, vd_labels, args)
    return loss


def setup_eval_model(vd_feats, sent_feats, train_phase, vd_labels):
    # im_feats b x video_feature_dim
    # sent_feats 5b x sent_feature_dim
    # train_phase bool (Should be False.)
    # im_labels 5b x b
    v_embed, s_embed = embedding_model(vd_feats, sent_feats, train_phase, vd_labels)
    recall = recall_k(v_embed, s_embed, vd_labels, ks=tf.convert_to_tensor([1,5,10]))
    return recall


def setup_sent_eval_model(vd_feats, sent_feats, train_phase, vd_labels, args):
    # im_feats b x video_feature_dim
    # sent_feats 5b x sent_feature_dim
    # train_phase bool (Should be False.)
    # im_labels 5b x b
    _, s_embed = embedding_model(vd_feats, sent_feats, train_phase, vd_labels)
    # Create 5b x 5b sentence labels, wherthe 5 x 5 blocks along the diagonal
    num_sent = args.batch_size * args.sample_size
    sent_labels = tf.reshape(tf.tile(tf.transpose(vd_labels),
                                     [1, args.sample_size]), [num_sent, num_sent])
    sent_labels = tf.logical_and(sent_labels, ~tf.eye(num_sent, dtype=tf.bool))
    # For topk, query k+1 since top1 is always the sentence itself, with dist 0.
    recall = recall_k(s_embed, s_embed, sent_labels, ks=tf.convert_to_tensor([2,6,11]))[:3]
    return recall
