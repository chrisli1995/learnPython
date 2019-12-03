import numpy as np
import tensorflow as tf
from tensorflow import layers

from tensorflow.python.ops import array_ops
from tensorflow.contrib import seq2seq
from tensorflow.contrib.seq2seq import BahdanauAttention
from tensorflow.contrib.seq2seq import LuongAttention
from tensorflow.contrib.seq2seq import AttentionWrapper
from tensorflow.contrib.seq2seq import BeamSearchDecoder

from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import ResidualWrapper

from word_sequence import WordSequence
from data_utils import _get_embed_device



class SequenceToSequence(object):
    '''
    基本流程
        __init__ 基本参数的保存，参数验证
        build_model 构建模型
        init_placeholder 初始化变量
        build_encoder 初始化编码器
            build_single_cell
            build_decoder_cell
        init_optimizer 如果在训练模式下进行，则需要初始化优化器
        train 训练一个batch数据
        predict 预测一个batch数据
    '''

    def __init__(self,
                 input_vocab_size,# 输入词表大小
                 target_vocab_size,# 输出词表大小
                 batch_size=32,# 数据batch的大小
                 embedding_size=300,# 输入词表和输出词表的维数
                 mode='train',# train：代表训练模式，decode：代表预训练模式
                 hidden_units=256,# RNN中间层大小，encoder和decoder层size相同
                 depth=1,# encoder和decoder的RNN层
                 beam_width=0,# beamsearch的超参数，用于解码
                 cell_type='lstm',# RNN神经元的类型，lstm/gru
                 dropout=0.2,# 随机丢弃数据的比例，是要0到1之间
                 use_dropout=False,# 是否使用dropout
                 use_residual=False,# 是否使用residual（残差）
                 optimizer='adam',# 使用哪一个优化器
                 learning_rate=1e-3,# 学习率
                 min_learning_rate=1e-6,# 最小学习率
                 decay_steps=50000,# 训练步数
                 max_gradient_norm=5.0,# 梯度正则剪裁的系数
                 max_decode_step=None,# 最长decode长度，可以非常大
                 attention_type='Bahdanau',# 使用的attention类型
                 bidirectional=False,# 是否是双向encoder
                 time_major=False,# 是否在计算过程中使用时间作为主要批量数据
                 seed=0,# 一些层间操作的随机数
                 parallel_iterations=None,# 并行执行RNN个数
                 share_embedding=False,# 是否让encoder和decoder公用一个embedding
                 pretrained_embedding=False# 是否使用预训练的embedding
                ):
        self.input_vocab_size=input_vocab_size
        self.target_vocab_size=target_vocab_size
        self.batch_size=batch_size
        self.embedding_size=embedding_size
        self.hidden_units=hidden_units
        self.depth=depth
        self.cell_type=cell_type
        self.use_dropout=use_dropout
        self.use_residual=use_residual
        self.attention_type=attention_type
        self.mode=mode
        self.optimizer=optimizer
        self.learning_rate=learning_rate
        self.min_learning_rate=min_learning_rate
        self.decay_steps=decay_steps
        self.max_gradient_norm=max_gradient_norm
        self.keep_prob=1.0-dropout
        self.bidirectional=bidirectional
        self.seed=seed
        self.pretrained_embedding=pretrained_embedding

        if isinstance(parallel_iterations,int):
            self.parallel_iterations=parallel_iterations
        else:
            self.parallel_iterations=batch_size
        self.time_major=time_major
        self.share_embedding=share_embedding

        # 生成均匀分布的随机数，有四个参数，最小值，最大值，种子，类型
        self.initializer=tf.random_uniform_initializer(
            -0.05,0.05,dtype=tf.float32
        )
        #校验数据
        assert self.cell_type in ('gru','lstm'),'cell_type应该是GRU或者LSTM'

        if share_embedding:
            assert input_vocab_size==target_vocab_size,'如果share_embedding为true，那么两个vocab_size必须一样'

        assert  mode in ('train','decode'),'mode必须是train或者decode，而不是{}'.format(mode)

        assert dropout>=0.0 and dropout<1.0,'dropout必须大于等于0小于等于1'

        assert attention_type.lower() in ('bahdanau','luong'),'attention_type必须是bahdanau或者luong，而不是{}'.format(attention_type)

        assert beam_width<target_vocab_size,'beam_width {} 应该小于target_vocab_size{}'.format(beam_width,target_vocab_size)

        self.keep_prob_placeholder=tf.placeholder(
            tf.float32,
            shape=[],
            name='keep_prob'
        )

        self.global_step=tf.Variable(
            0,trainable=False,name='global_step'
        )

        self.use_beamsearch_decode=False
        self.beam_width=beam_width
        self.use_beamsearch_decode=True if self.beam_width>0 else False
        self.max_decode_step=max_decode_step

        assert self.optimizer.lower() in ('adadelta','adam','rmsprop','momentum','sgd'),'optimizer必须是下列之一：adadelta、adam、rmsprop、momentum、sgd'

        self.build_model()

    def build_model(self):
        '''
            1、初始化训练、预测所需要的变量
            2、构建编码器（encoder）build_encoder->encoder_cell->build_single_cell
            3、构建解码器（decoder）
            4、构建优化器（optimizer）
            5、保存
        :return:
        '''
        #1、初始化训练、预测所需要的变量
        self.init_placeholders()
        encoder_outputs,encoder_state=self.build_encoder()
        self.build_decoder(encoder_outputs,encoder_state)

        # 使用优化器
        if self.mode=='train':
            self.init_optimizer()

        self.saver=tf.train.Saver()

    def init_placeholders(self):
        '''初始化训练、预测所需要的变量'''
        self.add_loss=tf.placeholder(
            dtype=tf.float32,
            name='add_loss'
        )
        # 编码器的输入，shape=(n,None)中None为任意长度
        self.encoder_inputs=tf.placeholder(
            dtype=tf.int32,
            shape=(self.batch_size,None),
            name='encoder_inputs'
        )
        # 编码器的输入长度，shape=(n,)后整个张量为1维的
        self.encoder_inputs_length=tf.placeholder(
            dtype=tf.int32,
            shape=(self.batch_size,),
            name='encoder_inputs_length'
        )
        if self.mode=='train':
            # 解码器的输入
            self.decoder_inputs=tf.placeholder(
                dtype=tf.int32,
                shape=(self.batch_size,None),
                name='decoder_inputs'
            )
            # 解码器输入的rewards，用于强化学习
            self.rewards=tf.placeholder(
                dtype=tf.float32,
                shape=(self.batch_size,1),
                name='rewards'
            )
            # 解码器长度的输入
            self.decoder_inputs_length=tf.placeholder(
                dtype=tf.int32,
                shape=(self.batch_size,),
                name='decoder_inputs_length'
            )

            self.decoder_start_token=tf.ones(
                shape=(self.batch_size,1),
                dtype=tf.int32
            )*WordSequence.START

            #实际训练时解码器的输入，start_token+decoder_input
            self.decoder_inputs_train=tf.concat([
                self.decoder_start_token,self.decoder_inputs
            ],
                axis=1
            )

    def build_single_cell(self,n_hidden,use_residual):
        """
        构建一个单独的RNN cell
        :param n_hidden: 隐藏层神经元的数量
        :param use_residual: 是否使用残差网络
        :return:
        """
        if self.cell_type=='gru':
            cell_type=GRUCell
        else:
            cell_type=LSTMCell

        cell=cell_type(n_hidden)

        if self.use_dropout:
            cell=DropoutWrapper(
                cell,
                dtype=tf.float32,
                output_keep_prob=self.keep_prob_placeholder,
                seed=self.seed
            )
        if use_residual:
            cell=ResidualWrapper(cell)

        return cell

    def build_encoder_cell(self):
        '''
            构建单独的编码器，返回一个cell的list
        :return:
        '''
        return MultiRNNCell(
            [self.build_single_cell(
                self.hidden_units,
                use_residual=self.use_residual
            )
            for _ in range(self.depth)
            ]
        )

    def build_encoder(self):
        '''
            构建编码器
        :return:
        '''
        with tf.variable_scope('encoder'):
            encoder_cell=self.build_encoder_cell()
            with tf.device(_get_embed_device(self.input_vocab_size)):
                # 如果有预训练好的词表
                if self.pretrained_embedding:
                    #encoder_embeddings为词表
                    self.encoder_embeddings=tf.Variable(
                        tf.constant(
                            0.0,
                            shape=(self.input_vocab_size,self.embedding_size)
                        ),
                        trainable=True,
                        name='embeddings'
                    )
                    self.encoder_embeddings_placeholder=tf.placeholder(
                        tf.float32,
                        (self.input_vocab_size,self.embedding_size)
                    )
                    self.encoder_embeddings_init=self.encoder_embeddings.assign(
                        self.encoder_embeddings_placeholder
                    )
                else:
                    self.encoder_embeddings=tf.get_variable(
                        name='embedding',
                        shape=(self.input_vocab_size,self.embedding_size),
                        initializer=self.initializer,
                        dtype=tf.float32
                    )
            #encoder_inputs_embedded是lookup操作的结果
            self.encoder_inputs_embedded=tf.nn.embedding_lookup(
                params=self.encoder_embeddings,
                ids=self.encoder_inputs
            )
            #是否加残差网络
            if self.use_residual:
                self.encoder_inputs_embedded=layers.dense(self.encoder_inputs_embedded,
                                                          self.hidden_units,
                                                          use_bias=False,
                                                          name='encoder_residual_projection')
            inputs=self.encoder_inputs_embedded
            # 转置，这里的理解还不是很深刻，大致的意思是原本的inputs第一维是batch_size，第二维是句子中的单词序列，两者转置后
            # 根据RNN的输入方式就变为了每次输入的为batch_size个句子中第n个单词
            if self.time_major:
                inputs=tf.transpose(inputs,(1,0,2))
            #单向RNN
            if not self.bidirectional:
                # state为最后一层的输出结果
                (encoder_outputs,encoder_state)=tf.nn.dynamic_rnn(
                    cell=encoder_cell,
                    inputs=inputs,
                    sequence_length=self.encoder_inputs_length,
                    dtype=tf.float32,
                    time_major=self.time_major,
                    parallel_iterations=self.parallel_iterations,
                    swap_memory=True
                )
            #双向RNN
            else:
                encoder_cell_bw=self.build_encoder_cell()
                (
                    (encoder_fw_outputs,encoder_bw_outputs),
                    (encoder_fw_state,encoder_bw_state)
                )=tf.nn.bidirectional_dynamic_rnn(
                    cell_bw=encoder_cell_bw,
                    cell_fw=encoder_cell,
                    inputs=inputs,
                    sequence_length=self.encoder_inputs_length,
                    dtype=tf.float32,
                    time_major=self.time_major,
                    parallel_iterations=self.parallel_iterations,
                    swap_memory=True
                )
                encoder_outputs=tf.concat(
                    (encoder_fw_outputs,encoder_bw_outputs),2
                )
                encoder_state=[]
                for i in range(self.depth):
                    encoder_state.append(encoder_fw_state[i])
                    encoder_state.append(encoder_bw_state[i])
                encoder_state=tuple(encoder_state)

            return encoder_outputs,encoder_state

    # 构建解码器cell
    def build_decoder_cell(self,encoder_outputs,encoder_state):
        '''
        构建解码器的cell
        :param encoder_outputs:
        :param encoder_state:
        :return:
        '''
        encoder_inputs_length = self.encoder_inputs_length
        batch_size = self.batch_size

        if self.bidirectional:
            encoder_state = encoder_state[-self.depth:]

        if self.time_major:
            encoder_outputs = tf.transpose(encoder_outputs, (1, 0, 2))

        # 使用 BeamSearchDecoder 的时候，必须根据 beam_width 来成倍的扩大一些变量

        if self.use_beamsearch_decode:
            encoder_outputs = seq2seq.tile_batch(
                encoder_outputs, multiplier=self.beam_width)
            encoder_state = seq2seq.tile_batch(
                encoder_state, multiplier=self.beam_width)
            encoder_inputs_length = seq2seq.tile_batch(
                self.encoder_inputs_length, multiplier=self.beam_width)
            # 如果使用了 beamsearch 那么输入应该是 beam_width 倍于 batch_size 的
            batch_size *= self.beam_width

        # 下面是两种不同的 Attention 机制
        if self.attention_type.lower() == 'luong':
            # 'Luong' style attention: https://arxiv.org/abs/1508.04025
            self.attention_mechanism = LuongAttention(
                num_units=self.hidden_units,
                memory=encoder_outputs,
                memory_sequence_length=encoder_inputs_length
            )
        else:  # Default Bahdanau
            # 'Bahdanau' style attention: https://arxiv.org/abs/1409.0473
            self.attention_mechanism = BahdanauAttention(
                num_units=self.hidden_units,
                memory=encoder_outputs,
                memory_sequence_length=encoder_inputs_length
            )

        # Building decoder_cell
        cell = MultiRNNCell([
            self.build_single_cell(
                self.hidden_units,
                use_residual=self.use_residual
            )
            for _ in range(self.depth)
        ])

        # 在非训练（预测）模式，并且没开启 beamsearch 的时候，打开 attention 历史信息
        alignment_history = (
                self.mode != 'train' and not self.use_beamsearch_decode
        )

        def cell_input_fn(inputs, attention):
            """根据attn_input_feeding属性来判断是否在attention计算前进行一次投影计算
            """
            if not self.use_residual:
                return array_ops.concat([inputs, attention], -1)

            attn_projection = layers.Dense(self.hidden_units,
                                           dtype=tf.float32,
                                           use_bias=False,
                                           name='attention_cell_input_fn')
            return attn_projection(array_ops.concat([inputs, attention], -1))

        cell = AttentionWrapper(
            cell=cell,
            attention_mechanism=self.attention_mechanism,
            attention_layer_size=self.hidden_units,
            alignment_history=alignment_history,
            cell_input_fn=cell_input_fn,
            name='Attention_Wrapper')

        # 空状态
        decoder_initial_state = cell.zero_state(
            batch_size, tf.float32)

        # 传递encoder状态
        decoder_initial_state = decoder_initial_state.clone(
            cell_state=encoder_state)

        return cell, decoder_initial_state

    # 构建解码器
    def build_decoder(self,encoder_outputs,encoder_state):
        with tf.variable_scope('decoder') as decoder_scope:
            (
                self.decoder_cell,
                self.decoder_initial_state
            ) = self.build_decoder_cell(encoder_outputs, encoder_state)

            # 解码器embedding
            with tf.device(_get_embed_device(self.target_vocab_size)):
                if self.share_embedding:
                    self.decoder_embeddings = self.encoder_embeddings
                elif self.pretrained_embedding:

                    self.decoder_embeddings = tf.Variable(
                        tf.constant(
                            0.0,
                            shape=(self.target_vocab_size,
                                   self.embedding_size)
                        ),
                        trainable=True,
                        name='embeddings'
                    )
                    self.decoder_embeddings_placeholder = tf.placeholder(
                        tf.float32,
                        (self.target_vocab_size, self.embedding_size)
                    )
                    self.decoder_embeddings_init = \
                        self.decoder_embeddings.assign(
                            self.decoder_embeddings_placeholder)
                else:
                    self.decoder_embeddings = tf.get_variable(
                        name='embeddings',
                        shape=(self.target_vocab_size, self.embedding_size),
                        initializer=self.initializer,
                        dtype=tf.float32
                    )

            self.decoder_output_projection = layers.Dense(
                self.target_vocab_size,
                dtype=tf.float32,
                use_bias=False,
                name='decoder_output_projection'
            )

            if self.mode == 'train':
                self.decoder_inputs_embedded = tf.nn.embedding_lookup(
                    params=self.decoder_embeddings,
                    ids=self.decoder_inputs_train
                )
                inputs = self.decoder_inputs_embedded

                if self.time_major:
                    inputs = tf.transpose(inputs, (1, 0, 2))

                training_helper = seq2seq.TrainingHelper(
                    inputs=inputs,
                    sequence_length=self.decoder_inputs_length,
                    time_major=self.time_major,
                    name='training_helper'
                )

                # 训练的时候不在这里应用 output_layer
                # 因为这里会每个 time_step 的进行 output_layer 的投影计算，比较慢
                # 注意这个trick要成功必须设置 dynamic_decode 的 scope 参数
                training_decoder = seq2seq.BasicDecoder(
                    cell=self.decoder_cell,
                    helper=training_helper,
                    initial_state=self.decoder_initial_state,
                )

                # Maximum decoder time_steps in current batch
                max_decoder_length = tf.reduce_max(
                    self.decoder_inputs_length
                )


                (
                    outputs,
                    self.final_state, # contain attention
                    _ # self.final_sequence_lengths
                ) = seq2seq.dynamic_decode(
                    decoder=training_decoder,
                    output_time_major=self.time_major,
                    impute_finished=True,
                    maximum_iterations=max_decoder_length,
                    parallel_iterations=self.parallel_iterations,
                    swap_memory=True,
                    scope=decoder_scope
                )

                self.decoder_logits_train = self.decoder_output_projection(
                    outputs.rnn_output
                )

                # masks: masking for valid and padded time steps,
                # [batch_size, max_time_step + 1]
                self.masks = tf.sequence_mask(
                    lengths=self.decoder_inputs_length,
                    maxlen=max_decoder_length,
                    dtype=tf.float32, name='masks'
                )

                decoder_logits_train = self.decoder_logits_train
                if self.time_major:
                    decoder_logits_train = tf.transpose(decoder_logits_train,
                                                        (1, 0, 2))

                self.decoder_pred_train = tf.argmax(
                    decoder_logits_train, axis=-1,
                    name='decoder_pred_train')

                # 下面的一些变量用于特殊的学习训练
                # 自定义rewards，其实我这里是修改了masks
                # train_entropy = cross entropy
                self.train_entropy = \
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self.decoder_inputs,
                        logits=decoder_logits_train)

                self.masks_rewards = self.masks * self.rewards

                self.loss_rewards = seq2seq.sequence_loss(
                    logits=decoder_logits_train,
                    targets=self.decoder_inputs,
                    weights=self.masks_rewards,
                    average_across_timesteps=True,
                    average_across_batch=True,
                )

                self.loss = seq2seq.sequence_loss(
                    logits=decoder_logits_train,
                    targets=self.decoder_inputs,
                    weights=self.masks,
                    average_across_timesteps=True,
                    average_across_batch=True,
                )

                self.loss_add = self.loss + self.add_loss

            elif self.mode == 'decode':
                # 预测模式，非训练

                start_tokens = tf.tile(
                    [WordSequence.START],
                    [self.batch_size]
                )
                end_token = WordSequence.END

                def embed_and_input_proj(inputs):
                    """输入层的投影层wrapper
                    """
                    return tf.nn.embedding_lookup(
                        self.decoder_embeddings,
                        inputs
                    )

                if not self.use_beamsearch_decode:
                    # Helper to feed inputs for greedy decoding:
                    # uses the argmax of the output
                    decoding_helper = seq2seq.GreedyEmbeddingHelper(
                        start_tokens=start_tokens,
                        end_token=end_token,
                        embedding=embed_and_input_proj
                    )
                    # Basic decoder performs greedy decoding at each time step
                    # print("building greedy decoder..")
                    inference_decoder = seq2seq.BasicDecoder(
                        cell=self.decoder_cell,
                        helper=decoding_helper,
                        initial_state=self.decoder_initial_state,
                        output_layer=self.decoder_output_projection
                    )
                else:
                    # Beamsearch is used to approximately
                    # find the most likely translation
                    # print("building beamsearch decoder..")
                    inference_decoder = BeamSearchDecoder(
                        cell=self.decoder_cell,
                        embedding=embed_and_input_proj,
                        start_tokens=start_tokens,
                        end_token=end_token,
                        initial_state=self.decoder_initial_state,
                        beam_width=self.beam_width,
                        output_layer=self.decoder_output_projection,
                    )

                if self.max_decode_step is not None:
                    max_decode_step = self.max_decode_step
                else:
                    # 默认 4 倍输入长度的输出解码
                    max_decode_step = tf.round(tf.reduce_max(
                        self.encoder_inputs_length) * 4)

                (
                    self.decoder_outputs_decode,
                    self.final_state,
                    _ # self.decoder_outputs_length_decode
                ) = (seq2seq.dynamic_decode(
                    decoder=inference_decoder,
                    output_time_major=self.time_major,
                    # impute_finished=True,	# error occurs
                    maximum_iterations=max_decode_step,
                    parallel_iterations=self.parallel_iterations,
                    swap_memory=True,
                    scope=decoder_scope
                ))

                if not self.use_beamsearch_decode:

                    dod = self.decoder_outputs_decode
                    self.decoder_pred_decode = dod.sample_id

                    if self.time_major:
                        self.decoder_pred_decode = tf.transpose(
                            self.decoder_pred_decode, (1, 0))

                else:
                    self.decoder_pred_decode = \
                        self.decoder_outputs_decode.predicted_ids

                    if self.time_major:
                        self.decoder_pred_decode = tf.transpose(
                            self.decoder_pred_decode, (1, 0, 2))

                    self.decoder_pred_decode = tf.transpose(
                        self.decoder_pred_decode,
                        perm=[0, 2, 1])
                    dod = self.decoder_outputs_decode
                    self.beam_prob = dod.beam_search_decoder_output.scores

    # 模型保存
    def save(self,sess,save_path='model.ckpt'):
        '''
        在TensorFlow里，保存模型的格式有两种：
        ckpt:训练模型后的保存，这里面会保存所有的训练参数，文件相对来讲比较大，可以用来进行模型的恢复和加载
        pb:用于模型最后的线上部署，这里面的线上部署指的是TensorFlow serving进行模型的发布，一般发布成grpc形式的接口
        :param sess:
        :param save_path:
        :return:
        '''
        self.saver.save(sess,save_path=save_path)
    # 模型加载
    def load(self,sess,save_path='model.ckpt'):
        print('try load model from',save_path)
        self.saver.restore(sess,save_path)

    # 构建优化器
    def init_optimizer(self):
        '''
        sgd,adadelta,adam,rmsprop,momentum
        :return:
        '''
        # 定义学习率
        learning_rate=tf.train.polynomial_decay(
            self.learning_rate,
            self.global_step,
            self.decay_steps,
            self.min_learning_rate,
            power=0.5
        )
        self.current_learning_rate=learning_rate

        # 返回所有需要训练的参数列表
        trainable_params=tf.trainable_variables()

        # 设置优化器
        if self.optimizer.lower()=='adadelta':
            self.opt=tf.train.AdadeltaOptimizer(
                learning_rate=learning_rate
            )
        elif self.optimizer.lower()=='adam':
            self.opt=tf.train.AdamOptimizer(
                learning_rate=learning_rate
            )
        elif self.optimizer.lower()=='rmsprop':
            self.opt=tf.train.RMSPropOptimizer(
                learning_rate=learning_rate
            )
        elif self.optimizer.lower()=='momentum':
            self.opt=tf.train.MomentumOptimizer(
                learning_rate=learning_rate
            )
        elif self.optimizer.lower()=='sgd':
            self.opt=tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate
            )

        gradients=tf.gradients(self.loss,trainable_params)
        # 梯度裁剪
        clip_gradients,_=tf.clip_by_global_norm(
            gradients,self.max_gradient_norm
        )
        # 更新model
        self.updates=self.opt.apply_gradients(
            zip(clip_gradients,trainable_params),
            global_step=self.global_step
        )

        gradients=tf.gradients(self.loss_rewards,trainable_params)
        clip_gradients,_=tf.clip_by_global_norm(
            gradients,self.max_gradient_norm
        )
        self.updates_rewards=self.opt.apply_gradients(
            zip(clip_gradients,trainable_params),
            global_step=self.global_step
        )

        # 添加self.loss_add
        gradients=tf.gradients(self.add_loss,trainable_params)
        clip_gradients,_=tf.clip_by_global_norm(
            gradients,self.max_gradient_norm
        )
        self.updates_add=self.opt.apply_gradients(
            zip(clip_gradients,trainable_params),
            global_step=self.global_step
        )

    # 检查输入
    def check_feeds(self,encoder_inputs,encoder_inputs_length,
                    decoder_inputs,decoder_inputs_length,decode):
        '''
        :param encoder_inputs: 一个整型的二维矩阵，[batch_size,max_source_time_steps]
        :param encoder_inputs_length: [batch_size],里面为encoder句子的真实长度
        :param decoder_inputs: 一个整型的二维矩阵，[batch_size,max_source_time_steps]
        :param decoder_inputs_length: [batch_size],里面为decoder句子的真实长度
        :param decode:是训练模式(False)还是预测模式(True)
        :return: TensorFlow所需要的input_feed
        '''
        input_batch_size=encoder_inputs.shape[0]
        if input_batch_size!=encoder_inputs_length.shape[0]:
            raise ValueError(
                'encoder_inputs和encoder_inputs_length的第一维度必须一致'
                '这个维度是batch_size,%d != %d'%(
                    input_batch_size,encoder_inputs_length.shape[0]
                )
            )

        if not decode:
            target_batch_size=decoder_inputs.shape[0]
            if target_batch_size!=input_batch_size:
                raise ValueError(
                    'encoder_inputs和decoder_inputs的第一个维度必须一致'
                    '这个维度是batch_size,%d != %d'%(
                        input_batch_size,target_batch_size
                    )
                )
            if target_batch_size!=decoder_inputs_length.shape[0]:
                raise ValueError(
                    'encoder_inputs和decoder_inputs的第一个维度必须一致'
                    '这个维度是batch_size,%d != %d' % (
                        target_batch_size, decoder_inputs_length.shape[0]
                    )
                )

        input_feed={}
        input_feed[self.encoder_inputs.name]=encoder_inputs
        input_feed[self.encoder_inputs_length.name]=encoder_inputs_length

        if not decode:
            input_feed[self.decoder_inputs.name]=decoder_inputs
            input_feed[self.decoder_inputs_length.name]=decoder_inputs_length

        return input_feed

    def train(self,sess,encoder_inputs,encoder_inputs_length,
              decoder_inputs,decoder_inputs_length,
              rewards=None,return_lr=False,
              loss_only=False,add_loss=None):
        # 输入
        input_feed=self.check_feeds(
            encoder_inputs,encoder_inputs_length,
            decoder_inputs,decoder_inputs_length,
            False
        )
        # 设置dropout
        input_feed[self.keep_prob_placeholder.name]=self.keep_prob

        if loss_only:
            return sess.run(self.loss,input_feed)

        if add_loss is not None:
            input_feed[self.add_loss.name]=add_loss
            output_feed=[
                self.updates_add,self.add_loss,self.current_learning_rate
            ]
            _,cost,lr=sess.run(output_feed,input_feed)

            if return_lr:
                return cost,lr
            return cost

        if rewards is not None:
            input_feed[self.rewards.name]=rewards
            output_feed=[
                self.updates_rewards,self.loss_rewards,
                self.current_learning_rate
            ]
            _,cost,lr=sess.run(output_feed,input_feed)

            if return_lr:
                return cost,lr
            return cost

        output_feed=[
            self.updates,self.loss,
            self.current_learning_rate
        ]
        _, cost, lr = sess.run(output_feed, input_feed)

        if return_lr:
            return cost,lr
        return cost

    # 预测
    def predict(self,sess,encoder_inputs,encoder_inputs_length,attention=False):
        input_feed=self.check_feeds(encoder_inputs,encoder_inputs_length,None,None,True)

        input_feed[self.keep_prob_placeholder.name]=1.0
        if attention:
            assert not self.use_beamsearch_decode,'Attention模式'
            pred,atten=sess.run([
                self.decoder_pred_decode,
                self.final_state.aligment_history.stack()
            ],input_feed
            )
            return pred,atten

        if self.use_beamsearch_decode:
            pred,beam_prob=sess.run([
                self.decoder_pred_decode,
                self.beam_width
            ],input_feed)

            beam_prob=np.mean(beam_prob,axis=1)
            pred=pred[0]
            return pred

        pred=sess.run([
            self.decoder_pred_decode
        ],input_feed)

