import sys
import pickle
import numpy as np
import tensorflow as tf

def test(params):
    from sequence_to_sequence import SequenceToSequence
    from data_utils import batch_flow

    x_data,_=pickle.load(open('./data/chatbot.pkl','rb'))
    ws=pickle.load(open('./data/ws.pkl','rb'))

    for x in x_data[:5]:
        print(' '.join(x))

    config = tf.ConfigProto(
        device_count={'CPU':1,'GPU':0},
        allow_soft_placement=True,
        log_device_placement=False
    )

    save_path='./model/s2s_chatbot_anti.ckpt'
    model_pred=SequenceToSequence(
        input_vocab_size=len(ws),
        target_vocab_size=len(ws),
        batch_size=1,
        mode='decode',
        beam_width=0,
        **params
    )
    print('----------------')
    init=tf.global_variables_initializer()

    with tf.Session(config=config) as sess:
        sess.run(init)
        model_pred.load(sess,save_path)
        while True:
            user_text=input('请输入您的句子：')
            if user_text in ('exit','quit'):
                exit(0)
            x_test=[list(user_text.lower())]
            bar=batch_flow([x_test],ws,1)
            x,xl=next(bar)
            x=np.flip(x,axis=1)

            print(x,xl)
            pred=model_pred.predict(
                sess,
                np.array(x),
                np.array(xl)
            )
            print(ws.inverse_transform(x[0]))

            for p in pred:
                answer=ws.inverse_transform(p)
                print(answer)


def main():
    import json
    test(json.load(open('params.json')))

if __name__ == '__main__':
    main()