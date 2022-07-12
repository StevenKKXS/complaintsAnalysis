from LSTM_keras import *


class LSTM_model:
    def __init__(self):
        # load model
        model_zip = load_model('models/LSTM')

        # 解压model_zip
        self.m, self.token, self.label_m, self.his = model_zip.values()

        self.reverse_label_map()

    def reverse_label_map(self):
        tmp = {}
        for key in self.label_m:
            tmp[self.label_m[key]] = key
        self.label_m = tmp

    def predict(self, txt):
        txt = cut_text(txt)
        # print(txt)

        X = self.token.texts_to_sequences([txt])

        X = pad_sequences(X, maxlen=150)
        # X = tf.squeeze(X)
        res = self.m.predict(X)
        res = res[0]

        best_class = 0
        for i in range(len(res)):
            if res[i] > res[best_class]:
                best_class = i
        res = self.label_m[best_class]
        return res
