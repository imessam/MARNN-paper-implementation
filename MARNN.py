
import gc
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class MARNN:

    def __init__(self):

        text_data, audio_data, video_data, labels, mask = self.preprocess_data()

        self.train_text,  self.test_text = text_data
        self.train_audio, self.test_audio = audio_data
        self.train_video,  self.test_video = video_data
        self.train_label, self.test_label = labels
        self.train_mask,  self.test_mask = mask
        
        ########### Input Layer ############

        self.in_text = Input(shape=(self.train_text.shape[1], self.train_text.shape[2]))
        self.in_audio = Input(shape=(self.train_audio.shape[1], self.train_audio.shape[2]))
        self.in_video = Input(shape=(self.train_video.shape[1], self.train_video.shape[2]))

        ########### Masking Layer ############


        self.masked_text = Masking(mask_value=0)(self.in_text)
        self.masked_audio = Masking(mask_value=0)(self.in_audio)
        self.masked_video = Masking(mask_value=0)(self.in_video)

        self.dropDense = 0.2

    def preprocess_data(self):

        # Retreive Data
        (train_text, train_label, test_text, test_label, max_utt_len, train_len, test_len) = pickle.load(
            open('./input/text.pickle', 'rb'))
        (train_audio, _, test_audio, _, _, _, _) = pickle.load(open('./input/audio.pickle', 'rb'))
        (train_video, _, test_video, _, _, _, _) = pickle.load(open('./input/video.pickle', 'rb'))
        #print(train_label.shape)
        train_label, test_label = self.create_one_hot_labels(train_label.astype('int'), test_label.astype('int'))
        #print(train_label.shape)

        train_mask, test_mask = self.create_mask(train_text, test_text, train_len, test_len)

        
        text_data = [train_text, test_text]
        audio_data = [train_audio,  test_audio]
        video_data = [train_video,  test_video]
        labels = [train_label,  test_label]
        mask = [train_mask,  test_mask]


        return text_data, audio_data, video_data, labels, mask

    def create_one_hot_labels(self, train_label, test_label):  # by3ml two columns wa7ed lel zeroes we wa7ed lel ones
        """
        # Arguments
            train and test labels (2D matrices)

        # Returns
            one hot encoded train and test labels (3D matrices)
        """

        maxlen = int(max(train_label.max(), test_label.max()))

        train = np.zeros((train_label.shape[0], train_label.shape[1], maxlen + 1))
        test = np.zeros((test_label.shape[0], test_label.shape[1], maxlen + 1))

        for i in range(train_label.shape[0]):
            for j in range(train_label.shape[1]):
                train[i, j, train_label[i, j]] = 1

        for i in range(test_label.shape[0]):
            for j in range(test_label.shape[1]):
                test[i, j, test_label[i, j]] = 1

        return train, test

    def create_mask(self, train_data, test_data, train_length, test_length):
        """
        # Arguments
            train, test data (any one modality (text, audio or video)), utterance lengths in train, test videos

        # Returns
            mask for train and test data
        """

        train_mask = np.zeros((train_data.shape[0], train_data.shape[1]), dtype='float')
        for i in range(len(train_length)):
            train_mask[i, :train_length[i]] = 1.0

        test_mask = np.zeros((test_data.shape[0], test_data.shape[1]), dtype='float')
        for i in range(len(test_length)):
            test_mask[i, :test_length[i]] = 1.0

        return train_mask, test_mask

    def calc_test_result(self, result, test_label, test_mask, print_detailed_results=False):
        """
        # Arguments
            predicted test labels, gold test labels and test mask

        # Returns
            accuracy of the predicted labels
        """

        true_label = []
        predicted_label = []

        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if test_mask[i, j] == 1:
                    true_label.append(np.argmax(test_label[i, j]))
                    predicted_label.append(np.argmax(result[i, j]))

        if print_detailed_results:
            print("Confusion Matrix :")
            print(confusion_matrix(true_label, predicted_label))
            print("Classification Report :")
            print(classification_report(true_label, predicted_label))
        print("Accuracy ", accuracy_score(true_label, predicted_label))
        return accuracy_score(true_label, predicted_label)

    def ScaledDotProductAttentionForMultimodalData(self, masked_text, masked_audio, masked_video):
        
        masked_text=tf.keras.layers.BatchNormalization(trainable=True, epsilon=1e-9, momentum=0.99)(masked_text)
        masked_audio=tf.keras.layers.BatchNormalization(trainable=True, epsilon=1e-9, momentum=0.99)(masked_audio)
        masked_video==tf.keras.layers.BatchNormalization(trainable=True, epsilon=1e-9, momentum=0.99)(masked_video)

        d_t = tf.keras.layers.Dense(100, activation=None)(masked_text)
        d_a = tf.keras.layers.Dense(100, activation=None)(masked_audio)
        d_v = tf.keras.layers.Dense(100, activation=None)(masked_video)
        D = concatenate([d_v, d_a, d_t])

        utt = D.shape[1]
        d = 100
        d_dash = 400
        D = tf.reshape(D, (-1, utt, d, 3))
        # print(D.shape)
        r = 150  # for multihead attention

        # Treating the term "tanh(Wf.D)/sqrt(d)" stated in the paper as a neural network for each utterance "u" with
        # two layers, input layer x = D[u] with dimension d=3x100, output layer with dimension d'=3x400 and a tanh
        # activation funtion which results to a neural network weights Wf of dimensions dxd' =100x400 like what
        # stated in the paper.

        dense1 = TimeDistributed(Dense(d_dash, activation='tanh'))
        dropout1 = Dropout(0.2)

        # Then we treat the output of the previous neural network "tanh(Wf.D)/sqrt(d)" with "wf" as the term
        # "Softmax(tanh(Wf.D).wf/sqrt(d))"stated in the paper as another neural network for each utterance with two
        # layers, input layer x= tanh(Wf.D)/sqrt(d)[u] with dimension d'=3x400, output layer with dimension =3x150
        # and a softmax activation function which results to a neural network weights wf of dimension d'xr = 400x150
        # like in the paper.

        dense2 = TimeDistributed(Dense(r, activation='softmax'))
        dropout2 = Dropout(0.2)

        DT = tf.reshape(D, (
            -1, D.shape[1], 3, d))  # Reshaping "D" to have a shape of (50,3,400) for appropriate dense layer shapes.
        temp = []
        for i in range(utt):
            z = dropout1(dense1(
                DT[:, i, :, :]))  # Inputting each uttereance tri-modal features to the same neural network and then
            # concatenate their outputs
            temp.append(z)

        zTAV = tf.transpose(tf.stack(temp),
                            perm=[1, 0, 2, 3])  # Concatenating the output of the neural network for each utterance and
        # transposing them to have a shape of (50,3,400)
        aTAV = zTAV / np.sqrt(d)
        #print(f"aTAV shape {aTAV.shape}")

        ###########Multi-head Attention###########

        dense2(aTAV[:, 0, :, :])

        # Extracting "wf" .
        wf = dense2.weights[0]
        #print(f"wf shape {wf.shape}")

        # Create "wf transpose".
        wfT = tf.reshape(wf, (wf.shape[1], wf.shape[0]))
        #print(f"wfT shape {wfT.shape}")

        # Multiplying "wf" with "wf transpose" and summing their product to get "wa" .
        wa = tf.math.reduce_sum(tf.matmul(wf, wfT), 1)
        #print(f"wa shape {wa.shape}")

        # Reshaping "wa" to have a shape of (400,1) instead of (400,)
        wa = tf.reshape(wa, (wa.shape[0], 1))
        #print(f"wa shape {wa.shape}")

        # Multiplying "wa" with the output of "tanh(Wf.D)/sqrt(d)" for each utterance
        temp = []
        for i in range(utt):
            z = tf.matmul(aTAV[:, i, :, :], wa)  # Multipltying each uttereance tri-modal features with "wa" and then
            # concatenate their outputs
            z = dropout2(Softmax()(z))
            temp.append(z)

        aTAV2 = tf.transpose(tf.stack(temp),
                             perm=[1, 0, 2, 3])  # Concatenating the output of the multiplication for each utterance and
        # transposing them to have a shape of (50,3,1)
        #print(f"aTAV2 shape : {aTAV2.shape}")

        AD = tf.matmul(D,
                       aTAV2)  # Multiply the attention weight with the input features to get attention scores like
        # in the paper
        #print(f"AD shape : {AD.shape}")

        AD = tf.reshape(AD, (-1, AD.shape[1], AD.shape[2]))

        return AD

    def AttentionBasedGRU(self, AD):

        H = Bidirectional(GRU(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(AD)
        H = tf.reshape(H, (-1, H.shape[2], H.shape[1]))
        # print(f" H shape : {H.shape}")

        Pt = Dropout(self.dropDense)(TimeDistributed(Dense(63, activation='tanh'))(H))
        # print(f" Pt shape : {Pt.shape}")

        alpha_t = Dropout(self.dropDense)(
            TimeDistributed(Dense(1, activation='softmax'))(tf.reshape(Pt, (-1, Pt.shape[2], Pt.shape[1]))))
        # print(f" alpha_t shape : {alpha_t.shape}")

        r = tf.matmul(H, alpha_t)
        # print(f" r shape : {r.shape}")

        temp1 = tf.add(r, H)
        # print(f" temp1 shape : {temp1.shape}")

        Hstar = tf.keras.activations.tanh(temp1)
        # print(f" Hstar shape : {Hstar.shape}")

        z_t = TimeDistributed(Dense(2, activation='softmax'))(tf.reshape(Hstar, (-1, Hstar.shape[2], Hstar.shape[1])))
        # print(f" z_t shape : {z_t.shape}")

        return z_t

    def train(self, runs, lr):

        accuracy = []

        for j in range(runs):
            np.random.seed(j + 1)
            tf.random.set_seed(j + 1)

            # compile model #
            model = Model(inputs=([self.in_text, self.in_audio, self.in_video]), outputs=self.AttentionBasedGRU(
                self.ScaledDotProductAttentionForMultimodalData(self.masked_text, self.masked_audio,
                                                                self.masked_video)))
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr)
                          , loss='categorical_crossentropy',
                          metrics=['accuracy'])

            # train model #
            history = model.fit([self.train_text, self.train_audio, self.train_video], self.train_label,
                                epochs=100,
                                batch_size=20,
                                shuffle=True,
                                validation_split=0.2,
                                verbose=1)

            # test results #
            test_predictions = model.predict([self.test_text, self.test_audio, self.test_video])
            test_accuracy = self.calc_test_result(test_predictions, self.test_label, self.test_mask)
            accuracy.append(test_accuracy)
            # release gpu memory #
            K.clear_session()
            del model, history
            gc.collect()

        # summarize test results #

        avg_accuracy = sum(accuracy) / len(accuracy)
        max_seed = np.argmax(accuracy)
        max_accuracy = accuracy[max_seed]

        print('Avg Test Accuracy:', '{0:.4f}'.format(avg_accuracy), '|| Max Test Accuracy:',
              '{0:.4f}'.format(max_accuracy), ' || Max Seed:', '{0:.4f}'.format(max_seed + 1))
        print('-' * 55)
        return (self.test_label,test_predictions)


if __name__ == "__main__":
    marnn = MARNN()
    marnn.train(3, 0.0001)
