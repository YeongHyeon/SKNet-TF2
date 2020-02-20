import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, MaxPool2D
from tensorflow.keras import Model

class CNN(object):

    def __init__(self, height, width, channel, num_class, leaning_rate=1e-3, ckpt_dir='./Checkpoint'):

        print("\nInitializing Neural Network...")
        self.height, self.width, self.channel = height, width, channel
        self.num_class, self.k_size = num_class, 3
        self.leaning_rate = leaning_rate
        self.ckpt_dir = ckpt_dir

        self.model = SKNet(num_class=self.num_class)

        self.model(tf.zeros([1, self.height, self.width, self.channel]), training=False, verbose=True)

        self.optimizer = tf.optimizers.Adam(self.leaning_rate)

        self.summary_writer = tf.summary.create_file_writer(self.ckpt_dir)

    def step(self, x, y, iteration=0, train=False):

        with tf.GradientTape() as tape:
            logits = self.model(x, training=train)
            smce = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
            loss = tf.math.reduce_mean(smce)

        score = tf.nn.softmax(logits)
        pred = tf.argmax(score, 1)
        correct_pred = tf.equal(pred, tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        if(train):
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            with self.summary_writer.as_default():
                tf.summary.scalar('CNN/loss', loss, step=iteration)
                tf.summary.scalar('CNN/accuracy', accuracy, step=iteration)

        return loss, accuracy, score

    def save_params(self):
        self.model.save_weights("%s/model.h5" %(self.ckpt_dir))

    def load_params(self):
        self.model.load_weights("%s/model.h5" %(self.ckpt_dir))

class SKNet(Model):

    def __init__(self, num_class):
        super(SKNet, self).__init__()

        self.num_class = num_class

        # Block 1
        self.conv1_a = Conv2D(filters=16, kernel_size=3, strides=1, padding="SAME")
        self.bn1_a = BatchNormalization()
        self.conv1_b = Conv2D(filters=16, kernel_size=5, strides=1, padding="SAME")
        self.bn1_b = BatchNormalization()

        self.fc1 = Dense(8, activation=None)
        self.bn1_fc = BatchNormalization()

        self.fc1_a = Dense(16, activation=None)


        # Block 2
        self.conv2_a = Conv2D(filters=32, kernel_size=3, strides=1, padding="SAME")
        self.bn2_a = BatchNormalization()
        self.conv2_b = Conv2D(filters=32, kernel_size=5, strides=1, padding="SAME")
        self.bn2_b = BatchNormalization()

        self.fc2 = Dense(16, activation=None)
        self.bn2_fc = BatchNormalization()

        self.fc2_a = Dense(32, activation=None)


        # Block 2
        self.conv3_a = Conv2D(filters=64, kernel_size=3, strides=1, padding="SAME")
        self.bn3_a = BatchNormalization()
        self.conv3_b = Conv2D(filters=64, kernel_size=5, strides=1, padding="SAME")
        self.bn3_b = BatchNormalization()

        self.fc3 = Dense(32, activation=None)
        self.bn3_fc = BatchNormalization()

        self.fc3_a = Dense(64, activation=None)


        # FC
        self.fc_out = Dense(self.num_class, activation=None)

        self.maxpool = MaxPool2D(pool_size=(2, 2))

    def call(self, x, training=False, verbose=False):

        if(verbose): print(x.shape)

        """ Conv-1 """
        # Split-1
        u1_a = tf.keras.activations.relu(self.bn1_a(self.conv1_a(x), training=training))
        u1_b = tf.keras.activations.relu(self.bn1_b(self.conv1_b(x), training=training))

        # Fuse-1
        u1 = u1_a + u1_b
        s1 = tf.math.reduce_sum(u1, axis=(1, 2))
        z1 = tf.keras.activations.relu(self.bn1_fc(self.fc1(s1), training=training))

        # Select-1
        a1 = tf.keras.activations.softmax(self.fc1_a(z1))
        a1 = tf.expand_dims(a1, 1)
        a1 = tf.expand_dims(a1, 1)
        b1 = 1 - a1
        v1 = (u1_a * a1) + (u1_b * b1)
        if(verbose): print(v1.shape)
        p1 = self.maxpool(v1)
        if(verbose): print(p1.shape)

        """ Conv-2 """
        # Split-2
        u2_a = tf.keras.activations.relu(self.bn2_a(self.conv2_a(p1), training=training))
        u2_b = tf.keras.activations.relu(self.bn2_b(self.conv2_b(p1), training=training))

        # Fuse-2
        u2 = u2_a + u2_b
        s2 = tf.math.reduce_sum(u2, axis=(1, 2))
        z2 = tf.keras.activations.relu(self.bn2_fc(self.fc2(s2), training=training))

        # Select-2
        a2 = tf.keras.activations.softmax(self.fc2_a(z2))
        a2 = tf.expand_dims(a2, 1)
        a2 = tf.expand_dims(a2, 1)
        b2 = 1 - a2
        v2 = (u2_a * a2) + (u2_b * b2)
        if(verbose): print(v2.shape)
        p2 = self.maxpool(v2)
        if(verbose): print(p2.shape)

        """ Conv-3 """
        # Split-3
        u3_a = tf.keras.activations.relu(self.bn3_a(self.conv3_a(p2), training=training))
        u3_b = tf.keras.activations.relu(self.bn3_b(self.conv3_b(p2), training=training))

        # Fuse-3
        u3 = u3_a + u3_b
        s3 = tf.math.reduce_sum(u3, axis=(1, 2))
        z3 = tf.keras.activations.relu(self.bn3_fc(self.fc3(s3), training=training))

        # Select-3
        a3 = tf.keras.activations.softmax(self.fc3_a(z3))
        a3 = tf.expand_dims(a3, 1)
        a3 = tf.expand_dims(a3, 1)
        b3 = 1 - a3
        v3 = (u3_a * a3) + (u3_b * b3)
        if(verbose): print(v3.shape)

        gap = tf.math.reduce_sum(v3, axis=(1, 2))
        if(verbose): print(gap.shape)
        out = self.fc_out(gap)
        if(verbose): print(out.shape)

        return out
